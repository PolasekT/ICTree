# -*- coding: utf-8 -*-

"""
Dataset featurization and data preparation utilities.
"""

import itertools
import os
import re
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sklearn.preprocessing as skp
import sklearn.utils as sku

from perceptree.common.cache import Cache
from perceptree.common.cache import deep_copy_dict
from perceptree.common.cache import update_dict_recursively
from perceptree.common.configuration import Config
from perceptree.common.configuration import Configurable
from perceptree.common.logger import Logger
from perceptree.common.logger import ParsingBar
from perceptree.common.util import dict_of_lists
from perceptree.common.util import numpy_op_or_zero
from perceptree.common.util import numpy_zero
from perceptree.common.util import parse_bool_string
from perceptree.common.util import reshape_scalar
from perceptree.common.util import numpy_array_to_tuple_numpy
from perceptree.common.util import tuple_array_to_numpy
from perceptree.common.util import recurse_dict
from perceptree.common.util import remove_nan_inf
from perceptree.common.processing import Scaler
from perceptree.data.loader import BaseDataLoader
from perceptree.data.loader import DataLoader
from perceptree.data.treeio import TreeFile
from perceptree.data.treeio import TreeImage
from perceptree.data.treeio import TreeStatistic

import matplotlib.pyplot as plt
import seaborn as sns


class DataFeaturizer(Logger, Configurable):
    """
    Dataset featurization helper.
    """

    COMMAND_NAME = "Featurize"
    """ Name of this command, used for configuration. """

    def __init__(self, config: Config):
        super().__init__(config=config)
        self._set_instance()

        self.__l.info("Initializing data featurization system...")

        self._data_loader = self.get_instance(DataLoader)
        self._feature_cache = Cache()

    @staticmethod
    def _parse_data_split(split: str) -> Tuple[Union[float, List[int]]]:
        """ Parse input TRAIN/VALIDATION/TEST string. """

        def parse_spec(s: str):
            if s.find(",") >= 0:
                return [ int(v) for v in s.split(",") ]
            else:
                return float(s)

        values = [ parse_spec(v) for v in split.split("/") ]
        missing = 3 - len(values)
        values = values + [ 0.0 for _ in range(missing) ]

        if np.sum([ v for v in values if isinstance(v, float) ]) != 1.0:
            raise ValueError(f"Data split values (\"{split}\") should sum to 1.0!")

        return tuple(values)

    @staticmethod
    def _parse_cross_validation_split(cross_validation_split: str, default_seed: int = 42) -> Tuple[int]:
        """ Parse input ID/MAX[/SEED] string. """
        values = [ int(v) for v in cross_validation_split.split("/") ]
        if len(values) < 3:
            values.append(default_seed)
        if len(values) != 3 or values[0] <= 0 or values[0] > values[1] or values[1] < 1:
            raise ValueError(f"Cross validation split value (\"{cross_validation_split}\") is invalid!")

        return tuple(values)

    @classmethod
    def register_options(cls, parser: Config.Parser):
        """ Register configuration options for this class. """

        option_name = cls._add_config_parameter("feature_cache_location")
        parser.add_argument("--feature-cache-location",
                            action="store",
                            default="", type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Base path used for caching of features.")

        option_name = cls._add_config_parameter("data_split")
        parser.add_argument("--data-split",
                            action="store",
                            default=(0.8, 0.1, 0.1), type=DataFeaturizer._parse_data_split,
                            metavar=("TRAIN/VALID/TEST|<SPEC>/<SPEC>/<SPEC>"),
                            dest=option_name,
                            help="Specify percentage distribution into train, "
                                 "validation and testing data. Total value should "
                                 "add up to one - e.g. 0.8/0.1/0.1 which is the default."
                                 "Alternatively, each specification may be a comma "
                                 "separated list of tree IDs instead. In this case, the "
                                 "percentage values are related to the complete set of "
                                 "tree IDs, without the ones already used.")

        option_name = cls._add_config_parameter("cross_validation_split")
        parser.add_argument("--cross-validation-split",
                            action="store",
                            default=None, type=DataFeaturizer._parse_cross_validation_split,
                            metavar=("ID/MAX[/SEED]"),
                            dest=option_name,
                            help="Use randomized cross-validation protocol for splitting "
                                 "the loaded data. First number is the current run index, "
                                 "second is maximum (e.g. 1/10 to 10/10). Last number is "
                                 "optional and specifies seed to use. If left unset the "
                                 "default seed will be used and results will always be the "
                                 "same!")

        option_name = cls._add_config_parameter("debug_calculate_features")
        parser.add_argument("--debug-calculate-features",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Calculate features for debugging purposes?")

        option_name = cls._add_config_parameter("debug_calculate_views")
        parser.add_argument("--debug-calculate-views",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Calculate views for debugging purposes?")

        option_name = cls._add_config_parameter("debug_calculate_scores")
        parser.add_argument("--debug-calculate-scores",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Calculate scores for debugging purposes?")

        option_name = cls._add_config_parameter("debug_display_meta_features")
        parser.add_argument("--debug-display-meta-features",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Display debug information about meta features?")

        option_name = cls._add_config_parameter("debug_display_hist_features")
        parser.add_argument("--debug-display-hist-features",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Display debug information about histogram features?")

        option_name = cls._add_config_parameter("debug_display_stats_features")
        parser.add_argument("--debug-display-stats-features",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Display debug information about statistic features?")

        option_name = cls._add_config_parameter("debug_display_image_features")
        parser.add_argument("--debug-display-image-features",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Display debug information about image features?")

        option_name = cls._add_config_parameter("debug_display_others_features")
        parser.add_argument("--debug-display-others-features",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Display debug information about others features?")

        option_name = cls._add_config_parameter("debug_display_views_features")
        parser.add_argument("--debug-display-views-features",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Display debug information about views features?")

        option_name = cls._add_config_parameter("debug_display_view_choices")
        parser.add_argument("--debug-display-view-choices",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Display debug information about tree-view choices?")

        option_name = cls._add_config_parameter("debug_display_scores")
        parser.add_argument("--debug-display-scores",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Display debug information about scores?")

    def _generate_cache_version(self, feature_name: str,
                                dataset_version: str,
                                *args, **kwargs) -> dict:
        """ Generate version information dictionary for given feature parameters. """

        return {
            "feature_name": feature_name,
            "dataset_version": dataset_version,
            "arguments": self.config.subcommand_arguments(self.COMMAND_NAME),
            "args": args,
            "kwargs": kwargs
        }

    def _is_valid_cache_record(self, record: any,
                               feature_name: str,
                               dataset_version: str,
                               *args, **kwargs) -> bool:
        """ Check whether given cache record contains valid feature data. """

        if record is None or not isinstance(record, dict):
            return False

        if "version" not in record or "data" not in record:
            return False

        current_version = self._generate_cache_version(
            feature_name=feature_name,
            dataset_version=dataset_version,
            *args, **kwargs
        )

        def versions_equal(first: dict, second: dict) -> bool:
            """ Check if given two versions are equal. """
            try:
                for key, v1, v2 in recurse_dict([ first, second ],
                                                raise_unaligned=True,
                                                only_endpoints=True):
                    if np.all(v1 != v2):
                        return False
                return True
            except:
                return False

        return versions_equal(record["version"], current_version)

    def _check_load_cached(self, cache_location: str, feature_path: str,
                           initializer: Callable, *args, **kwargs) -> any:
        """
        Check cache for pre-calculated features and when not located, prepare them
        using provided initializer(*args, **kwargs).

        :param cache_location: Base path to the cache.
        :param feature_path: Relative path to the feature - e.g. "views.full".
        :param initializer: Initialization function, callable as initializer(self, *args, **kwargs).
        :param args: Arguments passed to the initializer.
        :param kwargs: Keyword arguments passed to the initializer.

        :return: Returns the same data as the initializer function.
        """

        # Recover information about current dataset.
        dataset_version = self._data_loader.dataset_version

        # Helper for checking record validity.
        def is_valid(record: any) -> bool:
            return self._is_valid_cache_record(
                record=record, feature_name=feature_path,
                dataset_version=dataset_version, *args, **kwargs
            )

        # Check if the data is already loaded.
        cache_record = self._feature_cache.get_path(
            path=feature_path, create=False,
            none_when_missing=True
        )
        cache_record_valid = is_valid(cache_record)

        if not cache_record_valid and cache_location:
            # Cache does not contain requested feature -> Check filesystem.
            self._feature_cache.load_cache_path(
                cache_path=cache_location,
                path=feature_path
            )
            cache_record = self._feature_cache.get_path(
                path=feature_path, create=False,
                none_when_missing=True
            )
            cache_record_valid = is_valid(cache_record)

        if not cache_record_valid:
            # Filesystem cache does not contain requested feature -> Initialize it.
            cache_record = {
                "data": initializer(self, *args, **kwargs),
                "version": self._generate_cache_version(
                    feature_name=feature_path,
                    dataset_version=dataset_version,
                    *args, **kwargs
                )
            }
            # Cache the value for later.
            self._feature_cache.set_path(
                path=feature_path, value=cache_record,
                create=True
            )
            # Should not be necessary...
            #cache_record_valid = is_valid(cache_record)

        # cache_record should now contain valid record.
        return cache_record["data"]

    class Decorators(object):
        @classmethod
        def cached(cls, initializer) -> any:
            """ Decorator used for cached methods. """

            def inner(self, *args, **kwargs):
                if "cache_identifier" in kwargs:
                    cache_identifier = f"{kwargs.pop('cache_identifier')}." \
                                       f"{initializer.__name__}"
                else:
                    cache_identifier = f"{initializer.__name__}"

                return self._check_load_cached(
                    cache_location=self.c.feature_cache_location,
                    feature_path=cache_identifier,
                    initializer=initializer,
                    *args, **kwargs
                )
            return inner

    def _save_feature_cache(self, cache_location: str):
        """ Save the cache to given location. """

        self.__l.info("Exporting feature cache...")

        self._feature_cache.save_cache_path(
            cache_path=cache_location,
            path=""
        )

        self.__l.info("\tFeature cache export finished!")

    def _calculate_tree_features(self, tree: TreeFile):
        """ Calculate feature vector for a single tree. """
        pass

    def _generate_feature_dicts_for_loader(
            self, data_loader: BaseDataLoader) -> Tuple[dict, dict, dict, list]:
        """ Generate dictionary for statistics, images, and others. Also return a list of tree ids with no features. """

        no_feature_ids = [ ]
        stat_dict = { }
        image_dict = { }
        other_dict = { }

        for tree_id, tree_data in data_loader.tree_data.items():
            dynamic = tree_data.dynamic_meta_data if tree_data is not None else { }
            if "stats" not in dynamic:
                no_feature_ids.append(tree_id)
                continue

            for name, item in dynamic["stats"].items():
                if name == "visual":
                    for image_name, image_item in item.items():
                        if not TreeImage.is_image_dict(image_item):
                            continue
                        if image_name not in image_dict:
                            image_dict[image_name] = { }
                        image_dict[image_name][tree_id] = image_item["image"]
                elif TreeStatistic.is_stat_dict(item):
                    if name not in stat_dict:
                        stat_dict[name] = { }
                    stat_dict[name][tree_id] = item["statistic"]
                else:
                    if name not in other_dict:
                        other_dict[name] = { }
                    other_dict[name][tree_id] = item

        stat_available = [ len(stat) for stat in stat_dict.values() ]
        image_available = [ len(image) for image in image_dict.values() ]
        other_available = [ len(other) for other in other_dict.values() ]
        max_available = np.max(stat_available + image_available + other_available)

        assert((np.array(stat_available) == max_available).all())
        assert((np.array(image_available) == max_available).all())
        assert((np.array(other_available) == max_available).all())

        return stat_dict, image_dict, other_dict, no_feature_ids

    @Decorators.cached
    def _generate_feature_dicts(self) -> Tuple[dict, dict, dict, list]:
        """ Generate dictionary for statistics, images and others. Also return a list of tree ids with no features. """

        return self._generate_feature_dicts_for_loader(
            data_loader=self._data_loader
        )

    def _generate_splits(self, split_scheme: Union[dict, Tuple[float], List[List[int]]],
                         cross_validation_split: Optional[Tuple[int, int, int]] = None,
                         data_loader: Optional[BaseDataLoader] = None,
                         split_names: List[str] = [ "train", "valid", "test" ]
                         ) -> dict:
        """
        Generate tree lists according to given split scheme.

        :param split_scheme: Splitting scheme, must be one of:
            * Dictionary - Containing some of the keys "train",
              "valid" and "test", each containing a List[int] of
              tree ids.
            * Tuple - Tuple of splitting percentages summing to
              1.0. In order: "train", "valid" and "test".
            * Tuple - Tuple of splitting percentages, combined
              with lists of tree IDs.
            * List - 2D list of lists containing tree ids in order
              "train", "valid" and "test"
        :param cross_validation_split: Cross validation scheme. Applied
            only for the Tuple representation of the split_scheme!
        :param data_loader: Optional data loader used instead of
            the default one.
        :param split_names: Default split names used if not specified.

        :return: Returns splitting scheme - dictionary containing
        all of the keys ["train", "valid", "test"], each of which
        contains a list of tree ids.
        """

        data_loader = data_loader or self._data_loader

        if isinstance(split_scheme, dict):
            return {
                split_name: split_scheme.get(split_name, [ ])
                for split_name in split_names
            }
        elif isinstance(split_scheme, tuple):
            if len(data_loader.tree_catalogue) > 0:
                all_tree_ids = data_loader.tree_catalogue.index.unique()
                tree_id_variants = {
                    tree_id: list(
                        data_loader.full_tree_catalogue.loc[( tree_id, )].index.unique()
                    )
                    for tree_id in all_tree_ids
                }
            else:
                all_tree_ids = data_loader.view_catalogue.index.unique(level="tree_id")
                tree_id_variants = {
                    tree_id: list(
                        data_loader.full_view_catalogue.loc[ (tree_id,) ].index.unique(level="tree_variant_id")
                    )
                    for tree_id in all_tree_ids
                }
            spec_ids = np.unique(np.concatenate([ v if isinstance(v, list) else [ ] for v in split_scheme ])).astype(int)
            all_tree_ids = list(np.setdiff1d(all_tree_ids, spec_ids))
            if cross_validation_split is not None:
                all_tree_ids = sku.shuffle(all_tree_ids, random_state=cross_validation_split[2] + cross_validation_split[0])

            split_indices = np.cumsum([ 0.0 ] + list([
                0.0 if isinstance(v, list) else v for v in split_scheme
            ])) * len(all_tree_ids)
            split_indices = split_indices.round().astype(np.int)
            split_ids = np.split(all_tree_ids, split_indices[:-1])[1:]

            splits = { }
            for split_idx, (split, split_name) in enumerate(zip(split_scheme, split_names)):
                if isinstance(split, list):
                    if len(split) > 0:
                        splits[split_name] = split
                else:
                    if len(split_ids[split_idx]) > 0:
                        splits[split_name] = split_ids[split_idx]

            splits = {
                split_name: [
                    ( tree_id, variant_id )
                    for tree_id in split_data
                    for variant_id in tree_id_variants[tree_id]
                ]
                for split_name, split_data in splits.items()
            }

            return splits
        elif isinstance(split_scheme, list):
            return {
                split_name: split_scheme[split_idx]
                for split_idx, split_name in enumerate(split_names)
            }
        else:
            self.__l.warn("No valid splitting scheme provided, falling back "
                          "to train/valid/test = 0.8/0.1/0.1 .")
            return self._generate_splits(
                split_scheme=(0.8, 0.1, 0.1),
                cross_validation_split=cross_validation_split,
                data_loader=data_loader,
                split_names=split_names
            )

    @Decorators.cached
    def generate_default_splits(self) -> dict:
        """ Generate default splits for the currently loaded data. """

        return self._generate_splits(
            split_scheme=self.c.data_split,
            cross_validation_split=self.c.cross_validation_split
        )

    def _generate_split_feature_from_dicts(self, split_scheme: dict,
                                           full_stat_dict: dict,
                                           full_image_dict: dict,
                                           full_other_dict: dict,
                                           no_feature_ids: list,
                                           ) -> dict:
        """ Generate concrete splits from provided data. """

        # Generate the splits.
        return {
            split_name: {
                "stats": {
                    stat_name: {
                        tree_id: stats[tree_id]
                        for tree_id in split_trees
                        if tree_id not in no_feature_ids
                    }
                    for stat_name, stats in full_stat_dict.items()
                },
                "images": {
                    image_name: {
                        tree_id: images[tree_id]
                        for tree_id in split_trees
                        if tree_id not in no_feature_ids
                    }
                    for image_name, images in full_image_dict.items()
                },
                "others": {
                    other_name: {
                        tree_id: others[tree_id]
                        for tree_id in split_trees
                        if tree_id not in no_feature_ids
                    }
                    for other_name, others in full_other_dict.items()
                }
            }
            for split_name, split_trees in split_scheme.items()
        }

    def _generate_split_feature_dicts_custom(self, split_scheme: dict,
                                             data_loader: Optional[BaseDataLoader] = None
                                             ) -> dict:
        """
        Generate and split statistic and image dictionaries according to
        a given scheme.

        :param split_scheme: Splitting scheme.
        :param data_loader: Optional data-loader used instead of the default.

        :return: Returns dictionary containing dictionaries for each split
        by name.
        """

        # Generate dicts for all data.
        full_stat_dict, full_image_dict, full_other_dict, no_feature_ids = \
            self._generate_feature_dicts_for_loader(
                data_loader=data_loader
        )

        return self._generate_split_feature_from_dicts(
            split_scheme=split_scheme,
            full_stat_dict=full_stat_dict,
            full_image_dict=full_image_dict,
            full_other_dict=full_other_dict,
            no_feature_ids=no_feature_ids,
        )

    @Decorators.cached
    def _generate_split_feature_dicts(self, split_scheme: dict) -> dict:
        """
        Generate and split statistic and image dictionaries according to
        a given scheme.

        :param split_scheme: Splitting scheme.
        :param data_loader: Optional data-loader used instead of the default.

        :return: Returns dictionary containing dictionaries for each split
        by name.
        """

        # Generate dicts for all data.
        full_stat_dict, full_image_dict, full_other_dict, no_feature_ids = \
            self._generate_feature_dicts()

        return self._generate_split_feature_from_dicts(
            split_scheme=split_scheme,
            full_stat_dict=full_stat_dict,
            full_image_dict=full_image_dict,
            full_other_dict=full_other_dict,
            no_feature_ids = no_feature_ids,
        )

    def _calculate_meta_features(self,
                                 stat_dict: dict) -> dict:
        """
        Calculate meta features for given tree dict.

        :param stat_dict: Dictionary of statistics for trees.

        :return: Returns dictionary containing meta-data features for
        requested trees.
        """

        tree_ids = np.sort(np.unique(np.concatenate([
            tuple_array_to_numpy(list(stat_data.keys())) for stat_data in stat_dict.values()
        ])))
        stat_names = list(stat_dict.keys())

        return {
            "tree_ids": tree_ids,
            "stat_names": stat_names,
        }

    @Decorators.cached
    def _calculate_feature_configuration(self,
                                         splits: dict,
                                         prefer_split: Optional[str] = None,
                                         quant_start: float = 0.001,
                                         quant_end: float = 0.999,
                                         total_buckets: int = 32,
                                         resolution: Optional[int] = None,
                                         interpolation: Optional[str] = None,
                                         normalize_features: bool = False,
                                         standardize_features: bool = False) -> dict:
        """
        Calculate common feature configuration for given splits.

        :param splits: Dictionary containing data for all splits.
        :param prefer_split: Set to a split name to use in bucket calculation.
            Keep at None to use all split data in bucket calculation.
        :param quant_start: Quantile used for the first histogram bin.
        :param quant_end: Quantile used for the last histogram bin.
        :param total_buckets: Total number of buckets in the histograms.
        :param resolution: Target resolution to convert the views to.
            Set to None for no resizing.
        :param interpolation: Interpolation used for resize operations.
        :param normalize_features: Perform feature normalization?
        :param standardize_features: Perform feature standardization?

        :return: Returns configuration dictionary.
        """

        if prefer_split is not None and prefer_split in splits:
            # Prefer requested data.
            stat_dict = splits[prefer_split]["stats"]
            other_dict = splits[prefer_split]["others"]
        elif prefer_split is not None:
            # Choose the largest split.
            split_sizes = {
                split_name: np.max([
                    len(tree_data)
                    for tree_data in split_data.values()
                ])
                for split_name, split_data in splits.items()
            }
            largest_split_name = max(split_sizes, key=split_sizes.get)
            stat_dict = splits[largest_split_name]["stats"]
            other_dict = splits[prefer_split]["others"]
            self.__l.warn(f"Invalid preferred split \"{prefer_split}\", using \"{largest_split_name}\" instead!")
        else:
            # Use all split data.
            stat_dict = None
            other_dict = None
            for split_name, split_data in splits.items():
                if stat_dict is None:
                    stat_dict = deep_copy_dict(split_data["stats"])
                if other_dict is None:
                    other_dict = deep_copy_dict(split_data["others"])
                else:
                    stat_dict = update_dict_recursively(
                        stat_dict, split_data["stats"], create_keys=True)
                    other_dict = update_dict_recursively(
                        other_dict, split_data["others"], create_keys=True)

        # Concatenate values for each feature.
        stat_values = {
            name: np.concatenate([ stats.bucket_values for stats in tree_stats.values() ])
            for name, tree_stats in stat_dict.items()
        }
        other_names = {
            name: np.unique(features)
            for name, features in dict_of_lists([
                ( category_name, name )
                for category_name, category_data in other_dict.items()
                for tree_id, values in category_data.items()
                for name, value in values.items()
            ]).items()
        }
        other_values = {
            category_name: {
                name: [
                    values[name]
                    for tree_id, values in category_data.items()
                    if name in values
                ]
                for name in other_names[category_name]
            }
            for category_name, category_data in other_dict.items()
        }

        # Calculate statistics:
        stat_stats = {
            name: {
                "min": np.min(values),
                "max": np.max(values),
                "mean": np.mean(values),
                "std": np.std(values),
            }
            for name, values in stat_values.items()
        }
        other_stats = {
            category_name: {
                name: {
                    "min": np.min(values),
                    "max": np.max(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                }
                for name, values in category_data.items()
            }
            for category_name, category_data in other_values.items()
        }

        # Build buckets and visualization buckets.
        stat_buckets = { }
        stat_vis_buckets = { }
        for name, values in stat_values.items():
            # Create the "lower" bucket.
            true_min = numpy_op_or_zero(values, np.min)
            quantile_min = numpy_op_or_zero(values, lambda x: np.quantile(x, quant_start))
            true_min_valid = true_min < quantile_min

            # Create the "higher" bucket.
            true_max = numpy_op_or_zero(values, np.max)
            quantile_max = numpy_op_or_zero(values, lambda x: np.quantile(x, quant_end))
            true_max_valid = true_max > quantile_max

            # Create "middle" buckets.
            bucket_count = total_buckets + 1 - true_min_valid - true_max_valid
            assert (bucket_count >= 2)
            buckets = np.linspace(
                quantile_min, quantile_max,
                bucket_count
            )
            bucket_width = buckets[2] - buckets[1]

            # Aggregate buckets into the result.
            stat_buckets[name] = np.concatenate([
                [ true_min ] if true_min_valid else [ ],
                buckets,
                [ true_max ] if true_max_valid else [ ],
            ])

            # Create visualization buckets.
            vis_buckets = stat_buckets[name].copy()
            if true_min_valid:
                vis_buckets[0] = min(
                    vis_buckets[0],
                    vis_buckets[1] - bucket_width
                )
            if true_max_valid:
                vis_buckets[-1] = min(
                    vis_buckets[-1],
                    vis_buckets[-2] + bucket_width
                )

            stat_vis_buckets[name] = vis_buckets

        # Calculate per-feature histogram maximum values.
        stat_histograms_splits = {
            split_name: self._calculate_histograms(
                stat_dict=split_data["stats"],
                buckets=stat_buckets
            )
            for split_name, split_data in splits.items()
        }
        stat_histograms_stats = {
            name: {
                "min": np.min([
                    np.max(hist["counts"])
                    for tree_id, hist in hists.items()
                ]),
                "max": np.max([
                    np.max(hist[ "counts" ])
                    for tree_id, hist in hists.items()
                ]),
                "mean": np.mean([
                    np.max(hist[ "counts" ])
                    for tree_id, hist in hists.items()
                ]),
                "std": np.std([
                    np.max(hist[ "counts" ])
                    for tree_id, hist in hists.items()
                ]),
            }
            for split_name, stat_histograms in stat_histograms_splits.items()
            for name, hists in stat_histograms.items()
        }
        stat_histograms_maxes = {
            name: np.max([
                np.max(hist["counts"])
                for tree_id, hist in hists.items()
            ])
            for split_name, stat_histograms in stat_histograms_splits.items()
            for name, hists in stat_histograms.items()
        }

        return {
            "stat_stats": stat_stats,
            "other_stats": other_stats,
            "buckets": stat_buckets,
            "vis_buckets": stat_vis_buckets,
            "histogram_stats": stat_histograms_stats,
            "histogram_maxes": stat_histograms_maxes,
            "resolution": resolution,
            "interpolation": interpolation,
            "normalize_features": normalize_features,
            "standardize_features": standardize_features,
        }

    @staticmethod
    def _calculate_histograms(stat_dict: dict,
                              buckets: dict) -> dict:
        """ Calculate bucketed histograms for given statistics. """
        return {
            name: {
                tree_id: {
                    "counts":
                        np.histogram(a=stats.bucket_values, weights=stats.count_values,
                                     bins=buckets[name], density=True)[0]
                }
                for tree_id, stats in tree_stats.items()
            }
            for name, tree_stats in stat_dict.items()
            if name in buckets
        }

    def _calculate_histogram_features(self,
                                      stat_dict: dict,
                                      configuration: dict) -> dict:
        """
        Calculate histogram features for all trees in the given tree dict.

        :param stat_dict: Dictionary of statistics for trees.
        :param configuration: Common feature  configuration.

        :return: Returns dictionary containing histogram features for
        requested trees.
        """

        # Calculate histograms.
        stat_histograms = self._calculate_histograms(
            stat_dict=stat_dict, buckets=configuration["buckets"]
        )

        # Calculate histograms normalized over each tree.
        stat_histograms_normalized = {
            name: {
                tree_id: {
                    "counts": hist["counts"] / np.max(hist["counts"])
                }
                for tree_id, hist in hists.items()
            }
            for name, hists in stat_histograms.items()
        }

        # Calculate histograms normalized over each feature.
        stat_histograms_all_normalized = {
            name: {
                tree_id: {
                    "counts": hist["counts"] / configuration["histogram_maxes"][name]
                }
                for tree_id, hist in hists.items()
            }
            for name, hists in stat_histograms.items()
        }

        return {
            "buckets": configuration["buckets"],
            "vis_buckets": configuration["vis_buckets"],
            "hist": stat_histograms,
            "hist_names": list(stat_histograms.keys()),
            "hist_norm": stat_histograms_normalized,
            "hist_all_norm": stat_histograms_all_normalized
        }

    def stat_elements(self):
        """ Get list of statistic elements calculated from raw features. """

        return [ "min", "max", "mean", "var" ]

    def _calculate_statistics_features(self,
                                       stat_dict: dict,
                                       configuration: dict) -> dict:
        """
        Calculate statistics features for all trees in the given tree dict.

        :param stat_dict: Dictionary of statistics for trees.
        :param configuration: Common feature  configuration.

        :return: Returns dictionary containing statistics features for
        requested trees.
        """

        # Concatenate values for each feature.
        stat_tree_values = {
            stat_name: {
                tree_id: stats.values
                for tree_id, stats in tree_stats.items()
            }
            for stat_name, tree_stats in stat_dict.items()
        }
        """
        stat_all_values = {
            stat_name: stat_values if len(stat_values) else \
                np.array([ numpy_zero(dtype=stat_values.dtype) ], dtype=stat_values.dtype)
            for stat_name, tree_stats in stat_dict.items()
            for stat_values in [ np.concatenate([
                stats.values for stats in tree_stats.values()
            ]) ]
        }
        """

        normalize_features = configuration.get("normalize_features", False)
        standardize_features = configuration.get("standardize_features", False)

        def calculate_stats(tree_values: List[float], tree_stats: TreeStatistic, name: str) -> dict:
            if len(tree_values) == 0:
                return {
                    "min": reshape_scalar(numpy_op_or_zero(reshape_scalar(tree_stats.data["variable"]["min"]), np.min)),
                    "max": reshape_scalar(numpy_op_or_zero(reshape_scalar(tree_stats.data["variable"]["max"]), np.max)),
                    "mean": reshape_scalar(numpy_op_or_zero(reshape_scalar(tree_stats.data["stochastic"]["mean"]), np.mean)),
                    "var": reshape_scalar(numpy_op_or_zero(reshape_scalar(tree_stats.data["stochastic"]["variance"]), np.var))
                }
            else:
                values = tree_values
                if normalize_features:
                    values = (values - configuration["stat_stats"][name]["min"]) / \
                             (configuration["stat_stats"][name]["max"] - configuration["stat_stats"][name]["min"])
                if standardize_features:
                    values = (values - configuration["stat_stats"][name]["mean"]) / \
                             (configuration["stat_stats"][name]["std"])
                return {
                    "min": reshape_scalar(numpy_op_or_zero(values, np.min)),
                    "max": reshape_scalar(numpy_op_or_zero(values, np.max)),
                    "mean": reshape_scalar(numpy_op_or_zero(values, np.mean)),
                    "var": reshape_scalar(numpy_op_or_zero(values, np.var))
                }

        return {
            "stats_elements": self.stat_elements(),
            "stats_names": list(stat_dict.keys()),
            "stats": {
                stat_name: {
                    tree_id: calculate_stats(
                        tree_values=stat_tree_values[stat_name][tree_id],
                        tree_stats=stat_dict[stat_name][tree_id],
                        name=stat_name
                    )
                    for tree_id in tree_record.keys()
                }
                for stat_name, tree_record in stat_tree_values.items()
            }
        }

    def _calculate_image_features(self,
                                  image_dict: dict,
                                  configuration: dict) -> dict:
        """
        Calculate image features for all trees in the given tree dict.

        :param image_dict: Dictionary of image features for trees.
        :param configuration: Configuration common to all data.

        :return: Returns dictionary containing image features for
        requested trees.
        """

        resolution = configuration.get("resolution", None)
        interpolation = configuration.get("interpolation", None)

        images = {
            image_name: {
                tree_id: image.resize(
                    resolution=resolution,
                    interpolation=interpolation
                )
                for tree_id, image in image_stats.items()
            }
            for image_name, image_stats in image_dict.items()
        }

        return {
            "images": {
                image_name: {
                    tree_id: image
                    for tree_id, image in image_stats.items()
                }
                for image_name, image_stats in images.items()
            },
            "image_data": {
                image_name: {
                    tree_id: image.data
                    for tree_id, image in image_stats.items()
                }
                for image_name, image_stats in images.items()
            },
            "image_names": list(images.keys())
      }

    def _calculate_others_features(self,
                                   others_dict: dict,
                                   configuration:dict) -> dict:
        """
        Calculate other features for all trees in the given tree dict.

        :param others_dict: Dictionary of other features for trees.
        :param configuration: Common feature  configuration.

        :return: Returns dictionary containing other features for
        requested trees.
        """

        normalize_features = configuration.get("normalize_features", False)
        standardize_features = configuration.get("standardize_features", False)

        def standardize_normalize(values: np.array, category: str, name: str,
                                  normalize: bool, standardize: bool) -> np.array:
            if normalize:
                values = (values - configuration["other_stats"][category][name]["min"]) / \
                         (configuration["other_stats"][category][name]["max"] - configuration["other_stats"][category][name]["min"])
            if standardize:
                values = (values - configuration["other_stats"][category][name]["mean"]) / \
                         (configuration["other_stats"][category][name]["std"])

            return values

        others_names_dict = {
            other_name: np.sort(np.unique([
                stat_name
                for stats in other_stats.values()
                for stat_name in stats.keys()
            ]))
            for other_name, other_stats in others_dict.items()
        }
        others_names = np.sort(np.unique([
            f"{other_name}_{stat_name}"
            for other_name, other_stats in others_dict.items()
            for stats in other_stats.values()
            for stat_name in stats.keys()
        ]))
        tree_ids = np.sort(np.unique(tuple_array_to_numpy([
            tree_id
            for stats in others_dict.values()
            for tree_id in stats.keys()
        ])))

        return {
            "others_names": others_names,
            "others": {
                f"{other_name}_{stat_name}": {
                    tree_id: standardize_normalize(
                        values=reshape_scalar(others_dict[other_name][tree_id][stat_name]),
                        category=other_name, name=stat_name,
                        normalize=normalize_features,
                        standardize=standardize_features
                    )
                    for tree_id in tree_ids
                }
                for other_name, other_stats in others_dict.items()
                for stat_name in others_names_dict[other_name]
            }
        }

    def _calculate_features(self,
                            splits: dict,
                            configuration: dict) -> dict:
        """ Calculate features using provided splits and configuration. """

        split_features = {
            split_name: {
                "meta": self._calculate_meta_features(
                    stat_dict=split_data["stats"],
                ),
                "hist": self._calculate_histogram_features(
                    stat_dict=split_data["stats"],
                    configuration=configuration
                ),
                "stats": self._calculate_statistics_features(
                    stat_dict=split_data["stats"],
                    configuration=configuration
                ),
                "image": self._calculate_image_features(
                    image_dict=split_data["images"],
                    configuration=configuration
                ),
                "others": self._calculate_others_features(
                    others_dict=split_data["others"],
                    configuration=configuration
                )
            }
            for split_name, split_data in splits.items()
        }

        self.display_feature_information(
            split_features,
            display_meta=self.c.debug_display_meta_features,
            display_hist=self.c.debug_display_hist_features,
            display_stats=self.c.debug_display_stats_features,
            display_image=self.c.debug_display_image_features,
            display_others=self.c.debug_display_others_features,
        )

        return {
            "data": split_features,
            "configuration": configuration
        }

    def prepare_feature_configuration(self,
                                      quant_start: float = 0.001,
                                      quant_end: float = 0.999,
                                      total_buckets: int = 32,
                                      buckets_from_split: bool = True,
                                      resolution: Optional[int] = None,
                                      interpolation: Optional[str] = None,
                                      normalize_features: bool = False,
                                      standardize_features: bool = False) -> dict:
        """ Prepare feature configuration, see calculate_features for argument description. """

        split_scheme = self._generate_splits(
            split_scheme=self.c.data_split,
            cross_validation_split=self.c.cross_validation_split
        )
        splits = self._generate_split_feature_dicts(split_scheme=split_scheme)

        configuration = self._calculate_feature_configuration(
            splits=splits,
            prefer_split="train" if buckets_from_split else None,
            quant_start=quant_start, quant_end=quant_end,
            total_buckets=total_buckets,
            resolution=resolution, interpolation=interpolation,
            normalize_features=normalize_features,
            standardize_features=standardize_features
        )

        return configuration

    @Decorators.cached
    def calculate_features(self,
                           quant_start: float = 0.001,
                           quant_end: float = 0.999,
                           total_buckets: int = 32,
                           buckets_from_split: bool = True,
                           resolution: Optional[int] = None,
                           interpolation: Optional[str] = None,
                           normalize_features: bool = False,
                           standardize_features: bool = False) -> dict:
        """
        Calculate features and return dictionary containing data (training,
        validation and test) and configuration.

        :param quant_start: Quantile used for the first histogram bin.
        :param quant_end: Quantile used for the last histogram bin.
        :param total_buckets: Total number of buckets in the histograms.
        :param buckets_from_split: Use buckets generated from the training
            data only. Set to False to calculate buckets from all available
            data.
        :param resolution: Target resolution to convert the views to.
            Set to None for no resizing.
        :param interpolation: Interpolation used for resize operations.
        :param normalize_features: Perform feature normalization?
        :param standardize_features: Perform feature standardization?

        :return: Returns dictionary containing feature data.
        """

        split_scheme = self._generate_splits(
            split_scheme=self.c.data_split,
            cross_validation_split=self.c.cross_validation_split
        )
        splits = self._generate_split_feature_dicts(split_scheme=split_scheme)

        configuration = self._calculate_feature_configuration(
            splits=splits,
            prefer_split="train" if buckets_from_split else None,
            quant_start=quant_start, quant_end=quant_end,
            total_buckets=total_buckets,
            resolution=resolution, interpolation=interpolation,
            normalize_features=normalize_features,
            standardize_features=standardize_features
        )

        return self._calculate_features(
            splits=splits, configuration=configuration
        )

    def calculate_features_for_tree_ids(self,
                                        tree_ids: List[int],
                                        configuration: dict) -> dict:
        """ Calculate all features using given configuration and return only features for given trees under "data" split. """

        split_scheme = self._generate_splits(split_scheme={ "data": tree_ids })
        splits = self._generate_split_feature_dicts(split_scheme=split_scheme)

        return self._calculate_features(
            splits=splits, configuration=configuration
        )

    @Decorators.cached
    def calculate_features_for_data(self,
                                    data_loader: BaseDataLoader,
                                    configuration: dict) -> dict:
        """ Calculate all features for data loaded in given data_loader. """

        split_scheme = self._generate_splits(split_scheme=(1.0, ), data_loader=data_loader, split_names=["data"])
        splits = self._generate_split_feature_dicts_custom(split_scheme=split_scheme, data_loader=data_loader)

        return self._calculate_features(
            splits=splits, configuration=configuration
        )

    def _generate_view_dict_custom(self, data_loader: BaseDataLoader,
                                   view_types: Optional[List[str]] = None
                                   ) -> (dict, List[str]):
        """ Generate dictionary for view paths. """

        view_path_dict = { }
        view_paths = [ ]

        view_data = data_loader.full_view_catalogue.reset_index()
        for row_index, row_data in view_data.iterrows():
            if view_types is not None and row_data.view_type not in view_types:
                continue

            update_dict_recursively(view_path_dict, {
                row_data.view_type: {
                    ( row_data.tree_id, row_data.tree_variant_id ): {
                        ( row_data.view_id, row_data.view_variant_id ): (row_data.path, row_data.data)
                    }
                }
            }, create_keys=True)
            # TODO - Add support for tree variants, using 0 for now.
            if len(row_data.path) > 0:
                view_paths.append(row_data.path)

        view_paths = np.unique(view_paths)
        return view_path_dict, view_paths

    @Decorators.cached
    def _generate_view_dict(self, view_types: Optional[List[str]]) -> (dict, List[str]):
        """ Generate dictionary for view paths. """

        return self._generate_view_dict_custom(
            data_loader=self._data_loader, view_types=view_types
        )

    @Decorators.cached
    def _cached_load_image_from_path(self, image_path: str) -> TreeImage:
        """ Perform cached load image operation and return the result. """
        return TreeImage(image_path=image_path)

    def _generate_split_view_dicts_data(self, split_scheme: dict,
                                        view_path_dict: dict,
                                        view_paths: List[str]) -> dict:
        """
        Generate and split view data according to given split scheme.

        :param split_scheme: Splitting scheme.
        :param view_path_dict: Pre-generated view path dict.
        :param view_paths: Pre-generated view paths dict.

        :return: Returns dictionary containing dictionaries for each split
        by name.
        """

        self.__l.info(f"Loading view images...")

        loaded_views = { }
        parsing_progress = ParsingBar("", max=len(view_paths))
        for view_path in view_paths:
            loaded_views[view_path] = self._cached_load_image_from_path(
                image_path=f"{self._data_loader.view_base_path}/{view_path}"
            )
            parsing_progress.next(1)
        parsing_progress.finish()

        # Generate the splits.
        return {
            split_name: {
                "view_paths": {
                    view_type: {
                        tree_id: {
                            view_id: view_data[0]
                            for view_id, view_data in path_data[tree_id].items()
                        }
                        for tree_id in split_trees
                        if tree_id in path_data
                    }
                    for view_type, path_data in view_path_dict.items()
                },
                "views": {
                    view_type: {
                        tree_id: {
                            view_id: view_data[1] or loaded_views[view_data[0]]
                            for view_id, view_data in path_data[tree_id].items()
                        }
                        for tree_id in split_trees
                        if tree_id in path_data
                    }
                    for view_type, path_data in view_path_dict.items()
                },
            }
            for split_name, split_trees in split_scheme.items()
        }

    @Decorators.cached
    def _generate_split_view_dicts(self, split_scheme: dict,
                                   view_types: Optional[List[str]]) -> dict:
        """ Generate and split view data according to given split scheme. """

        # Generate dicts for all data.
        view_path_dict, view_paths = self._generate_view_dict(
            view_types=view_types
        )

        return self._generate_split_view_dicts_data(
            split_scheme=split_scheme,
            view_path_dict=view_path_dict,
            view_paths=view_paths
        )

    def _generate_split_view_dicts_custom(self, split_scheme: dict,
                                          data_loader: BaseDataLoader,
                                          view_types: Optional[List[str]] = None
                                          ) -> dict:
        """ Generate and split view data according to given split scheme. """

        # Generate dicts for all data.
        view_path_dict, view_paths = self._generate_view_dict_custom(
            data_loader=data_loader, view_types=view_types
        )

        return self._generate_split_view_dicts_data(
            split_scheme=split_scheme,
            view_path_dict=view_path_dict,
            view_paths=view_paths
        )

    @Decorators.cached
    def _calculate_view_features(self,
                                 view_dict: dict,
                                 resolution: Optional[int] = None,
                                 interpolation: Optional[str] = None) -> dict:
        """
        Calculate view features for all trees in the given tree dict.

        :param view_dict: Dictionary of view paths for trees.
        :param resolution: Target resolution to convert the views to.
            Set to None for no resizing.
        :param interpolation: Interpolation used for resize operations.

        :return: Returns dictionary containing view features for
        requested trees.
        """

        self.__l.info(f"Resizing view images to {resolution} using \"{interpolation}\"...")

        input_views = {
            (view_type, tree_id, view_id): view
            for view_type, view_data in view_dict["views"].items()
            for tree_id, tree_data in view_data.items()
            for view_id, view in tree_data.items()
        }

        tree_ids = numpy_array_to_tuple_numpy(np.unique([ view_spec[1] for view_spec in input_views.keys() ], axis=0))
        tree_view_ids = np.unique(tuple_array_to_numpy([ view_spec[1:] for view_spec in input_views.keys() ]))

        resized_views = { }
        parsing_progress = ParsingBar("", max=len(input_views))
        for view_info, view in input_views.items():
            resized_views[view_info] = view.resize(
                resolution=resolution,
                interpolation=interpolation
            )
            parsing_progress.next(1)
        parsing_progress.finish()

        views = {
            view_type: {
                tree_id: {
                    view_id: resized_views[(view_type, tree_id, view_id)]
                    for view_id, view in tree_data.items()
                }
                for tree_id, tree_data in view_data.items()
            }
            for view_type, view_data in view_dict["views"].items()
        }

        return {
            "view_paths": view_dict["view_paths"],
            "views": {
                view_type: {
                    tree_id: {
                        view_id: view
                        for view_id, view in tree_data.items()
                    }
                    for tree_id, tree_data in view_data.items()
                }
                for view_type, view_data in views.items()
            },
            "view_data": {
                view_type: {
                    tree_id: {
                        view_id: view.data
                        for view_id, view in tree_data.items()
                    }
                    for tree_id, tree_data in view_data.items()
                }
                for view_type, view_data in views.items()
            },
            "tree_ids": tree_ids,
            "tree_view_ids": tree_view_ids
        }

    @Decorators.cached
    def _calculate_views_configuration(self,
                                       resolution: Optional[int] = None,
                                       interpolation: Optional[str] = None,
                                       view_types: Optional[List[str]] = None
                                       ) -> dict:
        """ Create configuration object from given views options. """
        return {
            "resolution": resolution,
            "interpolation": interpolation,
            "view_types": view_types
        }

    def _calculate_views(self,
                         splits: dict,
                         configuration: dict) -> dict:
        """ Calculate view features using provided splits and configuration. """

        split_views = {
            split_name: {
                "views": self._calculate_view_features(
                    view_dict=split_data,
                    resolution=configuration["resolution"],
                    interpolation=configuration["interpolation"]
                )
            }
            for split_name, split_data in splits.items()
        }

        self.display_view_information(
            views=split_views,
            display_views=self.c.debug_display_views_features
        )

        return {
            "data": split_views,
            "configuration": configuration
        }

    @Decorators.cached
    def calculate_views(self,
                        resolution: Optional[int] = None,
                        interpolation: Optional[str] = None,
                        view_types: Optional[List[str]] = None
                        ) -> dict:
        """
        Calculate image features and return dictionary containing training, validation
        and test data.

        :param resolution: Target resolution to convert the views to.
            Set to None for no resizing.
        :param interpolation: Interpolation used for resize operations.
        :param view_types: Which view types should be loaded. None for all of them.

        :return: Returns dictionary containing image feature data.
        """

        split_scheme = self._generate_splits(
            split_scheme=self.c.data_split,
            cross_validation_split=self.c.cross_validation_split
        )
        splits = self._generate_split_view_dicts(
            split_scheme=split_scheme, view_types=view_types)

        configuration = self._calculate_views_configuration(
            resolution=resolution,
            interpolation=interpolation,
            view_types=view_types
        )

        return self._calculate_views(
            splits=splits, configuration=configuration
        )

    @Decorators.cached
    def calculate_views_for_data(self,
                                 data_loader: BaseDataLoader,
                                 configuration: dict) -> dict:
        """ Calculate all features for data loaded in given data_loader. """

        split_scheme = self._generate_splits(split_scheme=(1.0, ), data_loader=data_loader, split_names=["data"])
        splits = self._generate_split_view_dicts_custom(
            split_scheme=split_scheme, data_loader=data_loader,
            view_types=configuration["view_types"]
        )

        return self._calculate_views(
            splits=splits, configuration=configuration
        )

    def _generate_score_data_custom(self, data_loader: BaseDataLoader
                                    ) -> (pd.DataFrame, pd.DataFrame, dict, dict):
        """ Generate data dictionaries for scores. """

        tree_view_choices = data_loader.full_results
        tree_scores = { }
        tree_spherical_scores = { }

        scores = data_loader.full_scores_indexed

        for score_id, row_data in scores.iterrows():
            tree_id = ( score_id[0], score_id[1] )
            view_id = ( score_id[2], score_id[3] )
            if tree_id in tree_scores:
                tree_data = tree_scores[tree_id]
            else:
                tree_data = { }

            key = view_id if view_id[0] >= 0 else "aggregate"

            tree_data.update({
                key: {
                    "jod": row_data["jod"],
                    "jod_low": row_data["jod_low"],
                    "jod_high": row_data["jod_high"],
                    "jod_var": row_data["jod_var"]
                }
            })
            tree_scores[tree_id] = tree_data

        spherical_scores = data_loader.spherical_scores_indexed

        for score_id, row_data in spherical_scores.iterrows():
            tree_id = ( score_id[0], score_id[1] )
            view_id = ( score_id[2], score_id[3] )
            if tree_id in tree_scores:
                tree_data = tree_scores[tree_id]
            else:
                tree_data = { }

            key = view_id if view_id[0] >= 0 else "aggregate"

            tree_data.update({
                key: {
                    "jod": row_data["jod"],
                    "jod_low": row_data["jod_low"],
                    "jod_high": row_data["jod_high"],
                    "jod_var": row_data["jod_var"]
                }
            })
            tree_spherical_scores[tree_id] = tree_data

        return tree_view_choices, scores, tree_scores, tree_spherical_scores

    @Decorators.cached
    def _generate_score_data(self) -> (pd.DataFrame, pd.DataFrame, dict, dict):
        """ Generate data dictionaries for scores. """

        return self._generate_score_data_custom(data_loader=self._data_loader)

    def _generate_split_score_dicts_data(self, split_scheme: dict,
                                         tree_view_choices: pd.DataFrame,
                                         indexed_scores: pd.DataFrame,
                                         tree_scores: dict,
                                         tree_spherical_scores: dict) -> dict:
        """
        Generate and split score data according to given split scheme.

        :param split_scheme: Splitting scheme.
        :param tree_view_choices: Pre-generated tree-view choices catalogue.
        :param indexed_scores: Pre-generated indexed scores catalogue.
        :param tree_scores: Pre-generated tree score information.
        :param tree_spherical_scores: Pre-generated spherical tree score information.

        :return: Returns dictionary containing dictionaries for each split
        by name.
        """

        def filter_row(row: pd.Series, split: list) -> bool:
            return (( row.first_tree_id, row.first_tree_variant_id ) in split) and \
                   (( row.second_tree_id, row.second_tree_variant_id ) in split)

        # Generate the splits.
        return {
            split_name: {
                "tree_view_choices": tree_view_choices.loc[
                    #tree_view_choices.apply(func=filter_row, axis=1, split=list(split_trees))
                    tree_view_choices.first_tree_id.isin(split_trees_ids) &
                    tree_view_choices.first_tree_variant_id.isin(split_tree_variant_ids) &
                    tree_view_choices.second_tree_id.isin(split_trees_ids) &
                    tree_view_choices.second_tree_variant_id.isin(split_tree_variant_ids)
                    # TODO - Add support for tree variants, apply is too slow.
                ],
                "indexed_scores": indexed_scores,
                "scores": {
                    # Use full tree id, fall back to the base variant scores.
                    tree_id: tree_scores[tree_id] \
                        if tree_id in tree_scores else \
                        tree_scores[( tree_id[0], 0 )]
                    for tree_id in split_trees
                },
                "spherical_scores": {
                    # Use full tree id, fall back to the base variant scores.
                    tree_id: tree_spherical_scores[tree_id] \
                        if tree_id in tree_spherical_scores else \
                        tree_spherical_scores[ (tree_id[0], 0) ]
                    for tree_id in split_trees
                },
            }
            for split_name, split_trees in split_scheme.items()
            for split_trees_ids, split_tree_variant_ids in [ (
                np.array(list(split_trees))[:, 0],
                np.array(list(split_trees))[:, 1]
            ) ]
        }

    @Decorators.cached
    def _generate_split_score_dicts(self, split_scheme: dict) -> dict:
        """ Generate and split score data according to given split scheme. """

        # Generate data.
        tree_view_choices, indexed_scores, tree_scores, tree_spherical_scores = \
            self._generate_score_data()

        return self._generate_split_score_dicts_data(
            split_scheme=split_scheme,
            tree_view_choices=tree_view_choices,
            indexed_scores=indexed_scores,
            tree_scores=tree_scores,
            tree_spherical_scores=tree_spherical_scores
        )

    def _generate_split_score_dicts_custom(self, split_scheme: dict,
                                           data_loader: BaseDataLoader) -> dict:
        """ Generate and split score data according to given split scheme. """

        # Generate data.
        tree_view_choices, indexed_scores, tree_scores, tree_spherical_scores = \
            self._generate_score_data_custom(data_loader=data_loader)

        return self._generate_split_score_dicts_data(
            split_scheme=split_scheme,
            tree_view_choices=tree_view_choices,
            indexed_scores=indexed_scores,
            tree_scores=tree_scores,
            tree_spherical_scores=tree_spherical_scores
        )

    def _calculate_scores(self, splits: dict, configuration: dict) -> dict:
        """
        Calculate scores and return dictionary containing training, validation
        and test data.

        :return: Returns dictionary containing score data.
        """

        self.display_score_information(
            scores=splits,
            display_view_choices=self.c.debug_display_view_choices,
            display_scores=self.c.debug_display_scores
        )

        return {
            "data": splits,
            "configuration": configuration
        }

    @Decorators.cached
    def calculate_scores(self) -> dict:
        """
        Calculate scores and return dictionary containing training, validation
        and test data.

        :return: Returns dictionary containing score data.
        """

        split_scheme = self._generate_splits(
            split_scheme=self.c.data_split,
            cross_validation_split=self.c.cross_validation_split
        )
        splits = self._generate_split_score_dicts(split_scheme=split_scheme)

        return self._calculate_scores(splits=splits, configuration={ })

    @Decorators.cached
    def calculate_scores_for_data(self,
                                  data_loader: BaseDataLoader,
                                  configuration: dict) -> dict:
        """ Calculate scores for data loaded in given data_loader. """

        split_scheme = self._generate_splits(split_scheme=(1.0, ), data_loader=data_loader, split_names=["data"])
        splits = self._generate_split_score_dicts_custom(split_scheme=split_scheme, data_loader=data_loader)

        return self._calculate_scores(
            splits=splits, configuration=configuration
        )

    def _generate_skeleton_dict_custom(self, data_loader: BaseDataLoader
                                       ) -> dict:
        """ Generate dictionary for skeletons. """

        return data_loader.tree_data

    @Decorators.cached
    def _generate_skeleton_dict(self) -> dict:
        """ Generate dictionary for view paths. """

        return self._generate_skeleton_dict_custom(
            data_loader=self._data_loader
        )

    def _generate_split_skeleton_dicts_data(self, split_scheme: dict,
                                            skeleton_dict: dict,
                                            skeleton_types: Optional[List[str]]) -> dict:
        """
        Generate and split skeleton data according to given split scheme.

        :param split_scheme: Splitting scheme.
        :param skeleton_dict: Dictionary of all skeleton files.
        :param skeleton_types: Types of data requested.

        :return: Returns dictionary containing dictionaries for each split
        by name.
        """

        self.__l.info(f"Loading skeleton data...")

        loaded_skeletons = { }
        parsing_progress = ParsingBar("", max=len(skeleton_dict))
        for tree_id, skeleton_file in skeleton_dict.items():
            # TODO - Load the skeleton data, if not loaded already...
            loaded_skeletons[tree_id] = skeleton_file
            parsing_progress.next(1)
        parsing_progress.finish()

        # Generate the splits.
        return {
            split_name: {
                "skeletons": {
                    tree_id: loaded_skeletons[tree_id]
                    for tree_id in split_trees
                },
            }
            for split_name, split_trees in split_scheme.items()
        }

    @Decorators.cached
    def _generate_split_skeleton_dicts(self, split_scheme: dict,
                                       skeleton_types: Optional[List[str]]) -> dict:
        """ Generate and split skeleton data according to given split scheme. """

        # Generate dicts for all data.
        skeleton_dict = self._generate_skeleton_dict()

        return self._generate_split_skeleton_dicts_data(
            split_scheme=split_scheme,
            skeleton_dict=skeleton_dict,
            skeleton_types=skeleton_types
        )

    def _generate_split_skeleton_dicts_custom(self, split_scheme: dict,
                                              data_loader: BaseDataLoader,
                                              skeleton_types: Optional[List[str]] = None
                                              ) -> dict:
        """ Generate and split skeleton data according to given split scheme. """

        # Generate dicts for all data.
        skeleton_dict = self._generate_skeleton_dict_custom(
            data_loader=data_loader
        )

        return self._generate_split_skeleton_dicts_data(
            split_scheme=split_scheme,
            skeleton_dict=skeleton_dict,
            skeleton_types=skeleton_types
        )

    @Decorators.cached
    def _calculate_skeleton_features(self,
                                     skeleton_dict: dict,
                                     skeleton_types: Optional[List[str]]) -> dict:
        """
        Calculate skeleton features for all trees in the given tree dict.

        :param skeleton_dict: Dictionary of skeleton files for trees.
        :param skeleton_types: Type of skeleton data requested:
            "segment", "position", "thickness".

        :return: Returns dictionary containing skeleton features for
        requested trees.
        """

        self.__l.info(f"Extracting data from skeletons: {skeleton_types}...")

        input_skeletons = skeleton_dict
        tree_ids = numpy_array_to_tuple_numpy(list(input_skeletons.keys()))

        skeleton_data = { "node_ids": { } }
        for data_type in skeleton_types:
            skeleton_data[data_type] = { }

        for tree_id, skeleton_file in skeleton_dict.items():
            node_data = skeleton_file.node_data

            skeleton_data["node_ids"][tree_id] = [ node["id"] for node in node_data ]

            if "segment" in skeleton_types:
                skeleton_data["segment"][tree_id] = [
                    (node[ "parent" ], node[ "id" ])
                    for node in node_data if node[ "parent" ] >= 0
                ]

            if "position" in skeleton_types:
                skeleton_data["position"][tree_id] = [
                    (node[ "x" ], node[ "y" ], node[ "z" ])
                    for node in node_data
                ]

            if "thickness" in skeleton_types:
                skeleton_data["thickness"][tree_id] = [
                    node[ "thickness" ]
                    for node in node_data if node[ "parent" ]
                ]

        return {
            "skeleton_data": skeleton_data,
            "tree_ids": tree_ids
        }

    @Decorators.cached
    def _calculate_skeletons_configuration(self,
                                           skeleton_types: Optional[List[str]] = None
                                           ) -> dict:
        """ Create configuration object from given skeletons options. """
        return {
            "skeleton_types": skeleton_types
        }

    def _calculate_skeletons(self,
                             splits: dict,
                             configuration: dict) -> dict:
        """ Calculate skeleton features using provided splits and configuration. """

        split_skeletons = {
            split_name: {
                "skeletons": self._calculate_skeleton_features(
                    skeleton_dict=split_data,
                    skeleton_types=configuration["skeleton_types"]
                )
            }
            for split_name, split_data in splits.items()
        }

        # TODO - Display info about skeletons?
        """
        self.display_skeleton_information(
            skeletons=split_skeletons,
            display_skeletons=self.c.debug_display_skeletons_features
        )
        """

        return {
            "data": split_skeletons,
            "configuration": configuration
        }

    @Decorators.cached
    def calculate_skeletons(self,
                            skeleton_types: Optional[List[str]] = None
                            ) -> dict:
        """
        Calculate image features and return dictionary containing training, validation
        and test data.

        :param skeleton_types: Type of skeleton data requested:
            "segment", "position", "thickness.

        :return: Returns dictionary containing image feature data.
        """

        split_scheme = self._generate_splits(
            split_scheme=self.c.data_split,
            cross_validation_split=self.c.cross_validation_split
        )
        splits = self._generate_split_skeleton_dicts(
            split_scheme=split_scheme, skeleton_types=skeleton_types)

        configuration = self._calculate_skeletons_configuration(
            skeleton_types=skeleton_types
        )

        return self._calculate_skeletons(
            splits=splits, configuration=configuration
        )

    @Decorators.cached
    def calculate_skeletons_for_data(self,
                                     data_loader: BaseDataLoader,
                                     configuration: dict) -> dict:
        """ Calculate all features for data loaded in given data_loader. """

        split_scheme = self._generate_splits(split_scheme=(1.0, ), data_loader=data_loader, split_names=["data"])
        splits = self._generate_split_skeleton_dicts_custom(
            split_scheme=split_scheme, data_loader=data_loader,
            skeleton_types=configuration["skeleton_types"]
        )

        return self._calculate_skeletons(
            splits=splits, configuration=configuration
        )

    def scale_data(self, data: np.array, scaler_name: str = "none") -> Tuple[np.array, any]:
        """
        Perform scaling on provided data.

        :param data: Input data to scale.
        :param scaler_name: Scaler to use, must be one of:
            "standard", "minmax", "maxabs", "robust"
            "probust", "quantile", "power", "normal",
            "none"
        :return: Returns scaled data and the scaler instance which may be used to
            reverse the operation.
        """

        scaler = Scaler(scaler_type=scaler_name)

        scaler.fit(x=data)
        result = scaler.transform(x=data)

        return result, scaler

    def unscale_tree_data(self, data: np.array, scaler: any) -> np.array:
        """
        Perform un-scaling on provided data.

        :param data: Input data to reverse scaling on.
        :param scaler: Scaler instance returned by the scaling function.
        :return: Returns data in the same format with the scaling operation reversed.
        """

        return scaler.inverse_transform(x=data)

    def scale_tree_data(self, data: dict, scaler_name: str = "none") -> Tuple[dict, any]:
        """ Perform scaling on provided data, formatted as dictionary of tree_id -> values. Returns data and scaler. """

        arr_ids = list(data.keys())
        arr_data = np.concatenate([ data[tree_id] for tree_id in arr_ids ], axis=0)

        scaled_arr_data, scaler = self.scale_data(data=arr_data, scaler_name=scaler_name)

        result = {
            tree_id: scaled_arr_data[idx]
            for idx, tree_id in enumerate(arr_ids)
        }

        return result, scaler

    def unscale_tree_data(self, data: dict, scaler: any) -> dict:
        """ Perform un-scaling on provided data, formatted as dictionary of tree_id -> values. Returns data. """

        arr_ids = list(data.keys())
        arr_data = np.concatenate([ data[tree_id] for tree_id in arr_ids ], axis=0)

        unscaled_arr_data = self.unscale_tree_data(data=arr_data, scaler=scaler)

        result = {
            tree_id: unscaled_arr_data[idx]
            for idx, tree_id in enumerate(arr_ids)
        }

        return result

    def _display_meta_feature_information(self, meta_features: dict, split_name: str, indent: str):
        """ Display information about provided meta-data features. """

        self.__l.info(f"{indent}\tTrees: {meta_features['tree_ids']}")
        self.__l.info(f"{indent}\tStatistics: {meta_features['stat_names']}")

    def _display_hist_feature_information(self, hist_features: dict, split_name: str, indent: str):
        """ Display information about provided histogram features. """

        if "vis_buckets" in hist_features:
            buckets = hist_features["vis_buckets"]
        elif "buckets" in hist_features:
            self.__l.info(f"{indent}No visualization buckets, falling back to buckets")
            buckets = hist_features["buckets"]
        else:
            self.__l.warn(f"{indent}No buckets, invalid data!")
            return

        hist = hist_features.get("hist", None)
        hist_norm = hist_features.get("hist_norm", None)
        hist_all_norm = hist_features.get("hist_all_norm", None)
        feature_names = buckets.keys()

        IMAGES_PER_PLOT = 8
        idx = -1

        self.__l.info(f"{indent}Histograms for {split_name}: ")
        for idx, stat_name in enumerate(feature_names):
            self.__l.info(f"{indent}\t\"{stat_name}\"")

            bs = buckets[stat_name]
            hss = [ ]
            dfs = [ ]
            names = [ ]

            def gen_hss_dfs(hist: dict, name: str):
                names.append(name)
                hss.append([(n, bs[i], v) for n, h in
                            hist[stat_name].items() for i, v in enumerate(h["counts"])])
                dfs.append(pd.DataFrame(data=hss[-1], columns=("tree_id", "bucket", "count")))

            if hist is not None:
                gen_hss_dfs(hist, "Density")

            if hist_norm is not None:
                gen_hss_dfs(hist_norm, "Normalized")

            if hist_all_norm is not None:
                gen_hss_dfs(hist_all_norm, "All Normalized")

            if len(hss) == 0:
                self.__l.warn(f"{indent}\t\tNo histogram data found!")
                continue

            if (idx + 1) % IMAGES_PER_PLOT == 1:
                fig, axes = plt.subplots(IMAGES_PER_PLOT, len(hss), figsize=(15, 15), sharey=False)
                fig.suptitle(f"Histograms for {split_name}", fontsize=16)

            for name, df, axis in zip(names, dfs, axes[idx % IMAGES_PER_PLOT]):
                sns.histplot(x="bucket", y=None, hue="tree_id", weights="count", data=df.iloc[:], bins=list(bs),
                             legend=False, element="poly", ax=axis)
                axis.set_title(f"{stat_name}: {name}")

            if (idx + 1) % IMAGES_PER_PLOT == 0:
                plt.show()
        if (idx + 1) % IMAGES_PER_PLOT != 0:
            plt.show()

    def _display_stats_feature_information(self, stats_features: dict, split_name: str, indent: str):
        """ Display information about provided statistics features. """

        self.__l.info(f"{indent}Statistics for {split_name}: ")

        value_stats_names = stats_features["stats_elements"]
        stats_values = stats_features["stats"]

        stat_names = np.sort(np.unique([
            stat_name
            for stat_name in stats_values.keys()
        ]))

        tree_ids = np.sort(np.unique(tuple_array_to_numpy([
            tree_id
            for stats in stats_values.values()
            for tree_id in stats.keys()
        ])))

        columns = np.concatenate([[ "stat_name", "tree_id" ], value_stats_names])
        values_stats_data = [
            np.concatenate([[ stat_name, tree_id ], np.array([
                stats_values[stat_name][tree_id][value_stat_name]
                for value_stat_name in value_stats_names
            ]).reshape((-1, ))])
            for stat_name in stat_names
            for tree_id in tree_ids
        ]

        df = pd.DataFrame(data=values_stats_data, columns=columns)
        df.set_index(["stat_name", "tree_id"], drop=True, inplace=True)

        self.__l.info(f"{indent}\tValues: \n{str(df.to_string())}")

    def _display_image_feature_information(self, image_features: dict, split_name: str, indent: str):
        """ Display information about provided image features. """

        self.__l.info(f"{indent}Images for {split_name}: ")

        IMAGES_PER_ROW = 5
        IMAGES_PER_COL = 5
        IMAGES_PER_PLOT = IMAGES_PER_ROW * IMAGES_PER_COL

        for image_name, images in image_features["images"].items():
            self.__l.info(f"{indent}\tImage {image_name}: ")
            idx = -1
            for idx, (tree_id, image) in enumerate(images.items()):
                self.__l.info(f"{indent}\t\tTree #{tree_id}: {image.name}, {image.description}, {image.data.shape}")
                if (idx + 1) % IMAGES_PER_PLOT == 1:
                    fig, axes = plt.subplots(IMAGES_PER_ROW, IMAGES_PER_COL, figsize=(15, 15))
                    fig.suptitle(f"{image_name}", fontsize=16)

                ax = plt.subplot(IMAGES_PER_ROW, IMAGES_PER_COL, (idx % IMAGES_PER_PLOT) + 1)
                ax.set_title(f"Tree #{tree_id}")
                plt.imshow(image.data.squeeze(), origin="lower")

                if (idx + 1) % IMAGES_PER_PLOT == 0:
                    plt.show()
            if (idx + 1) % IMAGES_PER_PLOT != 0:
                plt.show()

    def _display_others_feature_information(self, others_features: dict, split_name: str, indent: str):
        """ Display information about provided others features. """

        self.__l.info(f"{indent}Others for {split_name}: ")

        others_names = others_features["others_names"]
        tree_ids = np.sort(np.unique([
            tree_id
            for stats in others_features["others"].values()
            for tree_id in stats.keys()
        ]))

        self.__l.info(f"{indent}\tFeatures: {others_names}")

        columns = np.concatenate([[ "tree_id" ], others_names])
        others_data = [
            np.concatenate([
                [tree_id] + [
                    other_values[tree_id]
                    for other_name, other_values in others_features["others"].items()
                ]
            ])
            for tree_id in tree_ids
        ]

        df = pd.DataFrame(data=others_data, columns=columns)
        df.set_index(["tree_id"], drop=True, inplace=True)
        df.index = df.index.astype(np.int)

        self.__l.info(f"{indent}\tValues: \n{str(df.to_string())}")

    def display_feature_information(self, features: dict,
                                    display_meta: bool = True,
                                    display_hist: bool = True,
                                    display_stats: bool = True,
                                    display_image: bool = True,
                                    display_others: bool = True):
        """ Display information about provided features. """

        if not display_meta and not display_hist and \
            not display_stats and not display_image:
            return

        self.__l.info("Describing features: ")

        for split_name, split_data in features.items():
            self.__l.info(f"Split \"{split_name}\"")

            if display_meta:
                self.__l.info("Meta data: ")
                if "meta" in split_data:
                    self._display_meta_feature_information(
                        split_data["meta"], split_name=split_name, indent="\t")
                else:
                    self.__l.info("\tNo data.")

            if display_hist:
                self.__l.info("Histogram data: ")
                if "hist" in split_data:
                    self._display_hist_feature_information(
                        split_data["hist"], split_name=split_name, indent="\t")
                else:
                    self.__l.info("\tNo data.")

            if display_stats:
                self.__l.info("Statistics data: ")
                if "stats" in split_data:
                    self._display_stats_feature_information(
                        split_data["stats"], split_name=split_name, indent="\t")
                else:
                    self.__l.info("\tNo data.")

            if display_image:
                self.__l.info("Image data: ")
                if "image" in split_data:
                    self._display_image_feature_information(
                        split_data["image"], split_name=split_name, indent="\t")
                else:
                    self.__l.info("\tNo data.")

            if display_others:
                self.__l.info("Other data: ")
                if "others" in split_data:
                    self._display_others_feature_information(
                        split_data["others"], split_name=split_name, indent="\t")
                else:
                    self.__l.info("\tNo data.")

        self.__l.info("\tFeature description end.")

    def _display_view_features_information(self, view_features: dict, split_name: str, indent: str):
        """ Display information about provided view features. """

        self.__l.info(f"{indent}Views for {split_name}: ")

        tree_ids = np.sort(np.unique([
            tree_id
            for view_types in view_features["views"].values()
            for tree_id in view_types.keys()
        ]))

        view_types = np.sort(np.unique([
            view_type
            for view_type in view_features["views"].keys()
        ]))

        view_count = np.max([
            len(tree_values)
            for view_types in view_features["views"].values()
            for tree_values in view_types.values()
        ])

        VIEWS_PER_ROW = view_count
        VIEWS_PER_COL = len(view_types)

        for tree_id in tree_ids:
            self.__l.info(f"{indent}\tViews for Tree #{tree_id}: ")

            fig, axes = plt.subplots(VIEWS_PER_COL, VIEWS_PER_ROW, figsize=(15, 15))
            fig.suptitle(f"Tree #{tree_id}", fontsize=16, backgroundcolor="white")

            for row_idx, view_type in enumerate(view_types):
                self.__l.info(f"{indent}\t\tType \"{view_type}\": ")
                for view_id in range(view_count):
                    tree_view = view_features["views"][view_type].get(tree_id, None)
                    view = tree_view.get(view_id, None) if tree_view is not None else None

                    if view is None:
                        self.__l.info(f"{indent}\t\tView #{view_id}: Not Available!")
                        continue

                    self.__l.info(f"{indent}\t\tView #{view_id}: {view.name}, {view.description}, {view.data.shape}")
                    ax = plt.subplot(VIEWS_PER_COL, VIEWS_PER_ROW, row_idx * VIEWS_PER_ROW + view_id + 1)
                    ax.set_title(f"{view_type}#{view_id}")
                    plt.imshow(view.data.squeeze())

            plt.show()

    def display_view_information(self, views: dict,
                                 display_views: bool = True):
        """ Display information about provided views. """

        if not display_views:
            return

        self.__l.info("Describing views: ")

        for split_name, split_data in views.items():
            self.__l.info(f"Split \"{split_name}\"")

            if display_views:
                self.__l.info("Views: ")
                if "views" in split_data:
                    self._display_view_features_information(
                        split_data["views"], split_name=split_name, indent="\t")
                else:
                    self.__l.info("\tNo data.")

        self.__l.info("\tView description end.")

    def _display_view_choice_information(self, choices: pd.DataFrame, split_name: str, indent: str):
        """ Display information about provided tree-view choices. """

        self.__l.info(f"{indent}View choices for {split_name}: ")
        self.__l.info(f"{indent}\tTotal choices: {len(choices)}")

        first_choices = choices.rename({"first_tree_id": "tree_id", "first_view_id": "view_id"}, axis=1)
        first_choices[["first_tree_id", "first_view_id"]] = first_choices[["tree_id", "view_id"]]
        first_choices["score"] = (first_choices["choice"] == 1) * 1.0
        second_choices = choices.rename({"second_tree_id": "tree_id", "second_view_id": "view_id"}, axis=1)
        second_choices[["second_tree_id", "second_view_id"]] = second_choices[["tree_id", "view_id"]]
        second_choices["score"] = (second_choices["choice"] == 2) * 1.0

        per_tree_choices = pd.concat(
            [ first_choices, second_choices ],
            axis=0
        )

        tree_ids = per_tree_choices.tree_id.unique()
        tree_id_count = len(tree_ids)
        tree_id_dict = {
            tree_id: tree_idx
            for tree_idx, tree_id in enumerate(tree_ids)
        }
        view_ids = per_tree_choices.view_id.unique()
        view_id_count = len(view_ids)
        view_id_dict = {
            view_id: view_idx
            for view_idx, view_id in enumerate(view_ids)
        }

        tree_view_score_matrix = np.zeros((tree_id_count * view_id_count, tree_id_count * view_id_count))
        tree_score_matrix = np.zeros((tree_id_count, tree_id_count))
        for row_id, row_data in choices.iterrows():
            won_tree_id = tree_id_dict[row_data["first_tree_id"] if row_data["choice"] == 1 else row_data["second_tree_id"]]
            won_view_id = view_id_dict[row_data["first_view_id"] if row_data["choice"] == 1 else row_data["second_view_id"]]
            lost_tree_id = tree_id_dict[row_data["first_tree_id"] if row_data["choice"] == 2 else row_data["second_tree_id"]]
            lost_view_id = view_id_dict[row_data["first_view_id"] if row_data["choice"] == 2 else row_data["second_view_id"]]

            tree_view_score_matrix[won_tree_id * view_id_count + won_view_id, lost_tree_id * view_id_count + lost_view_id] += 1.0
            tree_score_matrix[won_tree_id, lost_tree_id] += 1.0

        sns.heatmap(tree_score_matrix)
        plt.show()

        sns.heatmap(tree_view_score_matrix)
        plt.show()

    def _display_scores_information(self, scores: dict, indexed_scores: pd.DataFrame, split_name: str, indent: str):
        """ Display information about provided scores. """

        self.__l.info(f"{indent}Scores for {split_name}: ")

        tree_scores = indexed_scores.reset_index()
        tree_scores = tree_scores.loc[tree_scores.view_id == -1]

        sns.lineplot(x="tree_id", y="jod", data=tree_scores, estimator=None)
        plt.show()

    def display_score_information(self, scores: dict,
                                  display_view_choices: bool = True,
                                  display_scores: bool = True):
        """ Display information about provided scores. """

        if not display_view_choices and not display_scores:
            return

        self.__l.info("Describing scores: ")

        for split_name, split_data in scores.items():
            self.__l.info(f"Split \"{split_name}\"")

            if display_view_choices:
                self.__l.info("Choices: ")
                if "tree_view_choices" in split_data:
                    self._display_view_choice_information(
                        split_data["tree_view_choices"], split_name=split_name, indent="\t")
                else:
                    self.__l.info("\tNo data.")

            if display_scores:
                self.__l.info("Scores: ")
                if "scores" in split_data and "indexed_scores" in split_data:
                    self._display_scores_information(
                        split_data["scores"], split_data["indexed_scores"], split_name=split_name, indent="\t")
                else:
                    self.__l.info("\tNo data.")

        self.__l.info("\tScore description end.")

    @property
    def data_loader(self) -> DataLoader:
        """ Access the DataLoader used by this instance. """

        return self._data_loader

    def process(self):
        """ Export features if cache was provided. """

        self.__l.info("Starting feature export operations...")

        if self.c.feature_cache_location:
            self._save_feature_cache(cache_location=self.c.feature_cache_location)

        if self.c.debug_calculate_features:
            self.calculate_features()

        if self.c.debug_calculate_views:
            self.calculate_views()

        if self.c.debug_calculate_scores:
            self.calculate_scores()

        self.__l.info("\tFeature operations finished!")




