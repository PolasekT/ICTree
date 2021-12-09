# -*- coding: utf-8 -*-

"""
Dataset wrappers and helper functions.
"""

import copy
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union, Set

from perceptree.common.pytorch_safe import *

import numpy as np
import pandas as pd
import sklearn.utils as sku

from perceptree.common.logger import Logger
from perceptree.common.cache import Cache
from perceptree.common.cache import CacheDict
from perceptree.common.serialization import Serializer
from perceptree.common.util import dict_of_lists
from perceptree.common.util import numpy_array_to_tuple_numpy
from perceptree.common.util import tuple_array_to_numpy
from perceptree.common.util import recurse_dictionary_endpoint
from perceptree.common.util import reshape_scalar
from perceptree.data.loader import BaseDataLoader
from perceptree.data.treeio import TreeFile
from perceptree.data.treeio import TreeImage
from perceptree.data.treeio import TreeStatistic
from perceptree.data.featurizer import DataFeaturizer

import matplotlib.pyplot as plt
import seaborn as sns


class DatasetConfig(Logger):
    """ All in one configuration of datasets. """

    def __init__(self, featurizer: DataFeaturizer, data_loader: Optional[BaseDataLoader],
                 initial_config: Optional["DatasetConfig"]):
        super().__init__()

        self._featurizer = featurizer
        self._data_loader = data_loader or self._featurizer.data_loader
        self._feature_config = Cache()
        self._available_features = None
        self.reset_defaults(initial_config=initial_config)

    def __getstate__(self) -> dict:
        """ Get serializable state. """

        excluded_attributes = [ "_featurizer", "_data_loader" ]

        return {
            key: self.__dict__[key]
            for key in self.__dict__.keys()
            if key not in excluded_attributes
        }

    def __setstate__(self, state):
        """ Set de-serialized state. """

        self.__dict__ = state

        self._featurizer = Serializer.DUMP_CONFIG.get_instance(DataFeaturizer)
        self._data_loader = self._featurizer.data_loader

    @property
    def featurizer(self) -> DataFeaturizer:
        """ Access the featurizer. """
        return self._featurizer

    @property
    def data_loader(self) -> BaseDataLoader:
        """ Access the data_loader. """
        return self._data_loader

    def reset_defaults(self, initial_config: Optional["DatasetConfig"]) -> "DatasetConfig":
        """ Reset the config to default values where everything is disabled. """

        if initial_config is not None:
            available_features = initial_config._available_features
        else:
            available_features = self._data_loader.available_features
        self._available_features = available_features
        feature_elements = {
            "stat": np.concatenate([
                self._featurizer.stat_elements(),
                [
                    "hist",
                ]
            ]),
            "image": [ "image" ],
            "other": available_features["other"],
            "view": [ "image" ],
            "skeleton": [ "data" ]
        }

        # Initialize all features to be disabled by default.
        self._feature_config = Cache({
            category_name: {
                feature_name: {
                    element_name: False
                    for element_name in (feature_elements[category_name]
                                         if category_name != "other" else
                                         feature_elements[category_name][feature_name])
                }
                for feature_name in features
            }
            for category_name, features in available_features.items()
        })
        self._feature_config["score"] = CacheDict({
            "value": False
        })

        # Initialize all featurization properties to defaults.
        featurizer_config = CacheDict({
            # Split definition, use names or dictionary of specific ids.
            "splits": None,
            # Tree ID filter, optionally also containing views.
            "tree_filter": None,
            # First quantile used as a dust bin in histograms.
            "quant_start": 0.001,
            # Last quantile used as a dust bin in histograms.
            "quant_end": 0.999,
            # Total number of buckets in the histograms.
            "total_buckets": 32,
            # Calclulate histogram buckets from split (True) or from all data (False).
            "buckets_from_split": True,
            # Perform feature normalization?
            "normalize_features": False,
            # Perform feature standardization?
            "standardize_features": False,
            # Resolution to convert the images to, None for no resize.
            "image_resolution": None,
            # Interpolation used in image resizing, None for automatic.
            "image_interpolation": None,
            # Resolution to convert the views to, None for no resize.
            "view_resolution": None,
            # Interpolation used in view resizing, None for automatic.
            "view_interpolation": None,
            # Flatten the views into 1 dimensional data?
            "view_flatten": True,
            # Additional transform applied to all of the views.
            "view_transform": None,
            # Generate name array for features?
            "generate_names": True,
            # Comparison mode for pairwise answers (True) or scores (False).
            "comparison_mode": False,
            # Use score differences in the comparison mode (True) or just binary 1/2 choice (false).
            "comparison_score_difference": False,
            # Use view scores in the comparison mode (True) or the tree scores (false).
            "comparison_view_scores": False,
            # Sample only given percentage of samples for comparison mode.
            "comparison_sample_ptg": None,
            # Seed used for the random comparison sample percentage selection.
            "comparison_sample_ptg_seed": 42,
            # Use per-view scores when applicable?
            "use_view_scores": False,
            # Use tree variants in training pairs?
            "use_tree_variants": False,
            # Use views variants in training pairs?
            "use_view_variants": False,
            # Pre-generate all variant cases (True) or randomly choose at runtime (False)?
            "pre_generate_variants": False,
            # Seed used for the random sample generation.
            "rng_seed": 42,
            # Load pre-configured data?
            "data_config": None,
            # Split skeleton positions into special features?
            "skeleton_position_split": True,
            # Print progress messages for dataset preparation?
            "verbose_loading": True,
            # Perform index resolving operations?
            "resolve_indicies": True,
        })
        self._feature_config["featurizer"] = featurizer_config

    def set_data_configuration(self, data_configuration: Optional[dict]) -> "DatasetConfig":
        """  Set data configuration. If None, this operation is null."""

        if data_configuration is not None:
            self.set_featurizer_option("data_config", data_configuration)

        return self

    def enable_hist(self, enabled: bool = True) -> "DatasetConfig":
        """ Set all histogram features to enabled. """
        self.set("^stat.\\.*.hist\\.*$", enabled); return self

    def enable_stat(self, enabled: bool = True) -> "DatasetConfig":
        """ Set all statistical features to enabled. """
        self.set("^stat.\\.*.(?!hist)\\.*$", enabled); return self

    def enable_image(self, enabled: bool = True) -> "DatasetConfig":
        """ Set all image features to enabled. """
        self.set("^image.\\.*.image$", enabled); return self

    def enable_other(self, enabled: bool = True) -> "DatasetConfig":
        """ Set all other features to enabled. """
        self.set("^other.\\.*.\\.*$", enabled); return self

    def enable_view(self, enabled: bool = True) -> "DatasetConfig":
        """ Set all view features to enabled. """
        self.set("^view.\\.*.image$", enabled); return self

    def enable_skeleton(self, enabled: bool = True) -> "DatasetConfig":
        """ Set all skeleton features to enabled. """
        self.set("^skeleton.\\.*.data$", enabled); return self

    def enable_score(self, enabled: bool = True) -> "DatasetConfig":
        """ Set all score features to enabled. """
        self.set("score.value", enabled); return self

    def enable(self, path: str) -> "DatasetConfig":
        """ Enable all feature elements within given path - e.g. 'stat.hist' or 'stat'. """
        return self.set(path, True)

    def disable(self, path: str) -> "DatasetConfig":
        """ Disable all feature elements within given path - e.g. 'stat.hist' or 'stat'. """
        return self.set(path, False)

    @staticmethod
    def _recurse_endpoint(path_dict: CacheDict, path: str, name: str) -> Iterable:
        """ Perform recursive descent on end-points with dict, name and path. """

        if name not in path_dict:
            raise RuntimeError(f"Found invalid name in recurse_endpoint \"{name}\"!")
        elif isinstance(path_dict[name], dict):
            for n in path_dict[name].keys():
                yield from DatasetConfig._recurse_endpoint(
                    path_dict=path_dict[name],
                    path=Cache.CACHE_PATH_SEP.join(filter(None, [path, name])),
                    name=n
                )
        else:
            yield path_dict, name, path
        #else:
            #raise RuntimeError(f"Found invalid type in recurse_endpoint \"{name}\" -> \"{type(path_dict[name])}\"!")

    def _for_each_path_name(self, path: str) -> Iterable:
        """ Iterate over all end-points within given path. """

        path_dict_it = self._feature_config.get_path_dict_reg(regex_path=path)
        if path_dict_it is None:
            raise RuntimeError(f"Unknown feature \"{path}\" specified!")
        path_dicts = list(path_dict_it)
        if len(path_dicts) == 0 and path.find("*") == -1:
            # Only error when no * wildcards are present.
            raise RuntimeError(f"Unknown feature \"{path}\" specified!")

        for path_dict, name, dict_path in path_dicts:
            yield from DatasetConfig._recurse_endpoint(
                path_dict=path_dict, name=name, path=dict_path
            )

    def set(self, path: str, enabled: bool) -> "DatasetConfig":
        """ Set all feature elements within given path - e.g. 'stat.hist' or 'stat'. """

        for d, n, p in self._for_each_path_name(path):
            d[n] = enabled

        return self

    def set_value(self, path: str, value: any) -> "DatasetConfig":
        """ Set property to given value. """

        self._feature_config.set_path(path=path, value=value, create=True)

        return self

    def set_featurizer_option(self, path: str, value: any) -> "DatasetConfig":
        """ Set featurizer property to given value. """

        self._feature_config.set_path(path=f"featurizer.{path}", value=value, create=True)

        return self

    def get_featurizer_option(self, path: str) -> Union[CacheDict, any]:
        """ Get featurizer property to given value. """

        return self.__getitem__(path=f"featurizer.{path}")

    def set_options(self, options: dict) -> "DatasetConfig":
        """ Set all options in given dictionary to requested values. """

        for d, n, p in recurse_dictionary_endpoint(options):
            self.set_value(path=f"{p}{Cache.CACHE_PATH_SEP}{n}", value=d[n])

        return self

    def any(self, path: str) -> bool:
        """ Is anything enabled in given path? """

        return np.any([ d[n] for d, n, p in self._for_each_path_name(path) if type(d[n]) == bool ])

    def features_required(self) -> bool:
        """ Are any features requested? """

        return any([ self.any("stat"), self.any("image"), self.any("other") ])

    def enabled_hist_names(self) -> List[str]:
        """ Get list of statistics with enabled histogram features. """

        return [ p.split(Cache.CACHE_PATH_SEP)[1] for p in self.get_paths("^stat.\\.*.hist$", True)]

    def enabled_stat_names(self) -> Dict[str, List[str]]:
        """ Get list of statistics with enabled statistic features and their elements. """

        return dict_of_lists([
            [ split[1], split[-1] ]
            for p in self.get_paths("^stat.\\.*.(?!hist)\\.*$", True)
            for split in [ p.split(Cache.CACHE_PATH_SEP) ]
        ])

    def enabled_image_names(self) -> List[str]:
        """ Get list of statistics with enabled image features. """

        return [ p.split(Cache.CACHE_PATH_SEP)[1] for p in self.get_paths("^image.\\.*.image$", True)]

    def enabled_other_names(self) -> Dict[str, List[str]]:
        """ Get list of enabled other features and their elements. """

        return dict_of_lists([
            [ split[1], split[-1] ]
            for p in self.get_paths("^other.\\.*.\\.*$", True)
            for split in [ p.split(Cache.CACHE_PATH_SEP) ]
        ])

    def views_required(self) -> bool:
        """ Are any views requested? """

        return self.any("view")

    def enabled_view_names(self) -> List[str]:
        """ Get list of view with enabled image features. """

        return [ p.split(Cache.CACHE_PATH_SEP)[1] for p in self.get_paths("^view.\\.*.image$", True)]

    def skeletons_required(self) -> bool:
        """ Are any skeletons requested? """

        return self.any("skeleton")

    def enabled_skeleton_data(self) -> List[str]:
        """ Get list of data features requested for the skeleton. """

        return [ p.split(Cache.CACHE_PATH_SEP)[1] for p in self.get_paths("^skeleton.\\.*.data$", True)]

    def score_required(self) -> bool:
        """ Is score requested? """

        return self.__getitem__(path="score.value")

    def comparison_view_scores(self) -> bool:
        """ Should view scores be used? """

        return self.get_featurizer_option("comparison_view_scores")

    def use_view_scores(self) -> bool:
        """ Should view scores be used? """

        return self.get_featurizer_option("use_view_scores")

    def comparison_required(self) -> (bool, bool):
        """ Is comparison data requested? Should score differences be used? """

        return self.get_featurizer_option("comparison_mode"), \
               self.get_featurizer_option("comparison_score_difference")

    def get_paths(self, path: str, value: any) -> List[str]:
        """ Get list of paths which have corresponding value. """

        return list([
            Cache.CACHE_PATH_SEP.join(filter(None, [ p, n])) for d, n, p in self._for_each_path_name(path)
            if type(d[n]) == type(value) and d[n] == value
        ])

    def __getitem__(self, path: str) -> Union[CacheDict, any]:
        """ Get item specified by the path. """
        return self._feature_config.get_path(path=path, create=False, none_when_missing=False)

    def __setitem__(self, path: str, value: any):
        """ Set item specified by given path - it must be an end-point. """
        self._feature_config.set_path(path=path, value=value, create=False)


class TreeDataset(td.Dataset, Logger):
    """ Simple dataset wrapper around tree data paired with their score. """

    def __init__(self, config: DatasetConfig):
        super().__init__()

        self._config = config
        self._featurizer = self._config.featurizer
        self._data_loader = self._config.data_loader

        self._rng = np.random.default_rng(self._config.get_featurizer_option("rng_seed"))
        self._verbose = self._config.get_featurizer_option("verbose_loading")
        self._resolve_indices = self._config.get_featurizer_option("resolve_indices")

        self._data, self._configuration = self._calculate_data()

        self._current_splits = None
        self._current_inputs = None
        self._current_outputs = None
        self._current_indices = None

        self.set_current_splits(splits=None)

    def _prepare_split_tree_ids(self, features: dict, views: dict,
                                split_ids: List[str],
                                tree_filter: Optional[Union[List[Tuple[int, int]], Dict[Tuple[int, int], Set[Tuple[int, int]]]]]
                                ) -> dict:
        """ Prepare filtered tree id splits. """

        if len(features) != 0:
            split_tree_ids = {
                split: features["data"][split]["meta"]["tree_ids"]
                for split in split_ids
            }
        elif len(views) != 0:
            split_tree_ids = {
                split: views["data"][split]["views"]["tree_ids"]
                for split in split_ids
            }
        else:
            split_ids = { }

        if tree_filter is not None:
            allowed_tree_ids = tree_filter if isinstance(tree_filter, list) else list(tree_filter.keys())
            allowed_tree_ids = tuple_array_to_numpy(allowed_tree_ids)
            split_tree_ids = {
                split_name: np.intersect1d(split_ids, allowed_tree_ids)
                for split_name, split_ids in split_tree_ids.items()
            }

        return split_tree_ids

    def _prepare_split_tree_view_ids(self,
                                     features: dict, views: dict, scores: dict,
                                     split_ids: List[str],
                                     tree_filter: Optional[Union[List[int], Dict[int, Set[int]]]]
                                     ) -> dict:
        """ Prepare filtered tree view id splits. """

        if views:
            all_tree_view_ids = {
                split: np.unique(tuple_array_to_numpy([
                    ( tree_id, view_id )
                    for view_name, view_data in views["data"][split]["views"]["view_data"].items()
                    for tree_id, tree_views in view_data.items()
                    for view_id in tree_views.keys()
                ], axis=-2))
                for split in split_ids
            }
        elif scores:
            all_tree_view_ids = {
                split: np.unique(tuple_array_to_numpy([
                    ( tree_id, view_id )
                    for tree_id, tree_data in scores["data"][split]["spherical_scores"].items()
                    for view_id, view_data in tree_data.items()
                    if isinstance(view_id, tuple)
                ], axis=-2))
                for split in split_ids
            }
        else:
            all_tree_view_ids = { }

        if tree_filter is not None:
            allowed_tree_view_ids = tree_filter if isinstance(tree_filter, list) else [
                ( tree_id, view_id )
                for tree_id, tree_view_ids in tree_filter.items()
                for view_id in tree_view_ids
            ]
            allowed_tree_view_ids = tuple_array_to_numpy(allowed_tree_view_ids, axis=-2)
            filtered_tree_view_ids = { }
            for split_name, split_view_ids in all_tree_view_ids.items():
                allowed = [ ]
                for view_id in split_view_ids:
                    for allowed_id in allowed_tree_view_ids:
                        if view_id[0] == allowed_id[0] and (view_id[1] == allowed_id[1] or allowed_id[1][0] < 0):
                            allowed.append(view_id)
                            break
                filtered_tree_view_ids[split_name] = tuple_array_to_numpy(allowed, axis=-2)
            all_tree_view_ids = filtered_tree_view_ids

        return all_tree_view_ids

    def _prepare_tree_data(self, features: dict, views: dict, scores: dict,
                           split_ids: List[str],
                           tree_filter: Optional[Union[List[int], Dict[int, Set[int]]]]
                           ) -> dict:
        """ Prepare tree data aggregate split from given list of split identifiers. """

        split_tree_ids = self._prepare_split_tree_ids(
            features=features, views=views,
            split_ids=split_ids, tree_filter=tree_filter)
        all_tree_ids = np.concatenate([
            tree_ids for tree_ids in split_tree_ids.values()
        ]) if split_tree_ids else np.array([ ])

        hist_names = self._config.enabled_hist_names()
        stat_names = self._config.enabled_stat_names()
        image_names = self._config.enabled_image_names()
        other_names = self._config.enabled_other_names()
        score_required = self._config.score_required()

        all_data = [ ]
        all_names = [ ]

        generate_names = self._config.get_featurizer_option("generate_names")

        if hist_names:
            hist_data = np.array([
                np.concatenate([
                    features["data"][split]["hist"]["hist"][hist_name][tree_id]["counts"]
                    for hist_name in hist_names
                ])
                for split, tree_ids in split_tree_ids.items()
                for tree_id in tree_ids
            ])
            all_data.append(hist_data)
            hist_data_names = np.array([
                np.concatenate([
                    [
                        f"hist:{hist_name}:{idx}"
                        for idx in range(len(features["data"][split]["hist"]["hist"][hist_name][tree_id]["counts"]))
                    ]
                    for hist_name in hist_names
                ])
                for split in split_ids[:1]
                for tree_id in split_tree_ids[split][:1]
            ]) if generate_names else np.array([ ])
            all_names.append(hist_data_names)

        if stat_names:
            stat_data = np.array([
                np.concatenate([
                    [
                        features["data"][split]["stats"]["stats"][stat_name][tree_id][stat_element]
                        for stat_element in stat_elements
                    ]
                    for stat_name, stat_elements in stat_names.items()
                ])
                for split, tree_ids in split_tree_ids.items()
                for tree_id in tree_ids
            ])
            stat_data = stat_data.reshape((stat_data.shape[0], -1)) if len(stat_data) > 0 else np.array([ ])
            all_data.append(stat_data)
            stat_data_names = np.array([
                np.concatenate([
                    [
                        f"stat:{stat_name}:{stat_element}:{idx}"
                        for stat_element in stat_elements
                        for idx in range(len(features["data"][split]["stats"]["stats"][stat_name][tree_id][stat_element]))
                    ]
                    for stat_name, stat_elements in stat_names.items()
                ])
                for split in split_ids[:1]
                for tree_id in split_tree_ids[split][:1]
            ]) if generate_names else np.array([ ])
            all_names.append(stat_data_names)

        if image_names:
            image_data = np.array([
                np.concatenate([
                    features["data"][split]["image"]["image_data"][image_name][tree_id].flatten()
                    for image_name in image_names
                ])
                for split, tree_ids in split_tree_ids.items()
                for tree_id in tree_ids
            ])
            all_data.append(image_data)
            image_data_names = np.array([
                np.concatenate([
                    [
                        f"image:{image_name}:{np.unravel_index(idx, features['data'][split]['image']['image_data'][image_name][tree_id].shape)}"
                        for idx in range(len(features["data"][split]["image"]["image_data"][image_name][tree_id].flatten()))
                    ]
                    for image_name in image_names
                ])
                for split in split_ids[:1]
                for tree_id in split_tree_ids[split][:1]
            ]) if generate_names else np.array([ ])
            all_names.append(image_data_names)

        if other_names:
            other_data = np.array([
                np.concatenate([
                    [
                        features["data"][split]["others"]["others"][f"{other_name}_{other_element}"][tree_id]
                        for other_element in other_elements
                    ]
                    for other_name, other_elements in other_names.items()
                ])
                for split, tree_ids in split_tree_ids.items()
                for tree_id in tree_ids
            ])
            other_data = other_data.reshape((other_data.shape[0], -1)) if len(other_data) > 0 else np.array([ ])
            all_data.append(other_data)
            other_data_names = np.array([
                np.concatenate([
                    [
                        f"other:{other_name}_{other_element}:{idx}"
                        for other_element in other_elements
                        for idx in range(len(features["data"][split]["others"]["others"][f"{other_name}_{other_element}"][tree_id]))
                    ]
                    for other_name, other_elements in other_names.items()
                ])
                for split in split_ids[:1]
                for tree_id in split_tree_ids[split][:1]
            ]) if generate_names else np.array([ ])
            all_names.append(other_data_names)

        if all_data and (np.array([ len(d) for d in all_data ]) > 0).all():
            all_data = np.concatenate(all_data, axis=1)
        else:
            all_data = np.array([ ])
        if generate_names and all_names and (np.array([ len(n) for n in all_names ]) > 0).all():
            all_names = np.concatenate(all_names, axis=1)[0]
        else:
            all_names = np.array([ ])

        score_data = np.array([
            scores["data"][split]["spherical_scores"][tree_id]["aggregate"]["jod"]
            for split, tree_ids in split_tree_ids.items()
            for tree_id in tree_ids
        ]).reshape((-1, 1)) if score_required else np.array([ ])

        return {
            "data": all_data,
            "scores": score_data,
            "names": all_names,
            "tree_ids": all_tree_ids
        }

    def _prepare_view_data(self, features: dict, views: dict, scores: dict,
                           split_ids: List[str],
                           tree_filter: Optional[Union[List[int], Dict[int, Set[int]]]]
                           ) -> dict:
        """ Prepare view data for aggregate split from given list of split identifiers. """

        view_names = self._config.enabled_view_names()
        score_required = self._config.score_required()

        split_tree_view_ids = self._prepare_split_tree_view_ids(
            features=features, views=views, scores=scores,
            split_ids=split_ids, tree_filter=tree_filter)
        all_tree_view_ids = np.unique(np.concatenate(
            list(split_tree_view_ids.values())
        )) if split_tree_view_ids else np.array([ ])

        all_data = [ ]
        view_shapes = [ ]

        if view_names:
            view_data = np.array([
                [
                    views["data"][split]["views"]["view_data"][view_name][tree_view_id[0]][tree_view_id[1]]
                    for view_name in view_names
                ]
                for split, tree_view_ids in split_tree_view_ids.items()
                for tree_view_id in tree_view_ids
                if tree_view_id[1][0] >= 0
            ])

            # Optionally apply the transform to each image.
            view_transform = self._config.get_featurizer_option("view_transform")
            if view_transform is not None:
                view_data = np.array([
                    [
                        np.array(view_transform(view))
                        for view in tree_views
                    ]
                    for tree_views in view_data
                ])

            view_shapes = [ view.shape for view in view_data[0] ] if len(view_data) else [ ]
            if self._config.get_featurizer_option("view_flatten"):
                view_data = view_data.reshape((view_data.shape[0], -1)) if len(view_data) > 0 else view_data
            all_data.append(view_data)

        all_data = np.concatenate(all_data, axis=1) if all_data else np.array([ ])

        score_data = np.array([
            scores["data"][split]["spherical_scores"][tree_view_id[0]][tree_view_id[1]]["jod"]
            for split, tree_view_ids in split_tree_view_ids.items()
            for tree_view_id in tree_view_ids
            if tree_view_id[1][0] >= 0
        ]).reshape((-1, 1)) if score_required else np.array([ ])

        return {
            "data": all_data,
            "scores": score_data,
            "view_names": view_names,
            "view_shapes": view_shapes,
            "tree_view_ids": all_tree_view_ids
        }

    def _prepare_comparison_data(self, features: dict, views: dict, scores: dict,
                                 split_ids: List[str],
                                 tree_filter: Optional[Union[List[int], Dict[int, Set[int]]]]
                                 ) -> dict:
        """ Prepare comparison data for aggregate split from given list of split identifiers. """

        data = np.array([
            (
                ( ( item[0], item[1] ), ( item[2], item[3] ) ),
                ( ( item[4], item[5] ), ( item[6], item[7] ) ),
                # Move the choice from { 1, 2 } to { 0.0, 1.0 }
                item[8] - 1.0
            )
            for split in split_ids
            for item in scores["data"][split]["tree_view_choices"][[
                "first_tree_id", "first_tree_variant_id", "first_view_id", "first_view_variant_id",
                "second_tree_id", "second_tree_variant_id", "second_view_id", "second_view_variant_id",
                "choice"
            ]].to_numpy()
        ], dtype=object) if scores else np.array([ ], dtype=object)

        return {
            "data": data,
        }

    def _prepare_skeleton_data(self, features: dict, views: dict, scores: dict,
                               skeletons: dict,
                               split_ids: List[str],
                               tree_filter: Optional[Union[List[int], Dict[int, Set[int]]]]
                               ) -> dict:
        """ Prepare skeleton data for aggregate split from given list of split identifiers. """

        skeleton_names = self._config.enabled_skeleton_data()

        split_tree_ids = self._prepare_split_tree_ids(
            features=features, views=views,
            split_ids=split_ids, tree_filter=tree_filter)
        all_tree_ids = np.concatenate([
            tree_ids for tree_ids in split_tree_ids.values()
        ]) if split_tree_ids else np.array([ ])

        data = [ ]
        tree_ids = [ ]
        split_position = self._config.get_featurizer_option("skeleton_position_split")

        position_present = False
        segment_present = False
        thickness_present = False

        if skeleton_names:
            for split in split_ids:
                for tree_id in split_tree_ids[split]:
                    skeleton_data = skeletons["data"][split]["skeleton_data"]
                    node_ids = skeleton_data["node_ids"]
                    node_count = len(node_ids)

                    node_features = None
                    edge_index = None
                    edge_attr = None
                    pos = None

                    node_feature_list = [ ]
                    if "position" in skeleton_data:
                        position_present = True
                        if split_position:
                            pos = skeleton_data["position"][tree_id]
                        else:
                            node_feature_list.append(np.reshape(skeleton_data["position"][tree_id], (node_count, -1, )))

                    if node_feature_list:
                        node_features = np.concatenate(node_feature_list, axis=0)

                    if "segment" in skeleton_data:
                        segment_present = True
                        edge_index = np.reshape(skeleton_data["segment"][tree_id], (node_count, 2, ))

                    edge_feature_list = [ ]
                    if "thickness" in skeleton_data:
                        thickness_present = True
                        edge_feature_list.append(np.reshape(skeleton_data["thickness"][tree_id], (node_count, -1,)))

                    if edge_feature_list:
                        edge_attr = np.concatenate(edge_feature_list, axis=0)

                    dat = tg.data.Data(
                        x=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        #TODO - Do we need to provide y= here?
                        pos=pos
                    )

                    data.append(dat)
                    tree_ids.append(tree_id)

        data_names = [ "position" ] if position_present else [ ]
        data_names += [ "segment" ] if segment_present else [ ]
        data_names += [ "thickness" ] if thickness_present else [ ]

        return {
            "data": np.array(data),
            "tree_ids": tree_ids,
            "names": data_names
        }

    def _tree_view_id_variants(self, tree_view_id: tuple, use_tree_variants: bool, use_view_variants: bool,
                               tree_view_ids: dict) -> List[tuple]:
        """ Get list of all possible variants for given starting tree/view identifier. """

        # Pattern matching: -1 in id -> all ids, -1 in variant or use_*_variants -> all variants.
        return [
            tree_view_id_t
            for tree_view_id_t in tree_view_ids
            if (tree_view_id[0][0] == -1 or tree_view_id[0][0] == tree_view_id_t[0][0]) and \
               (tree_view_id[0][1] == -1 or use_tree_variants or tree_view_id[0][1] == tree_view_id_t[0][1]) and \
               (tree_view_id[1][0] == -1 or tree_view_id[1][0] == tree_view_id_t[1][0]) and \
               (tree_view_id[1][1] == -1 or use_view_variants or tree_view_id[1][1] == tree_view_id_t[1][1])
        ]

    def _augment_tree_view_id(self, tree_view_id: tuple, use_tree_variants: bool, use_view_variants: bool,
                              tree_variant_counts: dict, view_variant_counts: dict, tree_view_ids: dict) -> tuple:
        """ Augment input tree/view identifier with random variants. """
        augmented_tree_view_id = (
            (
                tree_view_id[0][0],
                int(self._rng.random() * tree_variant_counts[tree_view_id[0][0]]) \
                    if use_tree_variants else \
                tree_view_id[0][1]
            ),
            (
                tree_view_id[1][0],
                int(self._rng.random() * view_variant_counts[( tree_view_id[0], tree_view_id[1][0] )]) \
                    if use_view_variants else \
                tree_view_id[1][1]
            )
        )
        if augmented_tree_view_id not in tree_view_ids:
            augmented_tree_view_id = ( augmented_tree_view_id[0], (-1, 0) )

        return augmented_tree_view_id

    def _prepare_complete_data(self, tree_data: dict, view_data: dict,
                               comparison_data: dict, skeleton_data: dict,
                               tree_filter: Optional[Union[List[int], Dict[int, Set[int]]]]) -> dict:
        """ Combine requested types of data into complete data. """

        comparison, comparison_score_difference = self._config.comparison_required()
        features = self._config.features_required()
        views = self._config.views_required()
        skeletons = self._config.skeletons_required()
        score_required = self._config.score_required()
        use_view_scores = self._config.use_view_scores() and len(view_data["scores"]) > 0
        comparison_view_scores = self._config.comparison_view_scores() and len(view_data["scores"]) > 0

        use_tree_variants = self._config.get_featurizer_option("use_tree_variants")
        use_view_variants = self._config.get_featurizer_option("use_view_variants")
        pre_generate_variants = self._config.get_featurizer_option("pre_generate_variants")

        if comparison and not score_required:
            raise RuntimeError("Unable to get comparison data with no scoring information!")

        # Contiguous ordering for tree identifiers.
        sorted_tree_ids = tree_data["tree_ids"] if tree_data else np.array([ [ ] ])
        sorted_tree_ids.sort()
        sorted_tree_ids_map = { tree_id: idx for idx, tree_id in enumerate(sorted_tree_ids) }
        sorted_tree_variants = {
            tree_id: variant_count
            for tree_id, variant_count in zip(
                *np.unique(
                    np.array(list(sorted_tree_ids))[:, 0] if len(sorted_tree_ids) > 0 else [ ],
                    return_counts=True
                )
            )
        }

        # Contiguous ordering for tree/view identifiers.
        sorted_tree_view_ids = view_data["tree_view_ids"] if view_data else [ ]
        sorted_tree_view_ids = np.concatenate(
            [
                sorted_tree_view_ids,
                tuple_array_to_numpy([
                    ( tree_id, ( -1, 0 ) )
                    for tree_id in sorted_tree_ids
                ], axis=-2)
            ], axis=0
        )
        sorted_tree_view_ids.sort()

        # Mapping to the input data.
        tree_data_map = {
            tree_id: idx
            for idx, tree_id in enumerate(tree_data["tree_ids"])
        }
        view_data_map = {
            tree_view_id: idx
            for idx, tree_view_id in enumerate(view_data["tree_view_ids"])
        }
        skeleton_data_map = {
            tree_id: idx
            for idx, tree_id in enumerate(skeleton_data["tree_ids"])
        }

        # Filter out any invalid tree ids and construct a contiguous mapping.
        sorted_tree_view_ids = numpy_array_to_tuple_numpy(tuple_array_to_numpy([
            tree_view_id
            for tree_view_id in sorted_tree_view_ids
            if not features or tree_view_id[0] in tree_data_map
            if not views or tree_view_id in view_data_map
            if not skeletons or tree_view_id[0] in skeleton_data_map
        ]))
        sorted_tree_view_ids_map = { tree_view_id: idx for idx, tree_view_id in enumerate(sorted_tree_view_ids) }
        sorted_tree_view_variants = {
            ( ( tree_view_id[0], tree_view_id[1] ), tree_view_id[2] ): variant_count
            for tree_view_id, variant_count in zip(
                *np.unique(
                    tuple_array_to_numpy([
                        ( tree_view_id[0][0], tree_view_id[0][1], tree_view_id[1][0] )
                        for tree_view_id in sorted_tree_view_ids
                    ]),
                    return_counts=True
                )
            )
        }

        """
        tree_view_ids = [
            (tree_id, (-1, 0))
            for tree_id in tree_data["tree_ids"]
        ] if not views else view_data["tree_view_ids"]
        tree_view_ids_map = {
            tree_view_id: idx
            for idx, tree_view_id in enumerate(tree_view_ids)
        }
        """

        # Re-order input data to contiguous ordering.
        feature_inputs = [
            tree_data["data"][tree_data_map[tree_view_id[0]]]
            for tree_view_id in sorted_tree_view_ids
        ] if features else [ list() for _ in sorted_tree_view_ids ]

        view_inputs = [
            # Use view #0, variant #0 as representative for the whole tree.
            view_data["data"][view_data_map[tree_view_id]] \
                if tree_view_id[1][0] >= 0 else \
            view_data["data"][view_data_map[( tree_view_id[0], (0, 0) )]] \
                if ( tree_view_id[0], (0, 0) ) in view_data_map else \
            view_data["data"][view_data_map[( (tree_view_id[0][0], 0), (0, 0) )]]
                if ( (tree_view_id[0][0], 0), (0, 0) ) in view_data_map else \
            view_data["data"][view_data_map[( (tree_view_id[0][0], 0), (1, 0) )]]
            for tree_view_id in sorted_tree_view_ids
        ] if views else [ list() for _ in sorted_tree_view_ids ]

        skeleton_inputs = [
            skeleton_data["data"][skeleton_data_map[tree_view_id[0]]]
            for tree_view_id in sorted_tree_view_ids
        ] if skeletons else [ list() for _ in sorted_tree_view_ids ]

        # Aggregate inputs, replacing NaN -> 0:
        total_inputs = np.hstack([ feature_inputs, view_inputs, skeleton_inputs ])
        np.nan_to_num(x=total_inputs, copy=False, nan=0.0)
        total_names = {
            # Feature names:
            "features": tree_data["names"] if features else [ ],
            # View names, partially omitted:
            "views": view_data["view_names"] if views else [ ],
            # Skeleton names:
            "skeletons": skeleton_data["names"] if skeletons else [ ]
        }

        # Re-order output data to contiguous ordering.
        score_outputs = [
            # Use aggregate score as representative for the whole tree.
            view_data["scores"][view_data_map[tree_view_id]] \
                if tree_view_id[1][0] >= 0 else \
            tree_data["scores"][tree_data_map[tree_view_id[0]]] \
                if tree_view_id[0] in tree_data_map else \
            view_data["scores"][view_data_map[( tree_view_id[0], (0, 0) )]] \
                if ( tree_view_id[0], (0, 0) ) in view_data_map else \
            view_data["scores"][view_data_map[( (tree_view_id[0][0], 0), (0, 0) )]]
            for tree_view_id in sorted_tree_view_ids
        ] if score_required else [ ]

        # Aggregate outputs:
        total_outputs = np.hstack([ score_outputs ])

        # Prepare filter:
        if tree_filter is not None:
            if isinstance(tree_filter, dict):
                tree_filter = [
                    ( tree_id, view_id )
                    for tree_id, tree_view_ids in tree_filter.items()
                    for view_id in tree_view_ids
                ]

            if isinstance(tree_filter, list):
                tree_filter = tuple_array_to_numpy(tree_filter, axis=-2)

        # Generate indexing data.
        if comparison:
            # Comparing pairs of inputs:

            comparison_ptg = self._config.get_featurizer_option("comparison_sample_ptg")
            comparison_ptg_seed = self._config.get_featurizer_option("comparison_sample_ptg_seed")
            if comparison_ptg is not None:
                comparison_samples = sku.resample(
                    comparison_data["data"], replace=False,
                    n_samples=int(comparison_ptg * len(comparison_data["data"])),
                    random_state=comparison_ptg_seed
                )
            else:
                comparison_samples = comparison_data["data"]

            if use_tree_variants or use_view_variants:
                # Augment the data with variants:

                comparison_samples = np.array([
                    [
                        self._augment_tree_view_id(
                            tree_view_id=first,
                            use_tree_variants=use_tree_variants,
                            use_view_variants=use_view_variants,
                            tree_variant_counts=sorted_tree_variants,
                            view_variant_counts=sorted_tree_view_variants,
                            tree_view_ids=sorted_tree_view_ids_map,
                        ),
                        self._augment_tree_view_id(
                            tree_view_id=second,
                            use_tree_variants=use_tree_variants,
                            use_view_variants=use_view_variants,
                            tree_variant_counts=sorted_tree_variants,
                            view_variant_counts=sorted_tree_view_variants,
                            tree_view_ids=sorted_tree_view_ids_map,
                        ),
                        choice
                    ]
                    for [ first, second, choice ] in comparison_samples
                ], dtype=object)

            # Aggregate inputs
            total_indices = np.concatenate([ comparison_samples ])

            """
            # Using pairwise comparison data:
            if comparison_score_difference:
                # Use score differences:
                total_outputs = [
                    (
                        first_tree_view_idx, second_tree_view_idx,
                        first_tree_view_score - second_tree_view_score,
                    )
                    for pairwise_comparison in comparison_samples
                    for first_tree_view_idx, second_tree_view_idx in [ (
                        tree_view_ids_map.get(pairwise_comparison[0], None) \
                            if pairwise_comparison[0] in tree_view_ids_map else \
                            tree_view_ids_map[ (pairwise_comparison[0][0][0], -1) ],
                        tree_view_ids_map.get(pairwise_comparison[1], None) \
                            if pairwise_comparison[1] in tree_view_ids_map else \
                            tree_view_ids_map[ (pairwise_comparison[1][0][0], -1) ],
                    ) ]
                    for first_tree_view_score, second_tree_view_score in [ (
                        view_data["scores"][first_tree_view_idx][0] \
                            if len(view_data["scores"]) and comparison_view_scores else \
                            tree_data["scores"][first_tree_view_idx][0],
                        view_data["scores"][second_tree_view_idx][0] \
                            if len(view_data["scores"]) and comparison_view_scores else \
                            tree_data["scores"][second_tree_view_idx][0],
                    ) ]
                ] if score_required else [ ]
            else:
                # Use choice data:
                total_outputs = [
                    (
                        first_tree_view_idx, second_tree_view_idx,
                        pairwise_comparison[2],
                    )
                    for pairwise_comparison in comparison_samples
                    for first_tree_view_idx, second_tree_view_idx in[ (
                        tree_view_ids_map[pairwise_comparison[0]] \
                            if pairwise_comparison[0] in tree_view_ids_map else \
                            tree_view_ids_map[ (pairwise_comparison[0][0][0], -1) ],
                        tree_view_ids_map[pairwise_comparison[1]] \
                            if pairwise_comparison[1] in tree_view_ids_map else \
                            tree_view_ids_map[ (pairwise_comparison[1][0][0], -1) ],
                    ) ]
                ] if score_required else [ ]

            ordered_tree_ids = [
                ( pairwise_comparison[0][0], pairwise_comparison[1][0] )
                for pairwise_comparison in comparison_samples
            ] if score_required else [ ]
            ordered_tree_view_ids = [
                ( pairwise_comparison[0], pairwise_comparison[1] )
                for pairwise_comparison in comparison_samples
            ] if score_required else [ ]
            """
        else:
            # Using single inputs:

            if pre_generate_variants:
                single_samples = tuple_array_to_numpy([
                    tree_view_id
                    for tree_view_id in sorted_tree_view_ids
                    if tree_view_id[1][0] == -1 or views or use_view_scores
                    if (tree_view_id[0][1] == 0 or use_tree_variants) and \
                       (tree_view_id[1][1] == 0 or use_view_variants)
                ], axis=-2)
            else:
                single_samples = tuple_array_to_numpy([
                    (
                        (
                            tree_view_id[0][0],
                            tree_view_id[0][1] \
                                if tree_view_id[0][1] >= 0 else
                                -1 if use_tree_variants else 0
                        ),
                        (
                            tree_view_id[1][0],
                            tree_view_id[1][1] \
                                if tree_view_id[1][1] >= 0 else
                                -1 if use_view_variants else 0
                        )
                    )
                    for tree_view_id in sorted_tree_view_ids
                    if tree_view_id[1][0] == -1 or views
                    if (tree_view_id[0][1] == 0 or use_tree_variants) and \
                       (tree_view_id[1][1] == 0 or use_view_variants)
                ], axis=-2)

            # Aggregate inputs
            total_indices = np.concatenate([ single_samples ])

            """
            if use_view_scores and len(view_data["scores"]) != 0:
                total_outputs = [
                    view_data["scores"][view_data_map[tree_view_id]]
                    for tree_view_id in tree_view_ids
                ] if score_required else [ ]
            elif len(tree_data["scores"]) != 0:
                total_outputs = [
                    tree_data["scores"][tree_data_map[tree_view_id[0]]]
                    for tree_view_id in tree_view_ids
                ] if score_required else [ ]
            elif score_required:
                self.__l.warn("No scoring data available, using 0.0!")
                total_outputs = [
                    0.0
                    for tree_view_id in tree_view_ids
                ]
            else:
                total_outputs = [ ]
            ordered_tree_ids = tree_ids
            ordered_tree_view_ids = tree_view_ids
            """

        # Filter indices, if requested:
        if tree_filter is not None:
            # TODO - np.isin is broken for tuples, filter using intersect1d.
            total_indices = np.intersect1d(total_indices, tree_filter)

        assert(len(total_inputs) == len(total_outputs) or not score_required)
        assert(len(total_inputs) > np.max([
            sorted_tree_view_ids_map[idx]
            for idx in total_indices
            if idx in sorted_tree_view_ids_map
        ]))

        return {
            "inputs": total_inputs,
            "outputs": total_outputs,
            "indices": total_indices,
            "pairwise": comparison,
            "differential": comparison_score_difference,
            "names": total_names,
            "tree_ids": sorted_tree_ids,
            "tree_ids_map": sorted_tree_ids_map,
            "tree_ids_variants": sorted_tree_variants,
            "tree_view_ids": sorted_tree_view_ids,
            "tree_view_ids_map": sorted_tree_view_ids_map,
            "tree_view_ids_variants": sorted_tree_view_variants,
        }

    def _prepare_data(self, features: dict, views: dict, scores: dict,
                      skeletons: dict,
                      split_names: Dict[str, List[str]],
                      tree_filter: Optional[Union[List[int], Dict[int, Set[int]]]]
                      ) -> (dict, dict):
        """ Prepare split data from given dictionary of split identifiers. """

        if self._verbose:
            self.__l.info("\tPreparing dataset data structure...")

        data = { }

        # TODO - Add scalers to this part.
        # TODO - Scaling could be performed on the dataset as a whole.
        for split_name, split_ids in split_names.items():
            if self._verbose:
                self.__l.info(f"\t\tPreparing split \"{split_name}\" with {split_ids} sub-splits...")

            tree_data = self._prepare_tree_data(
                features=features, views=views, scores=scores,
                split_ids=split_ids, tree_filter=tree_filter
            )
            view_data = self._prepare_view_data(
                features=features, views=views, scores=scores,
                split_ids=split_ids, tree_filter=tree_filter
            )
            comparison_data = self._prepare_comparison_data(
                features=features, views=views, scores=scores,
                split_ids=split_ids, tree_filter=tree_filter
            )
            skeleton_data = self._prepare_skeleton_data(
                features=features, views=views, scores=scores,
                skeletons=skeletons,
                split_ids=split_ids, tree_filter=tree_filter
            )
            complete_data = self._prepare_complete_data(
                tree_data=tree_data, view_data=view_data,
                comparison_data=comparison_data,
                skeleton_data=skeleton_data,
                tree_filter=tree_filter
            )
            data[split_name] = {
                "tree": tree_data,
                "view": view_data,
                "comparison": comparison_data,
                "skeleton": skeleton_data,
                "data": complete_data
            }

        configuration = {
            "features": features.get("configuration", None),
            "views": views.get("configuration", None),
            "scores": scores.get("configuration", None),
            "skeletons": skeletons.get("configuration", None),
        }

        if self._verbose:
            self.__l.info(f"\tDone, dataset data structure prepared!")

        return data, configuration

    def _calculate_data(self) -> (dict, dict):
        """ Prepare data according to the config. """

        if self._verbose:
            self.__l.info("Calculating data for dataset...")

        # 1) Prepare raw feature, image and score data:
        data_config = self._config.get_featurizer_option("data_config")
        if data_config is None:
            # No pre-configuration, load normal data:

            if self._verbose:
                self.__l.info("\tNo pre-configuration, loading data...")

            features = self._featurizer.calculate_features(
                quant_start=self._config.get_featurizer_option("quant_start"),
                quant_end=self._config.get_featurizer_option("quant_end"),
                total_buckets=self._config.get_featurizer_option("total_buckets"),
                buckets_from_split=self._config.get_featurizer_option("buckets_from_split"),
                normalize_features=self._config.get_featurizer_option("normalize_features"),
                standardize_features=self._config.get_featurizer_option("standardize_features"),
                resolution=self._config.get_featurizer_option("image_resolution"),
                interpolation=self._config.get_featurizer_option("image_interpolation")
            ) if self._config.features_required() else { }
            views = self._featurizer.calculate_views(
                resolution=self._config.get_featurizer_option("view_resolution"),
                interpolation=self._config.get_featurizer_option("view_interpolation"),
                view_types=self._config.enabled_view_names()
            ) if self._config.views_required() else { }
            scores = self._featurizer.calculate_scores(
            ) if self._config.score_required() else { }
            skeletons = self._featurizer.calculate_skeletons(
                skeleton_types=self._config.enabled_skeleton_data()
            ) if self._config.skeletons_required() else { }
        else:
            # Pre-configured data:

            if self._verbose:
                self.__l.info("\tPre-configured data, fetching config...")

            features = self._featurizer.calculate_features_for_data(
                data_loader=self._data_loader,
                configuration=data_config["features"]
            ) if self._config.features_required() else { }
            views = self._featurizer.calculate_views_for_data(
                data_loader=self._data_loader,
                configuration=data_config["views"]
            ) if self._config.views_required() else { }
            scores = self._featurizer.calculate_scores_for_data(
                data_loader=self._data_loader,
                configuration=data_config["scores"]
            ) if self._config.score_required() else { }
            skeletons = self._featurizer.calculate_skeletons_for_data(
                data_loader=self._data_loader,
                configuration=data_config["skeletons"]
            ) if self._config.skeletons_required() else { }

        # 2) Prepare split aggregates:
        split_names = self._config.get_featurizer_option("splits")
        tree_filter = self._config.get_featurizer_option("tree_filter")

        if isinstance(split_names, str):
            split_names = [ split_names ]

        if split_names is None or len(split_names) == 0:
            if features:
                split_names = list(features["data"].keys())
            elif views:
                split_names = list(views["data"].keys())

        if isinstance(split_names, list):
            split_names = { name: [ name ] for name in split_names }

        # 3) Prepare data for each requested split and format it for easy usage.
        data, configuration = self._prepare_data(
            features=features, views=views, scores=scores,
            skeletons=skeletons,
            split_names=split_names, tree_filter=tree_filter
        )

        return data, configuration

    @property
    def splits(self) -> List[str]:
        """ Get list of available splits. """
        return list(self._data.keys())

    SplitsType = Optional[Union[str, List[str]]]
    """ Type used for specifying splits. """

    def names(self, splits: SplitsType = None, name_type: str = "features") -> Optional[np.array]:
        """ Get names array for requested splits, if it was generated. Use None for current splits. """
        splits = self.get_current_splits(splits=splits)
        return np.concatenate([ self._data[split]["data"]["names"][name_type] for split in splits ])

    def tree_ids(self, splits: SplitsType = None) -> np.array:
        """ Get ordered list of all tree IDs as ordered in the inputs()/outputs(). """
        splits = self.get_current_splits(splits=splits)
        return np.concatenate([ self._data[split]["data"]["tree_ids"] for split in splits ])

    def tree_view_ids(self, splits: SplitsType = None) -> np.array:
        """ Get ordered list of all tree IDs as ordered in the inputs()/outputs(). """
        splits = self.get_current_splits(splits=splits)
        return np.concatenate([ self._data[split]["data"]["tree_view_ids"] for split in splits ])

    def inputs(self, splits: SplitsType = None) -> np.array:
        """ Get the input array for requested splits. Use None for current splits. """
        return np.array(list(self.inputs_iter(splits=splits)))

    def _resolve_index_tuple(self, index: tuple, split_data: dict) -> int:
        """ Resolve index tuple with possible variance to concrete index. """

        if index[0][1] < 0 or index[1][1] < 0:
            index = self._augment_tree_view_id(
                tree_view_id=index,
                use_tree_variants=index[0][1] < 0,
                use_view_variants=index[1][1] < 0,
                tree_variant_counts=split_data["tree_ids_variants"],
                view_variant_counts=split_data["tree_view_ids_variants"],
                tree_view_ids=split_data["tree_view_ids_map"],
            )

        return split_data["tree_view_ids_map"][index]

    def _inputs_iter_yielder(self, splits: List[str]) -> Iterable:
        """ Helper used for input array yielding. """

        for split in splits:
            split_data = self._data[split]["data"]
            pairwise = split_data["pairwise"]

            if pairwise:
                # Pairwise indexing, return both elements:
                for index in split_data["indices"]:
                    yield (
                        split_data["inputs"][self._resolve_index_tuple(
                            index=index[0], split_data=split_data)],
                        split_data["inputs"][self._resolve_index_tuple(
                            index=index[1], split_data=split_data)]
                    )
            else:
                # Single indexing, return current element:
                for index in split_data["indices"]:
                    yield split_data["inputs"][self._resolve_index_tuple(
                        index=index, split_data=split_data)]

    def inputs_iter(self, splits: SplitsType = None) -> Iterable:
        """ Get the input array iterator for requested splits. Use None for current splits. """
        splits = self.get_current_splits(splits=splits)
        return self._inputs_iter_yielder(splits=splits)

    def inputs_shape(self, splits: SplitsType = None) -> tuple:
        """ Get shape of the input array for requested splits. Use None for current splits. """
        splits = self.get_current_splits(splits=splits)

        length = np.sum([ len(self._data[split]["data"]["indices"]) for split in splits ])
        shape = tuple(self._data[splits[0]]["data"]["inputs"][0].shape)
        pairwise, _ = self._config.comparison_required()

        return ((length, 2, ) + shape) if pairwise else ((length, ) + shape)

    def view_names(self) -> List[Tuple[int]]:
        """ Get list of view names. """
        split = self._get_any_split_name()
        return self._data[split]["view"]["view_names"] if split else [ ]

    def view_shapes(self) -> List[Tuple[int]]:
        """ Get list of view shapes, ordered as the names. """
        split = self._get_any_split_name()
        return self._data[split]["view"]["view_shapes"] if split else [ ]

    def outputs(self, splits: SplitsType = None) -> np.array:
        """ Get the outputs array for requested splits. Use None for current splits. """
        return np.array(list(self.outputs_iter(splits=splits)))

    def _outputs_iter_yielder(self, splits: List[str]) -> Iterable:
        """ Helper used for input array yielding. """

        for split in splits:
            split_data = self._data[split]["data"]
            pairwise = split_data["pairwise"]
            differential = split_data["differential"]

            if pairwise:
                # Pairwise indexing, return both elements:
                if differential:
                    # Differential outputs, calculate score differences:
                    for index in split_data["indices"]:
                        yield np.array([ float(
                            split_data["outputs"][self._resolve_index_tuple(
                                index=index[0], split_data=split_data)] - \
                            split_data["outputs"][self._resolve_index_tuple(
                                index=index[1], split_data=split_data)]
                        ) ])
                else:
                    # Binary outputs, use choice from index.
                    for index in split_data["indices"]:
                        yield np.array([ int(index[2]) ])
            else:
                # Single indexing, return current element:
                for index in split_data["indices"]:
                    yield split_data["outputs"][self._resolve_index_tuple(
                        index=index, split_data=split_data)]

    def outputs_iter(self, splits: SplitsType = None) -> Iterable:
        """ Get the outputs array iterator for requested splits. Use None for current splits. """
        splits = self.get_current_splits(splits=splits)

        return self._outputs_iter_yielder(splits=splits)

    def outputs_shape(self, splits: SplitsType = None) -> tuple:
        """ Get shape of the output array for requested splits. Use None for current splits. """
        splits = self.get_current_splits(splits=splits)

        length = np.sum([ len(self._data[split]["data"]["indices"]) for split in splits ])

        return (length, 1, )

    def _resolve_index_variants(self, index: tuple, split_data: dict,
                                offset: int = 0) -> Union[int, List[int]]:
        """ Resolve index tuple with possible variance to all possible indices. """

        if np.min(index) < 0 and self._resolve_indices:
            indices = self._tree_view_id_variants(
                tree_view_id=index,
                use_tree_variants=index[0][1] < 0,
                use_view_variants=index[1][1] < 0,
                tree_view_ids=split_data["tree_view_ids_map"],
            )
            return [
                split_data["tree_view_ids_map"][idx] + offset
                for idx in indices
            ]
        else:
            if index in split_data["tree_view_ids_map"]:
                return split_data["tree_view_ids_map"][index] + offset
            else:
                return split_data["tree_view_ids_map"][( index[0], ( index[1][0], 0) )] + offset

    def set_current_splits(self, splits: SplitsType = None):
        """ Set the set of currently active splits. Use None for all splits. """

        if self._verbose:
            self.__l.info(f"Preparing dataset for \"{splits if splits else 'all'}\" splits!")

        splits = reshape_scalar(splits) if splits else list(self._data.keys())
        self._current_splits = splits

        splits = self.get_current_splits()

        current_offset = 0
        self._current_inputs = [ ]
        self._current_outputs = [ ]
        self._current_indices = [ ]

        for split in splits:
            split_data = self._data[split]["data"]
            pairwise = split_data["pairwise"]
            differential = split_data["differential"]

            self._current_inputs.append(split_data["inputs"])
            self._current_outputs.append(split_data["outputs"])

            if pairwise:
                self._current_indices.append(tuple_array_to_numpy([
                    (
                        # Offset the indices for current split.
                        first_idx,
                        second_idx,
                        int(index[2]),
                        differential
                    )
                    for index in split_data["indices"]
                    for first_idx, second_idx in [(
                        self._resolve_index_variants(
                            index=index[0], split_data=split_data,
                            offset=current_offset
                        ),
                        self._resolve_index_variants(
                            index=index[1], split_data=split_data,
                            offset=current_offset
                        )
                    )]
                ]))
            else:
                self._current_indices.append(np.array([
                    idx if isinstance(idx, int) else np.array(idx)
                    for index in split_data["indices"]
                    for idx in [
                        self._resolve_index_variants(
                            index=index, split_data=split_data,
                            offset=current_offset
                        )
                    ]
                ], dtype=object))

            current_offset += len(self._current_indices[-1])

        self._current_inputs = np.concatenate(self._current_inputs)
        self._current_outputs = np.concatenate(self._current_outputs)
        self._current_indices = np.concatenate(self._current_indices)

        if self._verbose:
            self.__l.info(f"\tSplits prepared!")

    def duplicate_for_splits(self, splits: SplitsType = None) -> "TreeDataset":
        """ Create a shallow copy of this dataset with given splits activated. """

        new_dataset = copy.copy(self)
        new_dataset.set_current_splits(splits=splits)

        return new_dataset

    def _get_any_split_name(self) -> Optional[str]:
        """ Get any valid split name for this dataset. """
        return next(iter(self._data.keys())) if self._data else None

    def get_current_splits(self, splits: SplitsType = None) -> List[str]:
        """ Get list of currently active splits. Use None for current, other values for override. """
        return reshape_scalar(splits) if splits else self._current_splits

    @property
    def data_config(self) -> dict:
        """ Get data configuration of dataset and data. """
        return {
            "dataset": self._config,
            "data": self._configuration
        }

    def __len__(self) -> int:
        """ Get the total number of training pairs in this dataset. """
        return len(self._current_indices)

    def _resolve_random_index(self, index: Union[int, List[int]]) -> int:
        """ Choose a single concrete index if input is a list. """
        if isinstance(index, list) or isinstance(index, np.ndarray):
            return index[int(self._rng.random() * len(index))]
        else:
            return index

    def _index_to_item(self, index: Union[int, List[int], tuple]) -> tuple:
        """ Use provided index to recover item from dataset. """
        if isinstance(index, tuple):
            # Pairwise indexing, return both elements:
            index1 = self._resolve_random_index(index=index[0])
            index2 = self._resolve_random_index(index=index[1])

            return (
                (
                    self._current_inputs[index1],
                    self._current_inputs[index2],
                ),
                np.array([ float(
                    self._current_outputs[index1] -
                    self._current_outputs[index2]
                ) ]) \
                    if index[3] else \
                np.array([index[2]])
            )
        else:
            # Single indexing, return current element:
            index = self._resolve_random_index(index=index)
            return (
                self._current_inputs[index],
                self._current_outputs[index]
            )

    def __getitem__(self, idx: int) -> tuple:
        """ Get a single item from the dataset. """

        index = self._current_indices[idx]
        return self._index_to_item(index=index)

    def __iter__(self) -> Iterable:
        """ Get iterator for this dataset. """

        for index in self._current_indices:
            yield self._index_to_item(index=index)

