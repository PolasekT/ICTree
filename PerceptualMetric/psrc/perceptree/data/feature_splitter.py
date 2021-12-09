# -*- coding: utf-8 -*-

"""
Feature splitting and flattening utilities.
"""

import itertools
import os
import re
import sys
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import sklearn.preprocessing as skp
import sklearn.utils as sku

from perceptree.data.treeio import TreeFile
from perceptree.data.treeio import TreeImage
from perceptree.data.treeio import TreeStatistic


def deep_copy_dict(first: dict) -> dict:
    """ Perform a deep copy of all dictonaries on input and return result. """

    copy = first.copy()

    for key, value in copy.items():
        if isinstance(value, dict):
            copy[key] = deep_copy_dict(copy[key])

    return copy


DictT = TypeVar("DictT")
def update_dict_recursively(first: DictT, second: DictT,
                            create_keys: bool = False,
                            convert: bool = False) -> DictT:
    """
    Update the first dictionary with values from the second
    dictionary. Only keys common to both dictionaries will
    be used! The type of value from the second dictionary
    must also be the same or it will be skipped.
    Warning: The original dictionary (first) will be updated
    in place!

    :param first: Dictionary to update.
    :param second: Update with values from this dictionary
    :param create_keys: Create non-existent keys in first?
    :param convert: Convert dictionary instances in second
        into type used in first?

    :return: Returns updated first dictionary.
    """

    for key, value in second.items():
        if (not create_keys and key not in first) or \
                (key in first and not isinstance(first[key], type(value))):
            continue
        if isinstance(value, type(first)):
            first[key] = update_dict_recursively(
                first.get(key, type(first)()), value,
                create_keys=create_keys,
                convert=convert
            )
        elif convert and isinstance(value, dict):
            first[key] = update_dict_recursively(
                first.get(key, type(first)()), type(first)(value),
                create_keys=create_keys,
                convert=convert
            )
        else:
            first[key] = value

    return first


def reshape_scalar(val: any) -> np.array:
    """ Create array from given value, keeping dimensions for array types and creating [ val ] for scalars. """
    arr = np.array(val)
    return arr if arr.shape else arr.reshape((-1, ))


def dict_of_lists(values: Union[List[Tuple[any, any]], List[List[any]]]) -> Dict[any, List[any]]:
    """ Convert list of pairs into dictionary of lists. """

    result = { }

    for val in values:
        result[val[0]] = result.get(val[0], [ ])
        if len(val) > 2:
            result[val[0]].append(val[1:])
        else:
            result[val[0]].append(val[1])

    return result


def tuple_array_to_numpy(values: list) -> np.array:
    """ Convert input list of tuples into numpy array of tuples. """

    if len(values) == 0:
        return np.array([ ], dtype=object)

    result = np.empty(np.shape(values)[:-1], dtype=object)
    result[:] = values

    return result


def numpy_zero(dtype: np.dtype) -> any:
    """ Return zero of given dtype. """
    return np.zeros(1, dtype=dtype)[0]


def numpy_op_or_zero(arr: np.array, op: Callable) -> any:
    """ Perform operation on given list of not empty, else return zero. """
    return op(arr) if len(arr) else numpy_zero(dtype=arr.dtype)


class FeatureSplitter(object):
    """ Feature splitting and flattening tool. """

    def __init__(self, tree: TreeFile, *args, **kwargs):
        self._tree = TreeFile
        self._features = self.calculate_features(
            tree_data={ 0: tree },
            *args, **kwargs
        )

    @property
    def features(self) -> np.array:
        """ Get list of all features using the same order as names. """
        return self._features[0][0]

    @property
    def names(self) -> np.array:
        """ Get a list of names using the same order as all of the features. """
        return self._features[1]

    def _generate_default_splits(self, tree_data: Dict[int, TreeFile]) -> dict:
        """ Generate default splits for the currently loaded data. """

        return { "data": list(tree_data.keys()) }

    def _generate_split_feature_from_dicts(self, split_scheme: dict,
                                           full_stat_dict: dict,
                                           full_image_dict: dict,
                                           full_other_dict: dict
                                           ) -> dict:
        """ Generate concrete splits from provided data. """

        # Generate the splits.
        return {
            split_name: {
                "stats": {
                    stat_name: {
                        tree_id: stats[tree_id]
                        for tree_id in split_trees
                    }
                    for stat_name, stats in full_stat_dict.items()
                },
                "images": {
                    image_name: {
                        tree_id: images[tree_id]
                        for tree_id in split_trees
                    }
                    for image_name, images in full_image_dict.items()
                },
                "others": {
                    other_name: {
                        tree_id: others[tree_id]
                        for tree_id in split_trees
                    }
                    for other_name, others in full_other_dict.items()
                }
            }
            for split_name, split_trees in split_scheme.items()
        }

    def _generate_split_feature_dicts(self, tree_data: Dict[int, TreeFile],
                                      split_scheme: dict) -> dict:
        """
        Generate and split statistic and image dictionaries according to
        a given scheme.

        :param split_scheme: Splitting scheme.

        :return: Returns dictionary containing dictionaries for each split
        by name.
        """

        # Generate dicts for all data.
        full_stat_dict, full_image_dict, full_other_dict = self._generate_feature_dicts(
            tree_data=tree_data
        )

        return self._generate_split_feature_from_dicts(
            split_scheme=split_scheme,
            full_stat_dict=full_stat_dict,
            full_image_dict=full_image_dict,
            full_other_dict=full_other_dict
        )

    def _generate_feature_dicts(self, tree_data: Dict[int, TreeFile]
                                ) -> Tuple[dict, dict, dict]:
        """ Generate dictionary for statistics, images and others. """

        stat_dict = { }
        image_dict = { }
        other_dict = { }

        for tree_id, tree_file in tree_data.items():
            dynamic = tree_file.dynamic_meta_data
            if "stats" not in dynamic:
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

        return stat_dict, image_dict, other_dict

    def _calculate_meta_features(self,
                                 stat_dict: dict) -> dict:
        """
        Calculate meta features for given tree dict.

        :param stat_dict: Dictionary of statistics for trees.

        :return: Returns dictionary containing meta-data features for
        requested trees.
        """

        tree_ids = np.sort(np.unique(np.concatenate([
            list(stat_data.keys()) for stat_data in stat_dict.values()
        ])))
        stat_names = list(stat_dict.keys())

        return {
            "tree_ids": tree_ids,
            "stat_names": stat_names,
        }

    def _calculate_feature_configuration(self,
                                         splits: dict,
                                         prefer_split: Optional[str] = None,
                                         quant_start: float = 0.001,
                                         quant_end: float = 0.999,
                                         total_buckets: int = 32,
                                         resolution: Optional[int] = None,
                                         interpolation: Optional[str] = None) -> dict:
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

        :return: Returns configuration dictionary.
        """

        if prefer_split is not None and prefer_split in splits:
            # Prefer requested data.
            stat_dict = splits[prefer_split]["stats"]
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
            stat_dict = splits[largest_split_name]
        else:
            # Use all split data.
            stat_dict = None
            for split_name, split_data in splits.items():
                if stat_dict is None:
                    stat_dict = deep_copy_dict(split_data["stats"])
                else:
                    stat_dict = update_dict_recursively(
                        stat_dict, split_data["stats"], create_keys=True)

        # Concatenate values for each feature.
        stat_values = {
            name: np.concatenate([ stats.bucket_values for stats in tree_stats.values() ])
            for name, tree_stats in stat_dict.items()
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
        stat_histograms_maxes = {
            name: np.max([
                np.max(hist["counts"])
                for tree_id, hist in hists.items()
            ])
            for split_name, stat_histograms in stat_histograms_splits.items()
            for name, hists in stat_histograms.items()
        }

        return {
            "buckets": stat_buckets,
            "vis_buckets": stat_vis_buckets,
            "histogram_maxes": stat_histograms_maxes,
            "resolution": resolution,
            "interpolation": interpolation
        }

    @staticmethod
    def _calculate_histograms(stat_dict: dict,
                              buckets: dict) -> dict:
        """ Calculate bucketed histograms for given statistics. """
        return {
            name: {
                tree_id: {
                    "counts": np.histogram(a=stats.bucket_values, weights=stats.count_values,
                                           bins=buckets[name], density=True)[0]
                }
                for tree_id, stats in tree_stats.items()
            }
            for name, tree_stats in stat_dict.items()
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
                                       stat_dict: dict) -> dict:
        """
        Calculate statistics features for all trees in the given tree dict.

        :param stat_dict: Dictionary of statistics for trees.

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
        stat_all_values = {
            stat_name: stat_values if len(stat_values) else \
                np.array([ numpy_zero(dtype=stat_values.dtype) ], dtype=stat_values.dtype)
            for stat_name, tree_stats in stat_dict.items()
            for stat_values in [ np.concatenate([
                stats.values for stats in tree_stats.values()
            ]) ]
        }

        def calculate_stats(tree_values: List[float], stat_values: List[float]) -> dict:
            return {
                "min": reshape_scalar(numpy_op_or_zero(tree_values, np.min)),
                "max": reshape_scalar(numpy_op_or_zero(tree_values, np.max)),
                "mean": reshape_scalar(numpy_op_or_zero(tree_values, np.mean)),
                "var": reshape_scalar(numpy_op_or_zero(tree_values, np.var))
            }

        return {
            "stats_elements": self.stat_elements(),
            "stats_names": list(stat_dict.keys()),
            "stats": {
                stat_name: {
                    tree_id: calculate_stats(tree_values, stat_values)
                    for tree_id in tree_record.keys()
                    for tree_values, stat_values in [(stat_tree_values[stat_name][tree_id],
                                                      stat_all_values[stat_name][tree_id])]
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
                                   others_dict: dict) -> dict:
        """
        Calculate other features for all trees in the given tree dict.

        :param others_dict: Dictionary of other features for trees.

        :return: Returns dictionary containing other features for
        requested trees.
        """

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
        tree_ids = np.sort(np.unique([
            tree_id
            for stats in others_dict.values()
            for tree_id in stats.keys()
        ]))

        return {
            "others_names": others_names,
            "others": {
                f"{other_name}_{stat_name}": {
                    tree_id: reshape_scalar(others_dict[other_name][tree_id][stat_name])
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
                    stat_dict=split_data["stats"]
                ),
                "image": self._calculate_image_features(
                    image_dict=split_data["images"],
                    configuration=configuration
                ),
                "others": self._calculate_others_features(
                    others_dict=split_data["others"]
                )
            }
            for split_name, split_data in splits.items()
        }

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
                                      interpolation: Optional[str] = None) -> dict:
        """ Prepare feature configuration, see calculate_features for argument description. """

        split_scheme = self._generate_splits(
            split_scheme=self.c.data_split,
            cross_validation_split=self.c.cross_validation_split
        )
        splits = self._generate_split_feature_dicts(split_scheme=split_scheme)

        configuration = self._calculate_feature_configuration(
            splits=splits,
            prefer_split="data",
            quant_start=quant_start, quant_end=quant_end,
            total_buckets=total_buckets,
            resolution=resolution, interpolation=interpolation
        )

        return configuration

    def _prepare_feature_name_pairs(self, features: dict, split_tree_ids: dict) -> Tuple[np.array, np.array]:
        all_data = [ ]
        all_names = [ ]

        split_ids = list(split_tree_ids.keys())
        generate_names = True
        hist_names = np.unique(np.concatenate([
            list(features["data"][split]["hist"]["hist"].keys()) for split in split_tree_ids.keys()
        ]))

        element_names = np.unique(np.concatenate([
            list(tree_data.keys())
            for split in split_tree_ids.keys()
            for stat_name, stat_data in features["data"][split]["stats"]["stats"].items()
            for tree_id, tree_data in stat_data.items()
        ]))
        stat_base_names = np.unique([
            stat_name
            for split in split_tree_ids.keys()
            for stat_name in features["data"][split]["stats"]["stats"].keys()
        ])
        stat_names = {
            stat_name: element_names
            for stat_name in stat_base_names
        }

        other_names = dict_of_lists(np.unique(tuple_array_to_numpy([
            (other_base_name, other_element)
            for split in split_tree_ids.keys()
            for other_name in features["data"][split]["others"]["others"].keys()
            for other_base_name, other_element in [ other_name.split("_") ]
        ])))

        if len(hist_names):
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

        if len(stat_names):
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
            stat_data = stat_data.reshape((stat_data.shape[0], -1))
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

        if len(other_names):
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
            other_data = other_data.reshape((other_data.shape[0], -1))
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

        all_data = np.concatenate(all_data, axis=1) if all_data else np.array([ ])
        all_names = np.concatenate(all_names, axis=1)[0] if all_names and generate_names else np.array([ ])

        return all_data, all_names

    def calculate_features(self,
                           tree_data: Dict[int, TreeFile],
                           quant_start: float = 0.001,
                           quant_end: float = 0.999,
                           total_buckets: int = 32,
                           buckets_from_split: bool = True,
                           resolution: Optional[int] = None,
                           interpolation: Optional[str] = None) -> dict:
        """
        Calculate features and return dictionary containing data (training,
        validation and test) and configuration.

        :param tree_data: Input tree files.
        :param quant_start: Quantile used for the first histogram bin.
        :param quant_end: Quantile used for the last histogram bin.
        :param total_buckets: Total number of buckets in the histograms.
        :param buckets_from_split: Use buckets generated from the training
            data only. Set to False to calculate buckets from all available
            data.
        :param resolution: Target resolution to convert the views to.
            Set to None for no resizing.
        :param interpolation: Interpolation used for resize operations.

        :return: Returns dictionary containing feature data.
        """

        split_scheme = self._generate_default_splits(tree_data=tree_data)
        splits = self._generate_split_feature_dicts(tree_data=tree_data, split_scheme=split_scheme)

        configuration = self._calculate_feature_configuration(
            splits=splits,
            prefer_split="data",
            quant_start=quant_start, quant_end=quant_end,
            total_buckets=total_buckets,
            resolution=resolution, interpolation=interpolation
        )

        features = self._calculate_features(
            splits=splits, configuration=configuration
        )

        return self._prepare_feature_name_pairs(
            features=features, split_tree_ids=split_scheme
        )

