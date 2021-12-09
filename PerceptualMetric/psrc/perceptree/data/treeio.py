# -*- coding: utf-8 -*-

"""
Tree IO compatibility utilities and classes.
"""

import base64
import io
import itertools
import json
import math
import os
import pathlib
import re
import struct
import sys
from typing import Callable, Optional, Tuple, Union

import cv2
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from perceptree.common.logger import Logger


class TreeFile(Logger):
    """
    Wrapper around a single file in .tree format.

    :param file_path: Path to file containing the tree data.
    :param file_content: Content of the tree file.
    :param load_static: Load static meta-data?
    :param load_dynamic: Load dynamic meta-data?
    :param load_node: Load tree node data?
    """

    META_DATA_SPLITTER = "#####"
    """ String splitting meta-data from tree nodes. """

    STATIC_CSV_SEPARATOR = ","
    """ Separator used in static meta-data csv. """

    DYNAMIC_START_TAG = "<DynamicData>"
    """ Tag used to start dynamic meta-data section. """
    DYNAMIC_END_TAG = "</DynamicData>"
    """ Tag used to end dynamic meta-data section. """

    def __init__(self, file_path: Optional[str] = None,
                 file_content: Optional[str] = None,
                 load_static: bool = True, load_dynamic: bool = True,
                 load_node: bool = True, calculate_stats: bool = True):

        self._static_meta_data = {}
        self._dynamic_meta_data = {}
        self._node_data = {}

        if file_path:
            self.load_file(file_path, load_static=load_static,
                           load_dynamic=load_dynamic, load_node=load_node,
                           calculate_stats=calculate_stats)
        elif file_content:
            self.load_content(file_content, load_static=load_static,
                              load_dynamic=load_dynamic, load_node=load_node,
                              calculate_stats=calculate_stats)

    def load_file(self, file_path: str,
                  load_static: bool = True, load_dynamic: bool = True,
                  load_node: bool = True, calculate_stats: bool = True):
        """ Load tree format from file. """
        with open(file_path, "r") as f:
            self.load_content(f.read(), load_static=load_static,
                              load_dynamic=load_dynamic, load_node=load_node,
                              calculate_stats=calculate_stats)

    def load_content(self, file_content: str,
                     load_static: bool = True, load_dynamic: bool = True,
                     load_node: bool = True, calculate_stats: bool = True):
        """ Load tree format from tree file content. """

        static_meta_data, dynamic_meta_data, node_data = self._split_content_sections(
            content=file_content
        )

        if load_static:
            self._static_meta_data = self._parse_static_meta_data(static_meta_data)

        if load_dynamic:
            self._dynamic_meta_data = self._parse_dynamic_meta_data(
                dynamic_meta_data, calculate_stats=calculate_stats)

        if load_node:
            self._node_data = self._parse_node_data(node_data)

    @property
    def static_meta_data(self) -> dict:
        """ Get dictionary containing static meta-data. """
        return self._static_meta_data

    @property
    def dynamic_meta_data(self) -> dict:
        """ Get dictionary containing dynamic meta-data. """
        return self._dynamic_meta_data

    @property
    def node_data(self) -> dict:
        """ Get dictionary containing tree node data. """
        return self._node_data

    def _split_content_sections(self, content: str) -> (str, str, str):
        """
        Split file content into the three sections.

        :param content: Content to split.
        :return: Returns content sections:
            * Static meta-data
            * Dynamic meta-data
            * Tree node data
        """

        meta_node_split = content.split(self.META_DATA_SPLITTER)
        if len(meta_node_split) > 1:
            all_meta_data = "".join(meta_node_split[:-1])
            node_data = meta_node_split[-1]
        else:
            all_meta_data = ""
            node_data = content

        static_dynamic_meta_split = all_meta_data.split(self.DYNAMIC_START_TAG)

        if len(static_dynamic_meta_split) > 1:
            static_meta_data = static_dynamic_meta_split[0]
            dynamic_meta_data = "".join([
                d.replace(self.DYNAMIC_START_TAG, "").replace(self.DYNAMIC_END_TAG, "")
                for d in static_dynamic_meta_split[1:]
            ])
        else:
            dynamic_meta_data = ""
            static_meta_data = all_meta_data

        return static_meta_data, dynamic_meta_data, node_data

    def _parse_static_meta_data(self, static_content: str) -> dict:
        """ Parse given static meta-data string and return dictionary with values. """

        if len(static_content) == 0 or static_content == "null":
            return dict()

        content_io = io.StringIO(static_content)
        csv_df = pd.read_csv(content_io, sep=self.STATIC_CSV_SEPARATOR)

        if len(csv_df) >= 1:
            return csv_df.iloc[0].to_dict()
        else:
            return dict()

    def _parse_dynamic_meta_data(self, dynamic_content: str, calculate_stats: bool) -> dict:
        """ Parse given dynamic meta-data string and return dictionary with values. """

        if len(dynamic_content) == 0 or dynamic_content.strip() == "null":
            return dict()

        dynamic = json.loads(dynamic_content)

        if calculate_stats and "stats" in dynamic:
            # Search for statistics and create wrapper objects.

            def parse_dict(stat_dict):
                if "histogram" in stat_dict:
                    stat_dict["statistic"] = TreeStatistic(stat_dict)
                elif "image" in stat_dict:
                    stat_dict["image"] = TreeImage(stat_dict)

            stats = dynamic.get("stats", { })
            for stat_dict in stats.values(): parse_dict(stat_dict)

            visual = stats.get("visual", { })
            for stat_dict in visual.values(): parse_dict(stat_dict)

        return dynamic

    def _parse_node_single(self, content: str) -> (dict, str):
        """ Parse a single node and return its data and the rest of the content. """

        startPos = content.find("(")
        endPos = content.find(")")

        if startPos < 0 or endPos < 0:
            raise RuntimeError("No starting/ending bracket found for node specification!")

        node_data = content[startPos + 1:endPos]
        rest_content = content[endPos + 1:]

        value_names = [ "x", "y", "z", "width" ]
        node_values = [ float(v) for v in node_data.split(",") ]
        node_dict = { value_names[idx]: node_values[idx] for idx in range(len(node_values)) }

        return node_dict, rest_content

    def _parse_node_data(self, node_content: str) -> dict:
        """ Parse given tree node data string and return dictionary with values. """

        if len(node_content) == 0 or node_content == "null":
            return dict()

        content = node_content.strip()
        parent_stack = [ ]
        node_data = { }
        current_parent = -1
        current_node_idx = 0

        while len(content) > 0:
            if content[0] == "(":
                # Node data is starting -> Parse it.
                node_dict, content = self._parse_node_single(content)

                # Add node meta-data.
                node_dict["parent"] = current_parent
                node_dict["children"] = [ ]
                node_dict["id"] = current_node_idx
                node_data[current_node_idx] = node_dict
                # Add child to the parent node.
                if current_parent >= 0:
                    node_data[current_parent]["children"].append(current_node_idx)

                # Update current index.
                current_parent = current_node_idx
                current_node_idx += 1
            elif content[0] == "[":
                # Chain is starting -> Add parent to the stack.
                parent_stack.append(current_parent)
                content = content[1:]
            elif content[0] == "]":
                # Chain is ending -> Recover parent from the stack.
                current_parent = parent_stack.pop()
                content = content[1:]
            else:
                # Unknown character found.
                raise RuntimeError(f"Failed to parse tree node data, invalid character found \"{content[0]}\"!")

        return node_data


class TreeImage(Logger):
    """
    Simple wrapper for images passed in the tree dynamic meta-data or
    view files..

    :param image_dict: Image dictionary from the dynamic meta-data,
        containing "image" key.
    :param image_path: Path to the image to be loaded.
    """

    TYPE_NAME_FLOAT = "Float"
    """ Name of value type for float. """
    TYPE_NAME_UINT = "UInt"
    """ Name of value type for unsigned int. """

    def __init__(self,
                 image_dict: Optional[dict] = None,
                 image_path: Optional[Union[str, pathlib.Path]] = None):
        if image_dict is not None:
            self._data = self._parse_image_dict(image_dict)
        elif image_path is not None:
            self._data = self._load_image_file(image_path)
        else:
            raise RuntimeError("No input data provided to the TreeImage!")

    @staticmethod
    def is_image_dict(image_dict: dict) -> bool:
        """ Check if given dict is an image dict. """
        return "image" in image_dict

    @property
    def name(self) -> str:
        """ Get image name. """
        return self._data["name"]

    @property
    def description(self) -> str:
        """ Get image description. """
        return self._data["description"]

    @property
    def data(self) -> np.array:
        """ Get image data. """
        return self._data["values"]

    def display(self):
        """ Display the image using matplotlib.pyplot.imshow. """
        plt.imshow(self.data.squeeze(), origin="lower")
        plt.show()

    def save_to(self, path: Union[str, pathlib.Path], transform: Optional[Callable] = None):
        """ Save the image to given path, which must include the extension. """

        values = self._data["values"]

        if transform is not None:
            values = transform(values)

        cv2.imwrite(filename=str(path), img=values)

    def resize(self, resolution: Optional[int],
               interpolation: Optional[str]) -> "TreeImage":
        """
        Resize this image and return self.

        :param resolution: Requested resolution, use None for no
            resizing.
        :param interpolation: Interpolation used for resizing, use
            None for automatic.

        :return: Returns self.
        """

        interpolations = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4
        }

        current_size = self.data.shape[:2]
        if resolution is None or (current_size[0] == resolution and
                                  current_size[1] == resolution):
            return self

        decimation = (current_size[0] < resolution) and (current_size[1] < resolution)
        default_interpolation = interpolations["area"] if decimation else interpolations["linear"]
        interpolation = interpolations.get(interpolation, None) or default_interpolation

        self._data["values"] = cv2.resize(
            src=self._data["values"],
            dsize=(resolution, resolution),
            interpolation=interpolation
        )

        return self

    def _unpack_format_for(self, width: int, height: int,
                           channels: int, value_type: str) -> str:
        """ Get unpack format for given parameters. """

        type_map = {
            self.TYPE_NAME_FLOAT: "f",
            self.TYPE_NAME_UINT: "I",
        }
        if value_type not in type_map:
            raise RuntimeError(f"Unknown value type for image: \"{value_type}\".")

        unpack_type = type_map[value_type]
        total_values = width * height * channels

        return f"{total_values}{unpack_type}"

    def _parse_image_dict(self, image_dict: dict) -> dict:
        """ Parse given image dict and return parsed data. """

        if "image" not in image_dict:
            raise RuntimeError("Missing 'image' in meta-data image dict!")

        image = image_dict["image"]
        width = image["width"]
        height = image["height"]
        channels = image["channels"]
        value_type = image["valueType"]

        unpack_format = self._unpack_format_for(
            width=width, height=height,
            channels=channels, value_type=value_type
        )

        decoded = base64.b64decode(image["data"])
        values = struct.unpack(unpack_format, decoded)
        values = np.array(values).reshape((height, width, channels))

        name = image.get("name", "")
        description = image.get("description", "")

        return {
            "values": values,
            "name": name,
            "description": description
        }

    def _load_image_file(self, image_path: Union[str, pathlib.Path]) -> dict:
        """ Load given image file and return parsed data. """

        values = cv2.imread(
            filename=str(image_path)
        )

        if values is None:
            raise RuntimeError(f"Failed to load image from \"{image_path}\"")

        name = pathlib.Path(image_path).with_suffix("").name

        return {
            "values": values,
            "name": name,
            "description": image_path
        }


class TreeStatistic(Logger):
    """
    Simple wrapper for statistics passed in the tree dynamic meta-data.

    :param stat_dict: Statistic dictionary from the dynamic meta-data,
        containing "histogram", "stochastic", "variable" and other keys.
    """

    VT_SIMPLE = "simple"
    """ Identifier used for simple values. """

    VT_PAIRED = "paired"
    """ Identifier used for paired values. """

    VAL_INF = 3.4e+38
    """ Value used to signify floating point infinity. """

    def __init__(self, stat_dict: dict):
        self._data = self._parse_stat_dict(stat_dict)

    @staticmethod
    def is_stat_dict(stat_dict: dict) -> bool:
        """ Check if given dict is a statistic dict. """
        return "histogram" in stat_dict and \
               "stochastic" in stat_dict and \
               "variable" in stat_dict

    @property
    def data(self) -> dict:
        """ Get statistic data. """
        return self._data

    @property
    def values(self) -> np.array:
        """ Get values for this statistic. """
        arr_values = np.array(self.data["variable"]["values"])
        if len(arr_values.shape) <= 1:
            arr_values = arr_values.reshape((-1, 1))
        arr_values = arr_values[np.all(np.isfinite(arr_values) & \
                                       np.less(arr_values, self.VAL_INF) & \
                                       np.greater(arr_values, -self.VAL_INF), axis=-1)]

        return arr_values

    @property
    def bucket_values(self) -> np.array:
        """ Get values used for determining histogram buckets. """
        vals = self.values
        getters = {
            TreeStatistic.VT_SIMPLE: lambda: self.values,
            TreeStatistic.VT_PAIRED: lambda: self.values[:, 0]
        }
        return getters.get(self.data["variable"]["values_type"], None)() \
            if len(vals) else np.array([ ], dtype=vals.dtype)

    @property
    def count_values(self) -> np.array:
        """ Get values used for determining histogram counts. """
        vals = self.values
        getters = {
            TreeStatistic.VT_SIMPLE: lambda: np.ones(vals.shape),
            TreeStatistic.VT_PAIRED: lambda: vals[:, 1]
        }
        return getters.get(self.data["variable"]["values_type"], None)() \
            if len(vals) else np.array([ ], dtype=vals.dtype)

    def display_hist(self):
        """ Display the histogram using matplotlib.pyplot.hist. """
        hist = self._data["histogram"]
        plt.hist(hist["buckets"][:-1], hist["buckets"], weights=hist["counts"])
        plt.show()

    def display_value_hist(self):
        """ Display the value histogram using matplotlib.pyplot.hist. """
        values = self._data["variable"]["values"]
        #plt.hist(values)
        hist = self._data["histogram"]
        plt.hist(values, hist["buckets"])
        plt.show()

    def _parse_histogram_dict(self, hist_dict: dict) -> dict:
        """ Parse histogram dictionary and return the result. """
        data = hist_dict["data"] or [ ]
        min_val = hist_dict["min"]
        max_val = hist_dict["max"]
        bucket_count = hist_dict["buckets"]

        if not isinstance(data, list) or len(data) == 0:
            if bucket_count >= 1:
                step = (np.float128(max_val) - np.float128(min_val)) / (bucket_count - 1)
            else:
                step = 1
                min_val = 0
                max_val = -1
        else:
            step = data[0]["end"] - data[0]["start"]

        buckets = np.linspace(min_val, max_val, bucket_count)
        step = buckets[1] - buckets[0] if len(buckets) > 1 else step
        counts = np.zeros(max(0, len(buckets) - 1))
        for dat in data:
            idx = int(round((dat["start"] - min_val) / step))
            assert(idx >= 0 and idx + 1 < len(buckets))
            assert(abs(dat["start"] - buckets[idx]) < step and \
                   abs(dat["end"] - buckets[idx + 1]) < step)
            counts[idx] += dat["count"]

        return {
            "buckets": buckets,
            "counts": counts
        }

    def _parse_stochastic_dict(self, stoch_dict: dict) -> dict:
        """ Parse stochastic dictionary and return the result. """
        return stoch_dict.copy()

    def _parse_variable_dict(self, var_dict: dict) -> dict:
        """ Parse variable dictionary and return the result. """

        if "values" in var_dict:
            values_encoded = var_dict["values"]
            values_decoded = base64.b64decode(values_encoded)

            if len(values_decoded) == 0:
                if isinstance(var_dict["max"], list):
                    value_type = self.VT_PAIRED
                else:
                    value_type = self.VT_SIMPLE
                return {
                    "values": np.array([ ], dtype=np.float),
                    "values_type": value_type,
                    "count": var_dict["count"],
                    "min": var_dict["max"],
                    "max": var_dict["min"]
                }

            h5_data = io.BytesIO(values_decoded)
            h5_file = h5py.File(h5_data, "r")

            if "data" in h5_file:
                # Basic list of values.
                values = h5_file["data"][:]
                count_val = len(values)
                if count_val:
                    min_val = np.min(values)
                    max_val = np.max(values)
                else:
                    values = np.array([ ], dtype=values.dtype)
                    min_val = np.zeros(1, dtype=values.dtype)[0]
                    max_val = np.zeros(1, dtype=values.dtype)[0]
                values_type = self.VT_SIMPLE
            elif "data.first" in h5_file and "data.second" in h5_file:
                # Paired list with bucket value and delta.
                first_values = h5_file["data.first"][:]
                second_values = h5_file["data.second"][:]
                values = np.dstack([ first_values, second_values ])
                if len(first_values):
                    values = values.reshape(len(first_values), -1)
                    count_val = len(values)
                    min_val = values[np.argmin(second_values)]
                    max_val = values[np.argmax(second_values)]
                else:
                    values = np.array([ ], dtype=values.dtype)
                    count_val = 0
                    min_val = np.zeros(1, dtype=values.dtype)[ 0 ]
                    max_val = np.zeros(1, dtype=values.dtype)[ 0 ]
                values_type = self.VT_PAIRED

            assert (count_val == var_dict["count"])
        else:
            values = [ ]
            count_val = var_dict["count"]
            min_val = var_dict["min"]
            max_val = var_dict["max"]

        return {
            "values": values,
            "values_type": values_type,
            "count": count_val,
            "min": min_val,
            "max": max_val
        }

    def _parse_stat_dict(self, stat_dict: dict) -> dict:
        """ Parse given statistic dictionary and produced parsed data. """

        if "histogram" not in stat_dict:
            raise RuntimeError("Missing 'histogram' in meta-data statistic dict!")
        if "stochastic" not in stat_dict:
            raise RuntimeError("Missing 'stochastic' in meta-data statistic dict!")
        if "variable" not in stat_dict:
            raise RuntimeError("Missing 'variable' in meta-data statistic dict!")

        name = stat_dict.get("name", "")
        description = stat_dict.get("description", "")
        histogram = self._parse_histogram_dict(stat_dict["histogram"])
        stochastic = self._parse_stochastic_dict(stat_dict["stochastic"])
        variable = self._parse_variable_dict(stat_dict["variable"])

        return {
            "name": name,
            "description": description,
            "histogram": histogram,
            "stochastic": stochastic,
            "variable": variable
        }


