# -*- coding: utf-8 -*-

"""
Indexer system.
"""

import itertools
import json
import os
import pathlib
import re
import secrets
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from perceptree.common.configuration import Config
from perceptree.common.configuration import Configurable
from perceptree.common.logger import LoadingBar
from perceptree.common.logger import Logger
from perceptree.common.util import parse_bool_string
from perceptree.data.feature_splitter import FeatureSplitter
from perceptree.data.treeio import TreeFile


class TreeIndexer(Logger, Configurable):
    """
    Indexer system.
    """

    COMMAND_NAME = "Index"
    """ Name of this command, used for configuration. """

    def __init__(self, config: Config):
        super().__init__(config=config)
        self._set_instance()

        self.__l.info("Initializing indexer system...")

    @classmethod
    def register_options(cls, parser: Config.Parser):
        """ Register configuration options for this class. """

        option_name = cls._add_config_parameter("input_folder")
        parser.add_argument("-i", "--input-folder",
                            action="store",
                            default=None, type=str,
                            metavar=("PATH_TO_TREES"),
                            dest=option_name,
                            help="Path to the input folder containing the trees.")

        option_name = cls._add_config_parameter("input_base_folder")
        parser.add_argument("-b", "--input-base-folder",
                            action="store",
                            default=None, type=str,
                            metavar=("PATH_TO_TREES"),
                            dest=option_name,
                            help="Path to the base directory. Indexing will be "
                                 "completed for each sub-directory, with its own "
                                 "index.")

        option_name = cls._add_config_parameter("source_tag")
        parser.add_argument("-t", "--source-tag",
                            action="store",
                            default="unknown", type=str,
                            metavar=("NAME"),
                            dest=option_name,
                            help="Name tag used to discern the source of a record.")

        option_name = cls._add_config_parameter("output_file")
        parser.add_argument("-o", "--output-file",
                            action="store",
                            default=None, type=str,
                            metavar=("PATH_TO_OUT.csv"),
                            dest=option_name,
                            help="Path to the output file.")

        option_name = cls._add_config_parameter("output_skip_existing")
        parser.add_argument("-s", "--output-skip-existing",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Skip existing output files?")

    def _index_input_folder(self, input_folder: str, source_tag: str,
                            output_file: str, skip_existing: bool) -> Optional[pd.DataFrame]:
        """
        Create index for given input folder.

        :param input_folder: Folder containing the trees.
        :param source_tag: Tag to mark the rows with.
        :param output_file: File to save the outputs to.
        :param skip_existing: Skip indexing for already existing output files?

        :return: Returns the index data-frame.
        """

        if skip_existing and pathlib.Path(output_file).exists():
            self.__l.info(f"Skipping, index file \"{output_file}\" already exists!")
            return None

        input_path = pathlib.Path(input_folder)
        file_list = list(input_path.glob("**/*.tree"))
        feature_data = [ ]

        self.__l.info(f"Processing \"{input_path}\" with {len(file_list)} input files...")
        loading_progress = LoadingBar("", max=len(file_list))
        for tree_file_path in file_list:
            try:
                tree_file = TreeFile(file_path=str(tree_file_path))
            except Exception as e:
                self.__l.warning(f"Failed to load tree file \"{tree_file_path}\" with \"{e}\"!")
                continue
            if "stats" not in tree_file.dynamic_meta_data:
                loading_progress.next(1)
                continue
            split_features = FeatureSplitter(tree_file)

            feature_values = split_features.features
            feature_names = split_features.names

            feature_dict = {
                name: val
                for val, name in zip(feature_values, feature_names)
            }
            feature_dict["source_tag"] = source_tag
            feature_dict["file_path"] = str(tree_file_path.absolute())
            feature_data.append(feature_dict)

            loading_progress.next(1)
        loading_progress.finish()

        self.__l.info(f"\tDone!")

        self.__l.info(f"Creating resulting data frame from {len(feature_data)} values...")
        feature_df = pd.DataFrame(data=feature_data)
        self.__l.info(f"\tDone!")

        if output_file:
            self._save_ouput_file(
                feature_df=feature_df,
                output_file=output_file
            )

        return feature_df

    def _index_input_base_folder(self, input_base_folder: str, source_tag: str,
                                 skip_existing: bool):
        """
        Create index for each sub-folder in the base folder.

        :param input_base_folder: Folder containing indexing base.
        :param source_tag: Tag to mark the rows with.
        :param skip_existing: Skip indexing for already existing output files?
        """

        input_base_path = pathlib.Path(input_base_folder)
        dir_list = list(input_base_path.iterdir())

        self.__l.info(f"Processing base in \"{input_base_path}\" with "
                      f"{len(dir_list)} directories...")
        for idx, input_folder in enumerate(dir_list):
            self.__l.info(f"Processing \"{input_folder}\" ({idx}/{len(dir_list)})...")
            output_file = input_folder / "index.csv"
            self._index_input_folder(
                input_folder=str(input_folder),
                source_tag=source_tag,
                output_file=str(output_file),
                skip_existing=skip_existing
            )
            self.__l.info(f"\tDone, index saved to \"{output_file}\"!")

    def _save_ouput_file(self, feature_df: Optional[pd.DataFrame], output_file: str):
        """ Save the feature data-frame to given output file. """

        if feature_df is None or len(feature_df) == 0:
            self.__l.warning(f"No data-frame to save to \"{output_file}\"!")
            return

        self.__l.info(f"Saving data-frame with {len(feature_df)} rows to \"{output_file}\"")
        feature_df.to_csv(path_or_buf=output_file, sep=";")
        self.__l.info(f"\tDone!")

    def process(self):
        """ Perform data export operations. """

        self.__l.info("Starting indexer operations...")

        if self.c.input_folder:
            self._index_input_folder(
                input_folder=self.c.input_folder,
                source_tag=self.c.source_tag,
                output_file=self.c.output_file,
                skip_existing=self.c.output_skip_existing,
            )

        if self.c.input_base_folder:
            self._index_input_base_folder(
                input_base_folder=self.c.input_base_folder,
                source_tag=self.c.source_tag,
                skip_existing=self.c.output_skip_existing,
            )

        self.__l.info("\tIndexer operations finished!")

