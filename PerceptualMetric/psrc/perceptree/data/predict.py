# -*- coding: utf-8 -*-

"""
Score prediction processing system.
"""

import filecmp
import itertools
import json
import math
import os
import pathlib
import re
import secrets
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple, Union

import gzip
import h5py
import numpy as np
import pandas as pd
import pickle as pk

from perceptree.common.cache import deep_copy_dict
from perceptree.common.cache import update_dict_recursively
from perceptree.common.configuration import Config
from perceptree.common.configuration import Configurable
from perceptree.common.logger import Logger
from perceptree.common.util import reshape_scalar
from perceptree.common.util import numpy_array_to_tuple_numpy
from perceptree.common.util import tuple_array_to_numpy
from perceptree.common.logger import LoadingBar
from perceptree.data.loader import BaseDataLoader
from perceptree.data.loader import DataLoader
from perceptree.data.loader import CustomDataLoader
from perceptree.data.treeio import TreeFile
from perceptree.data.treeio import TreeImage
from perceptree.data.featurizer import DataFeaturizer
from perceptree.model.base import BaseModel


class Prediction(Logger):
    """ Wrapper around a single requested prediction. For details, see initialize_data(). """

    def __init__(self, tree_id: int, view_ids: Optional[Union[int, List[int]]] = None,
                 tree_file: Optional[TreeFile] = None,
                 tree_views: Optional[Dict[int, Dict[str, TreeImage]]] = None,
                 tree_view_types: Optional[List[str]] = None,
                 data_loader: Optional[BaseDataLoader] = None,
                 complete_tree: bool = True,
                 load_expected_scores: bool = True,
                 use_dataset_tree: bool = True,
                 external_scores: Optional[dict] = None,
                 clean_copy: Optional["Prediction"] = None,
                 data_source: Optional[str] = "unknown"):
        self._tree_id = None
        self._view_ids = None
        self._tree_file = None
        self._tree_views = None
        self._tree_view_types = None
        self._complete_tree = None
        self._score_prediction = None
        self._score_expected = None
        self._max_score = None
        self._data_source = None

        if clean_copy is not None:
            self._initialize_copy(clean_copy)
        else:
            self.initialize_data(tree_id=tree_id, view_ids=view_ids,
                                 tree_file=tree_file, tree_views=tree_views,
                                 tree_view_types=tree_view_types,
                                 complete_tree=complete_tree,
                                 load_expected_scores=load_expected_scores,
                                 external_scores=external_scores,
                                 use_dataset_tree=use_dataset_tree,
                                 data_loader=data_loader, data_source=data_source)

    def _initialize_copy(self, clean_copy: "Prediction"):
        """ Perform a shallow copy from given prediction and clear the score prediction. """
        self._tree_id = clean_copy._tree_id
        self._view_ids = clean_copy._view_ids
        self._tree_file = clean_copy._tree_file
        self._tree_views = clean_copy._tree_views
        self._tree_view_types = clean_copy._tree_view_types
        self._complete_tree = clean_copy._complete_tree
        self._score_expected = clean_copy._score_expected
        self._score_prediction = None
        self._max_score = clean_copy._max_score
        self._data_source = clean_copy._data_source

    def initialize_data(self, tree_id: Union[int, Tuple[int, int]],
                        view_ids: Optional[Union[Union[int, Tuple[int, int]], List[Union[int, Tuple[int, int]]]]] = None,
                        tree_file: Optional[TreeFile] = None,
                        tree_views: Optional[Dict[int, Dict[str, TreeImage]]] = None,
                        tree_view_types: Optional[List[str]] = None,
                        complete_tree: bool = True,
                        load_expected_scores: bool = True,
                        use_dataset_tree: bool = True,
                        external_scores: Optional[dict] = None,
                        data_loader: Optional[BaseDataLoader] = None,
                        data_source: Optional[str] = "unknown"):
        """
        Initialize data required for this prediction.

        :param tree_id: Identifier of the tree, optionally with variant identifier.
        :param view_ids: Optional identifiers of the view, -1 for all views.
            Optionally also can contain a list of view ids. Additionally, view
            variant may be specified.
        :param tree_file: Optional pre-loaded tree file data.
        :param tree_views: Optional pre-loaded views data indexed by view id
            and view type respectively. If specified, view_id should be >= -1!
        :param tree_view_types: Optional list of view types to load.
        :param complete_tree: Does this prediction cover the tree as a whole?
        :param load_expected_scores: Load ground-truth scores for this prediction?
        :param use_dataset_tree: Allow use of dataset trees if any information is
            missing?
        :param external_scores: Optional dictionary mapping tree IDs to external
            loaded scores.
        :param data_loader: Optional data-loader used to get
            TreeFile/TreeImage instances when they are not
            specified.
        :param data_source: Name of the source where the tree data came from.
        """

        if tree_views is not None and view_ids is None:
            raise RuntimeError("Non-negative view id must be specified when using tree_view!")

        if isinstance(tree_id, int):
            tree_id = (tree_id, 0)

        if isinstance(view_ids, int):
            view_ids = (view_ids, 0)
        all_views = isinstance(view_ids, tuple) and view_ids[0] < 0

        if isinstance(view_ids, list):
            view_ids = [
                (view_id, 0) if isinstance(view_ids, int) else view_id
                for view_id in view_ids
            ]
        elif isinstance(view_ids, tuple):
            view_ids = tuple_array_to_numpy([ view_ids ])
        else:
            view_ids = reshape_scalar([ ])

        if tree_views is not None and len(tree_views) != len(view_ids) and not all_views:
            raise RuntimeError("Number of view images and view ids must be equal!")

        prepare_views = len(view_ids) and tree_views is None
        data_loader_required = (tree_file is None) or prepare_views or load_expected_scores
        if data_loader_required and data_loader is None:
            raise RuntimeError("Data loader is necessary to load some required data, but it is unavailable!")

        # Prepare views if requested and not specified.
        if prepare_views:
            view_catalogue = data_loader.full_view_catalogue

            if tree_id not in view_catalogue.index:
                raise RuntimeError(f"Tree ID \"{tree_id}\" is not present in view catalogue!")

            if min(min(view_ids)) < 0:
                # Get all possible views.
                new_view_ids = [ ]
                for view_id in view_ids:
                    if view_id[0] >= 0 and view_id[1] >= 0:
                        new_view_ids.append(view_id)
                    else:
                        view_slice = tuple((
                            view_id[0] if view_id[0] >= 0 else slice(None),
                            view_id[1] if view_id[1] >= 0 else slice(None)
                        ))
                        new_view_ids += list(np.unique(numpy_array_to_tuple_numpy([
                            (idx[2], idx[3])
                            for idx in view_catalogue.loc[tree_id + view_slice].index
                        ])))
                view_ids = numpy_array_to_tuple_numpy(new_view_ids)

            tree_views = { }
            for view_id in view_ids:
                # Recover all modalities for requested views.
                single_view_catalogue = view_catalogue.loc[(tree_id[0], tree_id[1], view_id[0], view_id[1])]
                tree_views[view_id] = {
                    view_type: view_data.data or TreeImage(
                        image_path=f"{data_loader.view_base_path}/{view_data.path}"
                    )
                    for view_type, view_data in single_view_catalogue.iterrows()
                    if tree_view_types is None or view_type in tree_view_types
                }

        # Prepare tree data if not specified.
        if tree_file is None and use_dataset_tree:
            tree_data = data_loader.tree_data
            if tree_id not in tree_data:
                raise RuntimeError(f"Tree ID \"{tree_id}\" is not present in tree data catalogue!")

            tree_file = tree_data[tree_id]

        # Prepare score data if requested.
        if load_expected_scores:
            score_expected = { }
            for tree_view_id in [ (tree_id[0], tree_id[1], -1, 0) ] \
                    if complete_tree else \
                    [ (tree_id[0], tree_id[1], view_id[0], view_id[1]) for view_id in view_ids ]:
                if external_scores is not None:
                    if tree_view_id not in external_scores:
                        raise RuntimeError(f"Tree ID/View ID \"{tree_view_id}\" is not present in external score data!")
                    score_expected[( tree_view_id[2], tree_view_id[3] )] = external_scores[tree_view_id]
                elif use_dataset_tree:
                    if tree_view_id not in data_loader.full_scores_indexed.index:
                        raise RuntimeError(f"Tree ID/View ID \"{tree_view_id}\" is not present in tree score data!")
                    score_expected[( tree_view_id[2], tree_view_id[3] )] = \
                        data_loader.full_scores_indexed.loc[tree_view_id].to_dict()
        else:
            score_expected = None

        if len(data_loader.full_scores_indexed) > 0:
            max_score = data_loader.full_scores_indexed.jod.max()
            max_score = 4.5
        else:
            max_score = 4.5

        self._tree_id = tree_id
        self._view_ids = list(view_ids)
        self._complete_tree = complete_tree
        self._tree_file = tree_file
        self._tree_views = tree_views
        self._score_expected = score_expected
        self._max_score = max_score
        self._data_source = data_source

    @property
    def tree_id(self) -> int:
        """ Get tree identifier used by this prediction. """
        return self._tree_id

    @property
    def view_ids(self) -> List[int]:
        """ Get view identifiers used by this prediction. """
        return self._view_ids

    @property
    def data_source(self) -> str:
        """ Get source of where the tree data came from. """
        return self._data_source or "unknown"

    @property
    def complete_tree(self) -> List[int]:
        """ Does this prediction cover tree as a whole (True) or individual views (False). """
        return self._complete_tree

    @property
    def tree_file(self) -> TreeFile:
        """ Get pre-loaded tree file for the target tree. """
        return self._tree_file

    @property
    def tree_views(self) -> Optional[Dict[int, Dict[str, TreeFile]]]:
        """ Get pre-loaded tree views for the target tree. """
        return self._tree_views

    @property
    def score_expected(self) -> Optional[Dict[int, Dict[str, float]]]:
        """ Get expected scores, if they were requested. """
        return self._score_expected

    @property
    def score_prediction(self) -> Optional[Dict[int, Dict[str, float]]]:
        """ Get score prediction for this tree or its views depending on complete_tree(). """
        return self._score_prediction

    @score_prediction.setter
    def score_prediction(self, score: Union[List[float], Dict[int, float]]):
        """ Set new predicted score for this tree. """
        if self._complete_tree and len(score) != 1:
            raise RuntimeError(f"Unable to set non-scalar scores for a complete tree ({len(score)})!")
        if not self._complete_tree and self._score_expected and len(score) != len(self._score_expected):
            raise RuntimeError(f"Unable to set non-list scores for a non-complete tree "
                               f"({len(self._score_expected)} vs {len(score)})!")

        view_ids = [ ( -1, 0) ] if self._complete_tree else self.view_ids
        score = {
            view_id: score[view_idx]
            for view_idx, view_id in enumerate(view_ids)
        } if not isinstance(score, dict) else score

        self._score_prediction = {
            view_id: {
                "jod": min(self._max_score, score[view_id]),
                "jod_low": min(self._max_score, score[view_id]),
                "jod_high": min(self._max_score, score[view_id]),
                "jod_var": 0.0
            }
            for view_id in view_ids
        }

    def clean_copy(self) -> "Prediction":
        """ Create a clean clone of this tree without score prediction. """
        return Prediction(tree_id=-1, clean_copy=self)


class PredictionProcessor(Logger, Configurable):
    """
    Input file loading and caching system.
    """

    COMMAND_NAME = "Predict"
    """ Name of this command, used for configuration. """

    def __init__(self, config: Config):
        super().__init__(config=config)
        self._set_instance()

        self._featurizer = self.get_instance(DataFeaturizer)
        self._data_loader = self.get_instance(DataLoader)

        self.__l.info("Initializing prediction processing system...")

    @staticmethod
    def _parse_prediction_request(prediction_spec: str, complete_tree: bool) -> dict:
        """ Parse given prediction specification and return constructor arguments dictionary. """

        if prediction_spec == "-1":
            # Perform prediction on all trees in the test set.
            return {
                "tree_id": -1,
                "tree_id_variant": 0,
                "loading_required": False,
                "file_path": None,
                "view_paths": None,
                "complete_tree": complete_tree,
                "name": prediction_spec,
            }
        elif prediction_spec.isdecimal():
            # Perform a single prediction for given tree id.
            return {
                "tree_id": ( int(prediction_spec), 0 ),
                "loading_required": False,
                "file_path": None,
                "view_paths": None,
                "complete_tree": complete_tree,
                "name": prediction_spec,
            }
        elif prediction_spec == "train" or prediction_spec == "valid" or prediction_spec == "test":
            # Perform prediction on all trees in the split.
            return {
                "tree_id": ( prediction_spec, 0 ),
                "loading_required": False,
                "file_path": None,
                "view_paths": None,
                "complete_tree": complete_tree,
                "name": prediction_spec,
            }
        else:
            # Complex external data specification.
            # tree:<PATH>,id:<ID>(;<VID>),view:<ID>(;<VID>):<TYPE>:<PATH>,...
            spec_parts = prediction_spec.split(",")
            result = {
                "tree_id": None,
                "loading_required": False,
                "file_path": None,
                "view_paths": { },
                "complete_tree": complete_tree,
                "name": prediction_spec,
            }

            for spec_part in spec_parts:
                segments = spec_part.split(":")
                if len(segments) < 1:
                    raise TypeError(f"Prediction specification \"{prediction_spec}\" contains invalid segments!")
                if segments[0] == "id":
                    if ";" in segments[1]:
                        parts = segments[1].split(";")
                        segments[1] = ( int(parts[0]), int(parts[1]) )
                    else:
                        segments[1] = int(segments[1])

                    result["tree_id"] = segments[1]
                elif segments[0] == "tree":
                    result["file_path"] = segments[1]
                elif segments[0] == "view":
                    if ";" in segments[1]:
                        parts = segments[1].split(";")
                        segments[1] = ( int(parts[0]), int(parts[1]) )
                    else:
                        segments[1] = int(segments[1])
                    if len(segments) >= 4:
                        result["view_paths"] = update_dict_recursively(
                            result["view_paths"], {
                                segments[1]: {
                                    segments[2]: segments[3]
                                }
                            },
                            create_keys=True
                        )
                        result["loading_required"] = True
                    elif len(segments) >= 3:
                        result["view_paths"] = update_dict_recursively(
                            result["view_paths"], {
                                segments[1]: {
                                    segments[2]: None
                                }
                            },
                            create_keys=True
                        )
                        result["loading_required"] = False
                    else:
                        result["view_paths"] = update_dict_recursively(
                            result["view_paths"], {
                                segments[1]: { }
                            },
                            create_keys=True
                        )
                        result["loading_required"] = False
                else:
                    raise TypeError(f"Unknown segment specifier \"{segments[0]}\" provided in prediction specification!")

            #assert(result["tree_id"] is None)

            return result

    @classmethod
    def register_options(cls, parser: Config.Parser):
        """ Register configuration options for this class. """

        parser.add_argument("--predict-tree",
                            action="append",
                            default=[], type=lambda x : PredictionProcessor._parse_prediction_request(x, True),
                            metavar=("<NUM>|<PRED_SPEC>|train|valid|test"),
                            dest=cls._add_config_parameter("predict_tree"),
                            help="Request prediction of tree scores. Use -1 to predict all "
                                 "trees in the loaded set or tree ID for specific ID. Use "
                                 "tree:<PATH>,view:<ID>:<TYPE>:<PATH>,... to load "
                                 "external data. Specify train/valid/test to predict results "
                                 "for training/validation/testing data in the currently loaded "
                                 "data.")

        parser.add_argument("--predict-tree-folder",
                            action="append",
                            default=[], type=str,
                            metavar=("INPUT_FOLDER"),
                            dest=cls._add_config_parameter("predict_tree_folder"),
                            help="Predict for all trees within given folder.")

        parser.add_argument("--predict-views",
                            action="append",
                            default=[], type=lambda x : PredictionProcessor._parse_prediction_request(x, False),
                            metavar=("<NUM>|<PRED_SPEC>|train|valid|test"),
                            dest=cls._add_config_parameter("predict_views"),
                            help="Request prediction of view scores. Use -1 to predict all "
                                 "trees in the loaded set or tree ID for specific ID. Use "
                                 "tree:<PATH>,view:<ID>:<TYPE>:<PATH>,... to load "
                                 "external data. Specify train/valid/test to predict results "
                                 "for training/validation/testing data in the currently loaded "
                                 "data.")

        parser.add_argument("--predict-views-folder",
                            action="append",
                            default=[], type=str,
                            metavar=("INPUT_FOLDER"),
                            dest=cls._add_config_parameter("predict_views_folder"),
                            help="Predict for all trees within given folder.")

        parser.add_argument("--predict-views-folder-unstructured",
                            action="append",
                            default=[], type=str,
                            metavar=("INPUT_FOLDER"),
                            dest=cls._add_config_parameter("predict_views_folder_unstructured"),
                            help="Predict for all png images in given folder.")

        parser.add_argument("--predict-external",
                            action="append",
                            default=[], type=lambda x : PredictionProcessor._parse_prediction_request(x, False),
                            metavar=("<PRED_SPEC>"),
                            dest=cls._add_config_parameter("predict_external"),
                            help="Request prediction of view scores. Use -1 to predict all "
                                 "trees in the loaded set or tree ID for specific ID. Use "
                                 "tree:<PATH>,view:<ID>:<TYPE>:<PATH>,... to load "
                                 "external data. Specify train/valid/test to predict results "
                                 "for training/validation/testing data in the currently loaded "
                                 "data.")

        parser.add_argument("--predict-external-featurizer",
                            action="store",
                            default=None, type=str,
                            metavar=("<PATH/TO/TreeIOViewer>"),
                            dest=cls._add_config_parameter("predict_external_featurizer"),
                            help="Request prediction of scores for external tree file. Use "
                                 "tree:<PATH>,view:<ID>:<TYPE>:<PATH>,... to load external "
                                 "data and automatically perform rendering and feature "
                                 "extraction as necessary. <TYPE> and <PATH> for views are "
                                 "optional, if not specified they are generated automatically.")

        parser.add_argument("--predict-external-workdir",
                            action="store",
                            default="./featurization_cache/", type=str,
                            metavar=("<PATH/TO/WORKDIR>"),
                            dest=cls._add_config_parameter("predict_external_workdir"),
                            help="Path to work directory used to automatically featurize "
                                 "external tree files using --predict-external.")

        parser.add_argument("--predict-external-scores",
                            action="store",
                            default=None, type=str,
                            metavar=("SCORES.CSV"),
                            dest=cls._add_config_parameter("predict_external_scores"),
                            help="Location of external scores repository.")

        parser.add_argument("--export-predictions",
                            action="store",
                            default=None, type=str,
                            metavar=("PATH/TO/OUTPUT.PRE"),
                            dest=cls._add_config_parameter("export_predictions"),
                            help="Export prediction requests into an external storage.")

        parser.add_argument("--import-predictions",
                            action="store",
                            default=None, type=str,
                            metavar=("PATH/TO/INPUT.PRE"),
                            dest=cls._add_config_parameter("import_predictions"),
                            help="Import prediction requests from an external storage.")

    def _featurize_external_spec(self, external_spec: dict) -> dict:
        """
        Perform featurization of given external prediction specification and return
        record with filled tree_file and tree_views keys.

        :param external_spec: Input specification, which will be left unmodified.
        :return: Returns the filled specification.
        """

        # Make local copy to keep the original intact.
        spec = deep_copy_dict(external_spec)

        # Use following paths.
        cache_path = pathlib.Path(self.c.predict_external_workdir).absolute()
        featurizer_path = pathlib.Path(self.c.predict_external_featurizer).absolute()
        featurizer_cwd = featurizer_path.parent

        # Deduce paths for the input file and its cached variant.
        input_tree_path = pathlib.Path(spec["file_path"]).absolute()
        input_tree_name = input_tree_path.with_suffix("").name
        cache_input_tree_dir = cache_path / "input" / input_tree_name
        cache_input_tree_path = cache_input_tree_dir / input_tree_path.name
        cache_output_tree_dir = cache_path / "output"
        cache_output_tree_path = cache_output_tree_dir / input_tree_name / input_tree_path.name

        # Use cached version, if it already exists.
        cache_ready = cache_input_tree_path.is_file() and cache_output_tree_path.is_file() and \
                      filecmp.cmp(input_tree_path, cache_input_tree_path)
        if not cache_ready:
            # We need to featurize the tree -> Create the working directory structure.
            cache_input_tree_dir.mkdir(parents=True, exist_ok=True)
            cache_output_tree_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(input_tree_path, cache_input_tree_path)

            # Execute the featurizer on the target directories.
            exe_parameters = "-feature {input_path}:{output_path}:./data/genviews/genviews_start.tbat"
            args = [ str(featurizer_path) ] + exe_parameters.format(
                input_path=str(cache_input_tree_dir),
                output_path=str(cache_output_tree_dir)
            ).split(" ")
            self.__l.info(f"Running featurizer on \"{input_tree_path}\"...")
            with subprocess.Popen(args, cwd=featurizer_cwd) as p:
                p.wait()
            self.__l.info("\tFeaturization complete!")

        # Recover a list of views generated by the featurizer.
        all_views = { }
        for screen_path in cache_output_tree_path.parent.glob("*screen*.png"):
            screen_name = str(screen_path.name)
            matches = re.match(r".*_screen_(\d+)_(\w+).png", screen_name)
            if not matches:
                continue
            view_id = int(matches.group(1))
            view_type = matches.group(2)
            all_views = update_dict_recursively(
                all_views, { view_id: { view_type: str(screen_path) } },
                create_keys=True
            )

        if len(spec["view_paths"]) == 0:
            spec["complete_tree"] = True
            spec["view_paths"] = { -1: { } }
        else:
            spec["complete_tree"] = False

        view_paths = { }
        for view_id, view_spec in spec["view_paths"].items():
            if view_id < 0:
                # add all views generated by the featurizer.
                view_paths = update_dict_recursively(view_paths, all_views, create_keys=True)
            else:
                # Add only the single view or use pre-defined path.
                if len(view_spec) == 0:
                    # Add single featurized view.
                    if view_id not in all_views:
                        self.__l.error(f"Requested view id {view_id} was not provided, nor created "
                                       f"by featurizer, skipping!")
                        continue
                    view_paths = update_dict_recursively(
                        view_paths, { view_id: all_views[view_id] },
                        create_keys=True
                    )
                else:
                    # Add fully specified view.
                    view_paths = update_dict_recursively(
                        view_paths, { view_id: view_spec },
                        create_keys=True
                    )

        # Update the specification with featurized data.
        spec["file_path"] = str(cache_output_tree_path)
        spec["view_paths"] = view_paths

        return spec

    def _prepare_prediction_specs(self, tree_specs: List[dict], tree_folders: List[str],
                                  view_specs: List[dict], view_folders: List[str],
                                  image_folders: List[str], external_specs: List[dict]
                                  ) -> List[dict]:
        """
        Prepare all prediction specifications for prediction.

        :param tree_specs: List of requests for tree score prediction.
        :param tree_folders: List of folder to get trees to predict from.
        :param view_specs: List of requests for view score prediction.
        :param view_folders: List of folder to get trees to predict from.
        :param image_folders: List of folder to get unstructured images from.
        :param external_specs: List of requests for external score predictions.

        :return: Returns list of predictions.
        """

        def match_tree(name: str, ext: str):
            matches = re.match(r"(\d+)_.*" + ext, name)
            if not matches:
                matches = re.match(r"tree_(\d+)" + ext, name)
            if not matches:
                matches = re.match(r"tree(\d+)" + ext, name)
            if not matches:
                matches = re.match(r"(\d+)" + ext, name)
            if not matches:
                matches = re.match(r"tree(\d+)_branch_(\d+)" + ext, name)

            return matches

        def find_views(tree_path: Union[str, pathlib.Path]) -> Optional[Dict[int, pathlib.Path]]:
            tree_path = pathlib.Path(tree_path)

            view_path = tree_path.with_suffix(".png")
            if view_path.exists():
                return { ( 0, 0 ): view_path }

            view_path = pathlib.Path(str(tree_path.with_suffix("")) + "_screen_0_base.png")
            if view_path.exists():
                idx = 0
                result = { }
                while view_path.exists():
                    result[( idx, 0 )] = view_path
                    idx += 1
                    view_path = pathlib.Path(str(tree_path.with_suffix("")) + f"_screen_{idx}_base.png")

                return result

            view_path = pathlib.Path(str(tree_path.with_suffix("")) + "_screen_0.png")
            if view_path.exists():
                idx = 0
                result = { }
                while view_path.exists():
                    result[( idx, 0 )] = view_path
                    idx += 1
                    view_path = pathlib.Path(str(tree_path.with_suffix("")) + f"_screen_{idx}.png")

                return result

            return None

        tree_folder_specs = [ ]
        for tree_folder in tree_folders:
            for tree_path in pathlib.Path(tree_folder).glob("**/*.tree"):
                matches = match_tree(name=tree_path.name, ext=".tree")
                if not matches:
                    continue

                tree_id = int(matches.group(1))
                if len(matches.groups()) >= 2:
                    variant_id = int(matches.group(2))
                else:
                    variant_id = 0

                view_paths = find_views(tree_path)
                if view_paths is None:
                    continue

                tree_folder_specs.append({
                    "tree_id": ( tree_id, variant_id ),
                    "loading_required": True,
                    "file_path": str(tree_path),
                    "view_paths": {
                        view_idx: { "base": str(view_path) }
                        for view_idx, view_path in view_paths.items()
                    },
                    "complete_tree": True,
                    "name": tree_path.name,
                })

        tree_view_specs = [ ]
        for tree_folder in view_folders:
            for tree_path in pathlib.Path(tree_folder).glob("**/*.tree"):
                matches = match_tree(name=tree_path.name, ext=".tree")
                if not matches:
                    continue

                tree_id = int(matches.group(1))
                if len(matches.groups()) >= 2:
                    variant_id = int(matches.group(2))
                else:
                    variant_id = 0

                view_paths = find_views(tree_path)
                if view_paths is None:
                    continue

                tree_view_specs.append({
                    "tree_id": ( tree_id, variant_id ),
                    "loading_required": True,
                    "file_path": str(tree_path),
                    "view_paths": {
                        view_idx: { "base": str(view_path) }
                        for view_idx, view_path in view_paths.items()
                    },
                    "complete_tree": False,
                    "name": tree_path.name,
                })

        tree_image_specs = [ ]
        for image_folder in image_folders:
            for image_path in pathlib.Path(image_folder).glob("**/*.png"):
                matches = match_tree(name=image_path.name, ext=".png")
                if matches:
                    tree_id = int(matches.group(1))
                else:
                    tree_id = None

                tree_image_specs.append({
                    "tree_id": tree_id,
                    "loading_required": True,
                    "file_path": None,
                    "view_paths": { ( 0, 0 ): { "base": str(image_path) } },
                    "complete_tree": False,
                    "name": image_path.name,
                })

        prediction_specs = [ ]
        id_counter = self._data_loader.tree_catalogue.index.max() + 1
        if math.isnan(id_counter):
            id_counter = 0
        all_tree_ids = self._data_loader.tree_ids
        default_splits = self._featurizer.generate_default_splits()

        def determine_split(search_id: Tuple[int, int], splits: dict, default_name: str) -> str:
            for split_name, split_ids in splits.items():
                if search_id in list(split_ids):
                    return split_name
            return default_name

        for spec in tree_specs + tree_folder_specs + view_specs + tree_view_specs + tree_image_specs:
            if spec["tree_id"] is None:
                spec["tree_id"] = id_counter
                id_counter += 1

            tree_id = spec["tree_id"]

            if isinstance(tree_id, int):
                # TODO - Add support for tree variants.
                tree_id = ( tree_id, 0 )

            if isinstance(tree_id[0], str):
                # Add all trees from given split:
                tree_ids = default_splits[ tree_id ]
            elif tree_id[0] < 0:
                # Convert -1 to all of the available trees:
                if tree_id[1] < 0:
                    # Keep all variants.
                    tree_ids = all_tree_ids
                else:
                    # Keep only specified variant.
                    tree_ids = [
                        id
                        for id in all_tree_ids
                        if id[1] == tree_id[1]
                    ]
            else:
                # Just use the single tree ID provided.
                tree_ids = [ tree_id ]

            for tree_id in tree_ids:
                tree_id_spec = spec.copy()
                tree_id_spec["tree_id"] = tree_id
                tree_id_spec["tree_file"] = None
                tree_id_spec["tree_views"] = None
                tree_id_spec["load_expected_scores"] = not spec["loading_required"]
                #tree_id_spec["load_expected_scores"] = True
                tree_id_spec["source"] = determine_split(
                    search_id=tree_id, splits=default_splits,
                    default_name="external"
                ) if not spec["loading_required"] else "external"
                prediction_specs.append(tree_id_spec)

        for spec in external_specs:
            if spec["tree_id"] is None:
                spec["tree_id"] = id_counter
                id_counter += 1

            tree_id = spec["tree_id"]

            if isinstance(tree_id, int):
                # TODO - Add support for tree variants.
                tree_id = ( tree_id, 0 )

            tree_id_spec = self._featurize_external_spec(external_spec=spec)
            tree_id_spec["tree_id"] = tree_id
            tree_id_spec["load_expected_scores"] = False
            tree_id_spec["source"] = "featurized"
            prediction_specs.append(tree_id_spec)

        return prediction_specs

    def _load_external_scores(self, external_scores_path: str) -> dict:
        """ Load external scores from given path. """

        score_df = pd.read_csv(external_scores_path, sep=";")
        scores = { }

        if "condition" in score_df.columns:
            # External scores from matlab optimization script.
            tree_re = re.compile(r"tree([0-9]+)")
            for idx, score_row in score_df.iterrows():
                tree_id = int(tree_re.match(score_row["condition"]).group(1))
                score_record = {
                    "jod": float(score_row["jod"]),
                    "jod_low": float(score_row["jod_low"]),
                    "jod_high": float(score_row["jod_high"]),
                    "jod_var": float(score_row["var"]),
                }

                scores[( tree_id, 0, 0, 0 )] = score_record.copy()
                scores[( tree_id, 0, -1, 0 )] = score_record.copy()
        else:
            raise RuntimeError(f"Unknown format of external scores in given file \"{external_scores_path}\"!")

        # TODO - Re-scale the external scores to the same interval the dataset uses?
        """
        min_score = np.min([ sc["jod"] for sc in scores.values() ])
        max_score = np.max([ sc["jod"] for sc in scores.values() ])
        gt_min_score = 1.0
        gt_max_score = 4.0294
        for idx in scores.keys():
            scores[idx]["jod"] = (((scores[idx]["jod"] - min_score) / (max_score - min_score)) * \
                                  (gt_max_score - gt_min_score)) + gt_min_score
        """

        return scores

    def _prepare_predictions(self, prediction_specs: List[dict],
                             external_scores_path: Optional[str]) -> List[Prediction]:
        """
        Prepare list of requested predictions from given specs.

        :param prediction_specs: List of specs for score predictions.
        :param external_scores_path: Location of external scores repository.
        :return: Returns list of requested predictions.
        """

        # Load external scores if requested.
        if external_scores_path is not None:
            external_scores = self._load_external_scores(external_scores_path=external_scores_path)
        else:
            external_scores = None

        # Resulting list of prediction specifications.
        predictions = [ ]

        # Caches for repeated loading of the same files.
        tree_file_cache = { }
        tree_view_cache = { }

        load_node_data = self.config["data.load_node_data"]

        self.__l.info(f"Preparing {len(prediction_specs)} predictions...")
        prediction_mapping = [ ]

        loading_process = LoadingBar("", max=len(prediction_specs))
        for spec in prediction_specs:
            tree_file = None
            tree_views = { }
            view_types = [ ]

            if spec["loading_required"]:
                file_path = spec["file_path"]
                if file_path is not None:
                    if file_path not in tree_file_cache:
                        tree_file_cache[ file_path ] = TreeFile(
                            file_path=file_path,
                            load_node=load_node_data
                        )
                    tree_file = tree_file_cache[ file_path ]

                view_paths = spec.get("view_paths", { })
                for view_id, view_spec in view_paths.items():
                    if isinstance(view_id, int):
                        view_id = ( view_id, 0 )

                    if view_id[0] < 0:
                        view_types += list(view_spec.keys())
                        continue
                    for view_type, view_path in view_spec.items():
                        if view_path not in tree_view_cache:
                            tree_view_cache[ view_path ] = TreeImage(
                                image_path=view_path
                            )
                        tree_view = tree_view_cache[ view_path ]
                        tree_views = update_dict_recursively(
                            tree_views, {
                                view_id: {
                                    view_type: tree_view
                                }
                            },
                            create_keys=True
                        )
            else:
                tree_file = spec["tree_file"]
                tree_views = spec["tree_views"]

                view_paths = spec.get("view_paths", { })
                view_types = [
                    view_type
                    for view_id, view_spec in view_paths.items()
                    for view_type in view_spec.keys()
                ]

            view_ids = list(tree_views.keys()) if (tree_views and not spec["complete_tree"]) else ( -1, 0 )

            prediction_mapping.append({
                "name": spec["name"],
                "tree_id": spec["tree_id"],
                "view_ids": view_ids,
            })
            predictions.append(Prediction(
                tree_id=spec["tree_id"], view_ids=view_ids,
                tree_file=tree_file, tree_views=tree_views or None,
                tree_view_types=view_types or None,
                data_loader=self._data_loader,
                complete_tree=spec["complete_tree"],
                load_expected_scores=spec["load_expected_scores"] or external_scores is not None,
                use_dataset_tree=not spec["loading_required"],
                external_scores=external_scores,
                data_source=spec["source"]
            ))
            loading_process.next(1)
        loading_process.finish()

        self.__l.info("Prediction Mapping: ")
        for pred in prediction_mapping:
            self.__l.info(f"\t\"{pred['name']}\" as \"{pred['tree_id']}, {pred['view_ids']}\"")

        return predictions

    def _import_predictions(self, input_path: str
                            ) -> (List[Prediction], CustomDataLoader):
        """ Import predictions from given input path. """

        pre_compressed = gzip.open(input_path, "r")
        pre_data = pk.load(pre_compressed)

        return pre_data["predictions"], pre_data["dataset"]

    def _export_predictions(self, predictions: List[Prediction],
                            predictions_dataset: CustomDataLoader,
                            output_path: str):
        """ Export given predictions into the output path. """

        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pre_data = {
            "predictions": predictions,
            "dataset": predictions_dataset
        }

        pre_compressed = gzip.open(output_path, "w")
        pk.dump(pre_data, pre_compressed, protocol=pk.HIGHEST_PROTOCOL)


    def process(self, models: Dict[str, BaseModel]) -> Dict[str, Tuple[BaseModel, List[Prediction]]]:
        """ Perform prediction processing operations. """

        self.__l.info("Starting prediction processing operations...")

        if self.c.predict_external and self.c.predict_external_featurizer is None:
            raise RuntimeError("Unable to predict external files, please provided featurizer path!")

        if self.c.import_predictions is not None:
            predictions, predictions_dataset = self._import_predictions(
                input_path=self.c.import_predictions
            )
        else:
            prediction_specs = self._prepare_prediction_specs(
                tree_specs=self.c.predict_tree,
                tree_folders=self.c.predict_tree_folder,
                view_specs=self.c.predict_views,
                view_folders=self.c.predict_views_folder,
                image_folders=self.c.predict_views_folder_unstructured,
                external_specs=self.c.predict_external
            )

            predictions = self._prepare_predictions(
                prediction_specs=prediction_specs,
                external_scores_path=self.c.predict_external_scores
            )
            predictions_dataset = CustomDataLoader(
                data={ "predictions": predictions }
            )
            if self.c.export_predictions is not None:
                self._export_predictions(
                    predictions=predictions,
                    predictions_dataset=predictions_dataset,
                    output_path=self.c.export_predictions
                )

        self.__l.info(f"Prepared {len(predictions)} prediction requests!")
        self.__l.info(f"Prepared prediction dataset!")

        models_predictions = { }

        for model_name, model in models.items():
            self.__l.info(f"Predicting using model {model_name}...")

            model_predictions = [ ]
            for idx, prediction in enumerate(predictions):
                self.__l.info(f"Prediction {idx + 1}/{len(predictions)}")

                model_prediction = prediction.clean_copy()
                model.predict(prediction=model_prediction, data=predictions_dataset)
                model_predictions.append(model_prediction)

                self.__l.info(f"\t {idx + 1}/{len(predictions)} Done!")

            models_predictions[model_name] = (model, model_predictions)

        self.__l.info("\tPrediction processing finished!")

        return models_predictions


