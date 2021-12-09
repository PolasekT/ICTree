# -*- coding: utf-8 -*-

"""
Output file exporting system.
"""

import json
import os
import shutil
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from perceptree.common.configuration import Config
from perceptree.common.configuration import Configurable
from perceptree.common.logger import CopyBar
from perceptree.common.logger import Logger

from perceptree.data.loader import DataLoader

from perceptree.data.treeio import TreeFile
from perceptree.data.treeio import TreeImage


class DataExporter(Logger, Configurable):
    """
    Output file exporting system.
    """

    COMMAND_NAME = "Export"
    """ Name of this command, used for configuration. """

    def __init__(self, config: Config):
        super().__init__(config=config)
        self._set_instance()

        self._data_loader = self.get_instance(DataLoader)

        self.__l.info("Initializing data exporting system...")

    @classmethod
    def register_options(cls, parser: Config.Parser):
        """ Register configuration options for this class. """

        option_name = cls._add_config_parameter("export_dataset_path")
        parser.add_argument("--export-dataset-path",
                            action="store",
                            default="", type=str,
                            dest=option_name,
                            help="Export current data-set to given path. "
                                 "Requires a loaded data-set!")

        option_name = cls._add_config_parameter("export_dataset_resolution")
        parser.add_argument("--export-dataset-resolution",
                            action="store",
                            default=None, type=int,
                            dest=option_name,
                            help="Set resolution to resize views to, when "
                                 "exporting a data-set")

        option_name = cls._add_config_parameter("tree_score_sorted_path")
        parser.add_argument("--tree-score-sorted-path",
                            action="store",
                            default="", type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Specify path to save tree score sorted views to.")

        option_name = cls._add_config_parameter("view_score_sorted_path")
        parser.add_argument("--view-score-sorted-path",
                            action="store",
                            default="", type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Specify path to save view score sorted views to.")

        option_name = cls._add_config_parameter("score_sorted_filter")
        parser.add_argument("--score-sorted-filter",
                            action="store",
                            default="base", type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Specify filtering tag for copied views. Use base "
                                 "for standard views and None for all views.")

        option_name = cls._add_config_parameter("feature_image_path")
        parser.add_argument("--feature-image-path",
                            action="store",
                            default="", type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Specify path to save the feature images to.")

    def _copy_views(self, input_path: str, output_path: str,
                    view_catalogue: pd.DataFrame,
                    view_resolution: Optional[int]):
        """ Copy views from given view catalogue from input path to the output path. """

        view_progress = CopyBar("", max=len(view_catalogue))
        for index, view in view_catalogue.iterrows():
            src_view_path = f"{input_path}/{view.path}"
            dst_view_path = f"{output_path}/{view.path}"

            view_directory = os.path.dirname(dst_view_path)
            os.makedirs(view_directory, exist_ok=True)

            if view_resolution:
                # Resizing requested -> Load and resize.
                view_image = cv2.imread(
                    filename=src_view_path
                )
                resized_view_image = cv2.resize(
                    src=view_image,
                    dsize=(view_resolution, view_resolution),
                    interpolation=cv2.INTER_LINEAR
                )
                cv2.imwrite(
                    filename=dst_view_path,
                    img=resized_view_image
                )
            else:
                # No resize -> Copy the image file.
                shutil.copyfile(
                    src=src_view_path,
                    dst=dst_view_path
                )

            if view.json_path:
                src_json_path = f"{input_path}/{view.json_path}"
                dst_json_path = f"{output_path}/{view.json_path}"
                shutil.copyfile(
                    src=src_json_path,
                    dst=dst_json_path
                )

            view_progress.next(1)

        view_progress.finish()

    def _copy_trees(self, input_path: str, output_path: str,
                    tree_catalogue: pd.DataFrame):
        """ Copy trees from given view catalogue from input path to the output path. """

        tree_progress = CopyBar("", max=len(tree_catalogue))
        for index, tree in tree_catalogue.iterrows():
            src_view_path = f"{input_path}/{tree.path}"
            dst_view_path = f"{output_path}/{tree.path}"

            view_directory = os.path.dirname(dst_view_path)
            os.makedirs(view_directory, exist_ok=True)

            shutil.copyfile(
                src=src_view_path,
                dst=dst_view_path
            )

            if tree.json_path:
                src_json_path = f"{input_path}/{tree.json_path}"
                dst_json_path = f"{output_path}/{tree.json_path}"
                shutil.copyfile(
                    src=src_json_path,
                    dst=dst_json_path
                )

            tree_progress.next(1)

        tree_progress.finish()

    def _export_dataset(self, output_path: str,
                        view_resolution: Optional[int]):
        """
        Export the currently loaded data-set using provided settings.

        :param output_path: Output path to export the data to.
        :param view_resolution: Resize all views to this resolution.
            Set to None to disable resizing.
        """

        self.__l.info(f"Exporting data-set to \"{output_path}\"...")

        results = self._data_loader.full_results
        users = self._data_loader.users
        scores = self._data_loader.full_scores
        scores_indexed = self._data_loader.scores_indexed
        spherical_scores_indexed = self._data_loader.spherical_scores_indexed
        view_catalogue = self._data_loader.full_view_catalogue
        tree_catalogue = self._data_loader.full_tree_catalogue
        input_view_path = self._data_loader.view_base_path
        dataset_meta = self._data_loader.dataset_meta

        os.makedirs(output_path, exist_ok=True)
        results.to_csv(f"{output_path}/results.csv", sep=";", index=False)
        users.to_csv(f"{output_path}/users.csv", sep=";", index=True)
        scores.to_csv(f"{output_path}/scores.csv", sep=";", index=True)
        scores_indexed.to_csv(f"{output_path}/scores_indexed.csv", sep=";", index=True)
        spherical_scores_indexed.to_csv(f"{output_path}/spherical_scores_indexed.csv", sep=";", index=True)
        view_catalogue.drop(["data"], axis=1).to_csv(f"{output_path}/view_catalogue.csv", sep=";", index=True)
        tree_catalogue.drop(["data"], axis=1).to_csv(f"{output_path}/tree_catalogue.csv", sep=";", index=True)
        with open(f"{output_path}/dataset_meta.json", "w") as f:
            json.dump(dataset_meta, f)

        self._copy_views(
            input_path=input_view_path, output_path=output_path,
            view_catalogue=view_catalogue, view_resolution=view_resolution
        )
        self._copy_trees(
            input_path=input_view_path, output_path=output_path,
            tree_catalogue=tree_catalogue
        )

        self.__l.info(f"\tExporting completed!")

    @staticmethod
    def _generate_ordinal_prefix(current_idx: int, max_idx: int) -> str:
        """ Generate string prefix to keep sorted files in order. """
        return str(current_idx).zfill(len(str(max_idx)))

    def _export_score_sorted_views(self, scores: pd.DataFrame,
                                   output_path: str, filter: str):
        """
        Export views sorted by provided scores to given directory.

        :param output_path: Directory to output the results to.
        :param filter: Tag filter for the views. Use "None" for all views.
        """

        self.__l.info(f"Exporting {len(scores)} scored views to \"{output_path}\"...")

        os.makedirs(output_path, exist_ok=True)

        view_catalogue = self._data_loader.full_view_catalogue
        view_base_path = self._data_loader.view_base_path
        self.__l.warning("TODO - Add support for view variants.")

        scored_views = view_catalogue.merge(scores, left_index=True, right_index=True)

        if filter != "None":
            scored_views = scored_views[scored_views.index.get_level_values(level="view_type") == filter]

        scored_views = scored_views.sort_values("jod").reset_index()

        view_progress = CopyBar("", max=len(scored_views))
        for index, view in scored_views.iterrows():
            src_view_path = f"{view_base_path}/{view.path}"
            view_name = os.path.basename(src_view_path)
            ordinal_prefix = self._generate_ordinal_prefix(
                current_idx=index,
                max_idx=len(scored_views)
            )
            dst_view_path = f"{output_path}/{ordinal_prefix}_{view.jod}_{view_name}"

            shutil.copyfile(
                src=src_view_path,
                dst=dst_view_path
            )
            view_progress.next(1)

        view_progress.finish()

        self.__l.info(f"\tExporting completed!")

    def _export_tree_score_sorted_views(self, output_path: str, filter: str):
        """
        Export views sorted by tree score to given directory.

        :param output_path: Directory to output the results to.
        :param filter: Tag filter for the views. Use "None" for all views.
        """

        scores = self._data_loader.full_scores_indexed
        scores = scores[scores.index.get_level_values(level="view_id") < 0]
        scores = scores.reset_index()\
            .drop([ "tree_variant_id", "view_id", "view_variant_id" ], axis=1)\
            .assign(tree_variant_id=0, view_id=0, view_variant_id=0)\
            .set_index([ "tree_id", "tree_variant_id", "view_id", "view_variant_id" ])

        self._export_score_sorted_views(
            scores=scores,
            output_path=output_path,
            filter=filter
        )

    def _export_view_score_sorted_views(self, output_path: str, filter: str):
        """
        Export views sorted by tree score to given directory.

        :param output_path: Directory to output the results to.
        :param filter: Tag filter for the views. Use "None" for all views.
        """

        self._export_score_sorted_views(
            scores=self._data_loader.full_scores_indexed,
            output_path=output_path,
            filter=filter
        )

    def _export_feature_images(self, output_path: str):
        """
        Export feature images to given path.

        :param output_path: Directory to output the results to.
        """

        tree_data = self._data_loader.tree_data

        self.__l.info(f"Exporting feature images of {len(tree_data)} trees to \"{output_path}\"...")

        os.makedirs(output_path, exist_ok=True)

        def transform_feature_image(image: np.array) -> np.array:
            target_dtype = np.uint16
            src_limits = ( np.min(image), np.max(image) )
            dst_limits = ( np.iinfo(target_dtype).min, np.iinfo(target_dtype).max )

            image_0_1 = (image.astype(np.float) - src_limits[0]) / (src_limits[1] - src_limits[0])
            dst_image = (image_0_1 * (dst_limits[1] - dst_limits[0])) + dst_limits[0]

            return np.flip(dst_image.astype(target_dtype), axis=0)

        view_progress = CopyBar("", max=len(tree_data))
        for tree_id, tree_file in tree_data.items():
            visual_meta_data = tree_file.dynamic_meta_data["stats"].get("visual", None)
            if visual_meta_data is not None:
                for image_name, image_item in visual_meta_data.items():
                    if not TreeImage.is_image_dict(image_item):
                        continue
                    tree_image = image_item["image"]
                    dst_view_path = f"{output_path}/{tree_id}_{image_name}.png"
                    tree_image.save_to(
                        path=dst_view_path,
                        transform=transform_feature_image
                    )
            view_progress.next(1)

        view_progress.finish()

        self.__l.info(f"\tExporting completed!")

    def process(self):
        """ Perform data export operations. """

        self.__l.info("Starting data export operations...")

        if self.c.export_dataset_path:
            self._export_dataset(
                output_path=self.c.export_dataset_path,
                view_resolution=self.c.export_dataset_resolution
            )

        if self.c.tree_score_sorted_path:
            self._export_tree_score_sorted_views(
                output_path=self.c.tree_score_sorted_path,
                filter=self.c.score_sorted_filter
            )

        if self.c.view_score_sorted_path:
            self._export_view_score_sorted_views(
                output_path=self.c.view_score_sorted_path,
                filter=self.c.score_sorted_filter
            )

        if self.c.feature_image_path:
            self._export_feature_images(
                output_path=self.c.feature_image_path
            )

        self.__l.info("\tExporting operations finished!")


