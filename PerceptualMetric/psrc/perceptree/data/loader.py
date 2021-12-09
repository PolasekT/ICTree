# -*- coding: utf-8 -*-

"""
Input file loading and caching system.
"""

import itertools
import json
import os
import pathlib
import re
import secrets
import sys
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as scpi
import scipy.spatial.transform as scpt
import seaborn as sns
import swifter

from perceptree.common.cache import update_dict_recursively
from perceptree.common.configuration import Config
from perceptree.common.configuration import Configurable
from perceptree.common.graph_saver import GraphSaver
from perceptree.common.logger import Logger
from perceptree.common.logger import ParsingBar
from perceptree.common.logger import LoadingBar
from perceptree.common.math import carthesian_to_spherical
from perceptree.common.math import spherical_to_carthesian
from perceptree.common.util import dict_of_lists
from perceptree.common.util import parse_bool_string
from perceptree.common.util import tuple_array_to_numpy
from perceptree.data.treeio import TreeFile
from perceptree.data.treeio import TreeImage
from perceptree.data.treeio import TreeStatistic


class BaseDataLoader(Logger):
    """
    Input file loading and caching system.
    """

    @staticmethod
    def _create_empty_scores() -> pd.DataFrame:
        """ Create empty results dataframe as per _compile_scores return. """

        return pd.DataFrame({
            "tree_id": int(),
            "tree_variant_id": int(),
            "tree_jod": int(),
            "tree_jod_low": float(),
            "tree_jod_high": float(),
            "tree_jod_var": float()
        }, index=[ ]).set_index([ "tree_id", "tree_variant_id" ])

    @staticmethod
    def _generate_reduced_scores(full_scores: pd.DataFrame) -> pd.DataFrame:
        """ Generate reduced scores from given full scores. """
        return full_scores.reset_index() \
            .drop([ "tree_variant_id" ], axis=1) \
            .set_index([ "tree_id" ])

    @staticmethod
    def _create_empty_results() -> pd.DataFrame:
        """ Create empty results dataframe as per _extract_results return. """

        return pd.DataFrame({
            "index": int(),
            "first_tree_id": int(),
            "first_tree_variant_id": int(),
            "first_view_id": int(),
            "first_view_variant_id": int(),
            "second_tree_id": int(),
            "second_tree_variant_id": int(),
            "second_view_id": int(),
            "second_view_variant_id": int(),
            "worker_id": str(),
            "choice": int()
        }, index=[ ]).set_index("index")

    @staticmethod
    def _generate_reduced_results(full_results: pd.DataFrame) -> pd.DataFrame:
        """ Generate reduced results from given full results. """
        return full_results.reset_index() \
            .drop([ "first_tree_variant_id", "first_view_variant_id", "second_tree_variant_id", "second_view_variant_id" ], axis=1) \
            .set_index([ "index" ])

    @staticmethod
    def _create_empty_view_catalogue() -> pd.DataFrame:
        """ Create empty results dataframe as per _view_catalogue return. """

        return pd.DataFrame({
            "tree_id": int(),
            "tree_variant_id": int(),
            "view_id": int(),
            "view_variant_id": int(),
            "view_type": str(),
            "path": str(),
            "json_path": str(),
            "data": object()
        }, index=[ ]).set_index(["tree_id", "view_id", "view_variant_id", "view_type"])

    @staticmethod
    def _generate_reduced_view_catalogue(full_view_catalogue: pd.DataFrame) -> pd.DataFrame:
        """ Generate reduced view catalogue from given full view catalogue. """
        return full_view_catalogue.reset_index() \
            .drop([ "tree_variant_id", "view_variant_id", "json_path" ], axis=1) \
            .set_index([ "tree_id", "view_id", "view_type" ])

    @staticmethod
    def _create_empty_tree_catalogue() -> pd.DataFrame:
        """ Create empty results dataframe as per _tree_catalogue return. """

        return pd.DataFrame({
            "tree_id": int(),
            "tree_variant_id": int(),
            "path": str(),
            "json_path": str(),
            "data": object()
        }, index=[ ]).set_index(["tree_id", "tree_variant_id"])

    @staticmethod
    def _generate_reduced_tree_catalogue(full_tree_catalogue: pd.DataFrame) -> pd.DataFrame:
        """ Generate reduced tree catalogue from given full tree catalogue. """
        return full_tree_catalogue.reset_index() \
            .drop([ "tree_variant_id", "json_path" ], axis=1) \
            .set_index([ "tree_id" ])

    @staticmethod
    def _create_empty_tree_data() -> dict:
        """ Create empty results dict as per _tree_data return. """

        return { }

    @staticmethod
    def _create_empty_available_features() -> dict:
        """ Create empty results dict as per _available_features return. """

        return { }

    @staticmethod
    def _create_empty_dataset_path() -> str:
        """ Create empty results str as per _dataset_path return. """

        return ""

    @staticmethod
    def _create_empty_dataset_meta() -> dict:
        """ Create empty results dict as per _dataset_meta return. """

        return { "unique_id": "EMPTY" }

    @staticmethod
    def _create_empty_indexed_scores() -> pd.DataFrame:
        """ Create empty results dataframe as per _index_scores return. """

        return pd.DataFrame({
            "tree_id": int(),
            "tree_variant_id": int(),
            "view_id": int(),
            "view_variant_id": int(),
            "jod": float(),
            "jod_low": float(),
            "jod_high": float(),
            "jod_var": float()
        }, index=[ ]).set_index(["tree_id", "tree_variant_id", "view_id", "view_variant_id"])

    @staticmethod
    def _generate_reduced_scores_indexed(full_scores_indexed: pd.DataFrame) -> pd.DataFrame:
        """ Generate reduced indexed scores from given full indexed scores. """
        return full_scores_indexed.reset_index() \
            .drop([ "tree_variant_id", "view_variant_id" ], axis=1) \
            .set_index([ "tree_id", "view_id" ])

    @staticmethod
    def _create_empty_spherical_indexed_scores() -> pd.DataFrame:
        """ Create empty results dataframe as per _spherical_scores_indexed return. """

        return pd.DataFrame({
            "tree_id": int(),
            "tree_variant_id": int(),
            "view_id": int(),
            "view_variant_id": int(),
            "jod": float(),
            "jod_low": float(),
            "jod_high": float(),
            "jod_var": float()
        }, index=[ ]).set_index(["tree_id", "tree_variant_id", "view_id", "view_variant_id"])

    def _index_scores(self, scores: pd.DataFrame) -> pd.DataFrame:
        """
        Create indexed score data-frame, where -1st view is
        for the complete tree.

        :param scores: Input scores data-frame.

        :return: Returns data-frame indexed by ("tree_id", "view_id"),
            where view_id == -1 contains data for the whole tree. Result
            contains following columns:
            * tree_id, view_id - Integer index for unique tree/view.
            * jod, jod_low, jod_high, jod_var - JOD properties.
        """

        self.__l.info(f"Indexing {len(scores)} scores...")

        if len(scores) <= 0:
            self.__l.info(f"Input scores are empty, returning empty frame!")
            return BaseDataLoader._create_empty_indexed_scores()

        def convert_row(row):
            view_count = (len(row) // 4) - 1
            return pd.DataFrame(([ {
                "tree_id": row["tree_id"],
                "tree_variant_id": row["tree_variant_id"],
                "view_id": -1,
                "view_variant_id": 0,
                # TODO - Add support for tree and view variants.
                "jod": row["tree_jod"],
                "jod_low": row["tree_jod_low"],
                "jod_high": row["tree_jod_high"],
                "jod_var": row["tree_jod_var"],
            } ] if "tree_jod" in row else [ ])
            +
            ([ {
                "tree_id": row["tree_id"],
                "tree_variant_id": row["tree_variant_id"],
                "view_id": view_idx,
                "view_variant_id": 0,
                # TODO - Add support for tree and view variants.
                "jod": row[f"view{view_idx}_jod"],
                "jod_low": row[f"view{view_idx}_jod_low"],
                "jod_high": row[f"view{view_idx}_jod_high"],
                "jod_var": row[f"view{view_idx}_jod_var"],
            } for view_idx in range(view_count) ]))

        scores_indexed = pd.concat([ convert_row(row) for index, row in scores.reset_index().iterrows() ])
        scores_indexed["tree_id"] = scores_indexed["tree_id"].astype("int64")
        scores_indexed["tree_variant_id"] = scores_indexed["tree_variant_id"].astype("int64")
        scores_indexed.set_index(["tree_id", "tree_variant_id", "view_id", "view_variant_id"], inplace=True)

        self.__l.info(f"\tIndexing complete, resulting in {len(scores_indexed)} records.")

        return scores_indexed

    def _check_view_tree_catalogue(self, base_path: str,
                                   view_catalogue: pd.DataFrame,
                                   tree_catalogue: pd.DataFrame,
                                   scores_indexed: pd.DataFrame) -> bool:
        """
        Check whether all necessary view are accounted for and
        present in the data-set.

        :param base_path: Base path where the data-set exists.
        :param view_catalogue: Catalogue containing all information
            about the views.
        :param tree_catalogue: Catalogue containing all information
            about trees.
        :param scores_indexed: Indexed scores for trees and views.

        :return: Returns True if all necessary views are present.
        """

        self.__l.info(f"Checking view catalogue with {len(view_catalogue)} views...")

        tree_count = len(view_catalogue.index.unique(level=0))
        tree_variant_count = len(view_catalogue.index.unique(level=1))
        view_count = len(view_catalogue.index.unique(level=2))
        view_variant_count = len(view_catalogue.index.unique(level=3))
        view_type_count = len(view_catalogue.index.unique(level=4))

        expected_view_count = tree_count * tree_variant_count * view_count * view_variant_count * view_type_count

        if len(view_catalogue) != expected_view_count:
            self.__l.warning(f"\tView catalogue does not contain all expected "
                             f"views ({len(view_catalogue)} / {expected_view_count})!")
            #return False

        # Check views:
        if len(view_catalogue) < 1000:
            for index, view in view_catalogue.iterrows():
                if not os.path.isfile(f"{base_path}/{view.path}"):
                    self.__l.warning(f"\tView catalogue contains non-existent view "
                                     f"\"{view.path}\"!")
                    return False
                if view.json_path and not os.path.isfile(f"{base_path}/{view.json_path}"):
                    self.__l.warning(f"\tView catalogue contains non-existent json description "
                                     f"\"{view.json_path}\"!")
                    return False
        else:
            self.__l.warning(f"\tSkipping view catalog checking since it has {len(view_catalogue)} items!")

        self.__l.info(f"\tView catalogue successfully checked!")

        self.__l.info(f"Checking tree catalogue with {len(tree_catalogue)} trees...")

        # Check .tree files:
        for index, tree in tree_catalogue.iterrows():
            if not os.path.isfile(f"{base_path}/{tree.path}"):
                self.__l.warning(f"\tView catalogue contains non-existent tree "
                                 f"\"{tree.path}\"!")
                return False
            if tree.json_path and not os.path.isfile(f"{base_path}/{tree.json_path}"):
                self.__l.warning(f"\tView catalogue contains non-existent json description "
                                 f"\"{tree.json_path}\"!")
                return False

        self.__l.info(f"\tTree catalogue successfully checked!")

        return True

    def _prepare_spherical_knots(self, variant_jsons: dict,
                                 tree_scores: pd.DataFrame) -> (dict, np.array):
        """ Prepare tree view knot points for spherical interpolation. """

        base_view_json = variant_jsons[(0, 0)]
        base_height = base_view_json["state"]["camera"]["height"]["base"]
        base_distance = base_view_json["state"]["camera"]["distance"]["base"]

        origin_pos = np.array([ 0.0, 0.0, 0.0 ])
        bottom_pos = np.array([ 0.0, -base_distance, 0.0 ])
        top_pos = np.array([ 0.0, base_distance, 0.0 ])
        base_pos = np.array([ base_distance, base_height, 0.0 ])

        scores = tree_scores.set_index("view_id")

        knot_dict = {
            view_id: {
                "score": scores.loc[view_id].jod,
                "pos": scpt.Rotation.from_euler(
                    "XYZ", variant_json["tree"]["rotation"],
                    degrees=False
                ).apply(variant_json["camera"]["pos"])
            }
            for (view_id, variant_id), variant_json in variant_jsons.items()
            if variant_id == 0
        }
        knot_dict[-3] = {
            "score": scores.loc[-1].jod,
            "pos": origin_pos
        }
        knot_dict[-2] = {
            "score": scores.loc[-1].jod,
            "pos": bottom_pos
        }
        knot_dict[-1] = {
            "score": scores.loc[-1].jod,
            "pos": top_pos
        }

        knots = np.array([
            [spherical[1], spherical[2], score["score"]]
            for view_id, score in knot_dict.items()
            for spherical in [carthesian_to_spherical(score["pos"])]
        ])

        return knot_dict, knots

    def _prepare_spherical_lut(self, knots: np.array,
                               method: str) -> (object, dict):
        """ Calculate spherical interpolation look-up table for given knots. """

        if method == "rbf":
            lut = scpi.Rbf(knots[:, 0], knots[:, 1], knots[:, 2], function="multiquadric")
            lut_kws = { }
        if method == "wrap_rbf":
            def great_circle_distance(u: np.array, v: np.array) -> float:
                """ Calculate great circle distance. """
                u_lats, v_lats = u[1], v[1]
                u_lons, v_lons = u[0], v[0]
                delta_lons = np.abs(v_lons - u_lons)
                return np.arctan2(
                    np.sqrt(
                        (np.cos(v_lats) * np.sin(delta_lons)) ** 2.0 +
                        (np.cos(u_lats) * np.sin(v_lats) - np.sin(u_lats) * np.cos(v_lats) * np.cos(delta_lons)) ** 2.0
                    ),
                    np.sin(u_lats) * np.sin(v_lats) + np.cos(u_lats) * np.cos(v_lats) * np.cos(delta_lons)
                )

            def wrap_around_norm(u: np.array, v: np.array) -> np.array:
                return great_circle_distance(u, v)

            lut = scpi.Rbf(knots[:, 0], knots[:, 1], knots[:, 2], function="gaussian", norm=wrap_around_norm)
            lut_kws = { }
        elif method == "smooth":
            lut = scpi.SmoothSphereBivariateSpline(knots[:, 0], knots[:, 1], knots[:, 2], s=32.0)
            lut_kws = { "grid": False }
        elif method == "rect":
            orig_resolution = (
                np.where(knots[1:, 0] == knots[0, 0])[0][0] + 1,
                np.where((knots[1:, 1] - knots[:-1, 1]) > 0.0)[0][0] + 1
            )
            fit_knots = [
                knots[:orig_resolution[0], 0],
                knots[::orig_resolution[1], 1],
                knots[:, 2].reshape((orig_resolution[0], orig_resolution[1]))
           ]
            fit_knots[0][0] += 0.0001
            fit_knots[1][-1] -= 0.0001
            lut = scpi.RectSphereBivariateSpline(fit_knots[0], fit_knots[1], fit_knots[2],
                                                 pole_continuity=False)
            lut_kws = { "grid": False }
        elif method == "lsq":
            orig_resolution = (
                np.where(knots[1:, 0] == knots[0, 0])[0][0] + 1,
                np.where((knots[1:, 1] - knots[:-1, 1]) > 0.0)[0][0] + 1
            )
            fit_knots = [
                knots[:orig_resolution[0], 0],
                knots[::orig_resolution[1], 1],
                knots[:, 2].reshape((orig_resolution[0], orig_resolution[1]))
           ]
            fit_knots[0][0] += 0.0001
            fit_knots[0][-1] -= 0.0001
            fit_knots[1][0] += 0.0001
            fit_knots[1][-1] -= 0.0001
            lut = scpi.LSQSphereBivariateSpline(knots[:, 0], knots[:, 1], knots[:, 2], fit_knots[0],
                                                fit_knots[1])
            lut_kws = { "grid": False }

        return lut, lut_kws

    def _prepare_spherical_smooth_grid(self, knots: np.array,
                                       lut: object, lut_kws: dict,
                                       resolution: tuple, visualize: bool = False,
                                       visualize_knots: bool = False) -> (np.array, np.array):
        """ Calculate smooth grid using provided knot points and a look-up table. """

        smooth_grid = np.meshgrid(np.linspace(0.0, np.pi, resolution[0]),
                                  np.linspace(0.0, 2.0 * np.pi, resolution[1]))
        smooth_data = np.reshape(lut(smooth_grid[0].ravel(), smooth_grid[1].ravel(), **lut_kws), resolution)

        if visualize:
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot()
            vmin, vmax = np.min(smooth_data), np.max(smooth_data)
            cmap = plt.get_cmap("viridis")
            ax.imshow(smooth_data, origin="lower", extent=(0.0, np.pi, 0.0, 2.0 * np.pi),
                      interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
            if visualize_knots:
                ax.scatter(knots[:, 0], knots[:, 1], c=knots[:, 2], s=45,
                           cmap=cmap, vmin=vmin, vmax=vmax)
                ax.scatter(knots[:, 0], knots[:, 1], c="red", s=5)
            ax.set_xlabel("Phi")
            ax.set_ylabel("Theta")
            plt.show()

        return smooth_data, smooth_grid

    def _prepare_spherical_map(self, knots: np.array,
                               resolutions: List[tuple],
                               methods: List[str],
                               visualize_map: bool = False,
                               visualize_views: bool = False,
                               visualize_view_count: int = 10) -> (object, dict):
        """ Create final look-up table for spherical coordinate views, mapping to scores. """

        if len(resolutions) == 0 or len(resolutions) != len(methods):
            return None, { }

        current_knots = knots
        for idx, (resolution, method) in enumerate(zip(resolutions, methods)):
            is_first = idx == 0
            is_last = idx == len(resolutions) - 1

            lut, lut_kws = self._prepare_spherical_lut(
                knots=current_knots, method=method
            )

            if not is_last:
                smooth_data, smooth_grid = self._prepare_spherical_smooth_grid(
                    knots=current_knots, lut=lut, lut_kws=lut_kws,
                    resolution=resolution,
                    visualize=visualize_map,
                    visualize_knots=is_first
                )
                current_knots = np.array([
                    [ phi, theta, score ]
                    for phi, theta, score in
                    zip(smooth_grid[0].ravel(), smooth_grid[1].ravel(), smooth_data.ravel())
                ])

        if visualize_views:
            smooth_data, smooth_grid = self._prepare_spherical_smooth_grid(
                knots=current_knots, lut=lut, lut_kws=lut_kws,
                resolution=resolutions[-1],
                visualize=visualize_map,
                visualize_knots=False
            )
            points = np.array([
                spherical_to_carthesian([ 1.0, phi, theta ])
                for phi, theta, score in
                zip(smooth_grid[0].ravel(), smooth_grid[1].ravel(), smooth_data.ravel())
            ])
            colors = np.array([
                score
                for phi, theta, score in
                zip(smooth_grid[0].ravel(), smooth_grid[1].ravel(), smooth_data.ravel())
            ])
            colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))
            cmap = plt.get_cmap("viridis")

            fig = plt.figure(figsize=(4 * visualize_view_count, 4))
            for idx, rotation in enumerate(np.linspace(0.0, 360.0, visualize_view_count + 1)[:-1]):
                ax = fig.add_subplot(1, visualize_view_count, idx + 1, projection="3d")
                ax.plot_surface(points[:, 0].reshape(smooth_data.shape),
                                points[:, 1].reshape(smooth_data.shape),
                                points[:, 2].reshape(smooth_data.shape),
                                rstride=1, cstride=1,
                                facecolors=cmap(colors.reshape(smooth_data.shape)))
                ax.set_axis_off()
                ax.view_init(0, rotation)

            plt.show()

        return lut, lut_kws

    def _prepare_spherical_scores(self, variant_jsons: dict,
                                  tree_scores: pd.DataFrame,
                                  lut: object, lut_kws: dict) -> dict:
        """ Calculate interpolated variant scores using look-up table. """

        scores = tree_scores.set_index("view_id")
        spherical_scores = {
            (view_id, variant_id): {
                "car_pos": view_pos,
                "sph_pos": sph_pos,
                "base_score": scores.loc[view_id].jod,
                "score": lut(sph_pos[0], sph_pos[1], **lut_kws)
            }
            for (view_id, variant_id), variant_json in variant_jsons.items()
            for view_pos in [
                scpt.Rotation.from_euler(
                    "XYZ", variant_json["tree"]["rotation"],
                    degrees=False
                ).apply(variant_json["camera"]["pos"])
            ]
            for sph_pos in [ carthesian_to_spherical(view_pos) ]
        }

        # Add spherical score for the complete tree.
        spherical_scores[( -1, 0 )] = {
            "car_pos": np.array([ 0.0, 0.0, 0.0 ]),
            "sph_pos": carthesian_to_spherical(np.array([ 0.0, 0.0, 0.0 ])),
            "base_score": scores.loc[-1].jod,
            "score": scores.loc[-1].jod,
        }

        return spherical_scores

    def _prepare_spherical_indexed_scores(self, base_path: str,
                                          view_catalogue: pd.DataFrame,
                                          tree_catalogue: pd.DataFrame,
                                          scores_indexed: pd.DataFrame) -> pd.DataFrame:
        """
        Augment indexed score data-frame with spherical interpolation
        for each view/tree variant.

        :param base_path: Base path where the data-set exists.
        :param view_catalogue: Catalogue containing all information
            about the views.
        :param tree_catalogue: Catalogue containing all information
            about trees.
        :param scores_indexed: Indexed scores for trees and views.

        :return: Returns data-frame indexed by ("tree_id", "view_id",
            "view_variant_id"), where view_id == -1 contains data for
            the whole tree. Result contains following columns:
            * tree_id, view_id, view_variant_id - Integer index for
                unique tree/view and the specific view variant.
            * jod, jod_low, jod_high, jod_var - JOD properties.
        """

        if len(scores_indexed) <= 0:
            self.__l.info(f"Input scores are empty, returning empty spherical scores frame!")
            return BaseDataLoader._create_empty_spherical_indexed_scores()

        all_views = view_catalogue.reset_index()
        all_scores = scores_indexed.reset_index()
        tree_ids = scores_indexed.index.unique(level=0)
        # TODO - Support tree variants.
        tree_variant = 0
        spherical_indexed_scores = [ ]

        self.__l.info(f"Preparing spherical indexed scores for {len(tree_catalogue)} trees...")

        loading_progress = LoadingBar("", max=len(tree_catalogue))
        for tree_id in tree_ids:
            # Calculate interpolations for each tree.
            tree_views = all_views[
                (all_views.tree_id == tree_id) &
                (all_views.tree_variant_id == tree_variant)
            ]
            tree_scores = all_scores[
                (all_scores.tree_id == tree_id) &
                (all_scores.tree_variant_id == tree_variant)
            ]

            # Prepare variants and load descriptions.
            variant_jsons = { }
            variants = set()
            for idx, row in tree_views.iterrows():
                variants.add(( row.tree_id, row.tree_variant_id, row.view_id, row.view_variant_id ))
                if (row.view_id, row.view_variant_id) in variant_jsons or row.json_path == "":
                    continue
                with open(f"{base_path}/{row.json_path}", "r") as jf:
                    variant_jsons[(row.view_id, row.view_variant_id)] = json.load(jf)
            # Add variant for the complete tree.
            variants.add(( tree_id, 0, -1, 0 ))

            if len(variant_jsons) == 0:
                # No variants or missing json descriptions -> Use existing scores.
                for variant in variants:
                    scores = scores_indexed.loc[(variant[0], variant[1], variant[2], variant[3])]
                    spherical_indexed_scores.append((
                        variant[0], variant[1], variant[2], variant[3],
                        # Use the same JOD for variant as the base.
                        scores.jod,
                        scores.jod, scores.jod_low, scores.jod_high, scores.jod_var
                    ))
                continue

            # Sanity check, we should always have at least view, variant with ID (0, 0).
            assert((0, 0) in variant_jsons)

            # Calculate spherical interpolation map.
            knot_dict, knots = self._prepare_spherical_knots(
                variant_jsons=variant_jsons, tree_scores=tree_scores
            )
            lut, lut_kws = self._prepare_spherical_map(
                # TODO - Parameterize this by script arguments.
                knots=knots,
                resolutions=[ (36, 36), (72, 72) ],
                methods=[ "wrap_rbf", "rbf" ],
                visualize_map=False, visualize_views=False,
                visualize_view_count=10
            )

            # Interpolate variant scores using the spherical map.
            spherical_scores = self._prepare_spherical_scores(
                variant_jsons=variant_jsons,
                tree_scores=tree_scores,
                lut=lut, lut_kws=lut_kws
            )

            # Save results.
            for variant in variants:
                scores = scores_indexed.loc[(variant[0], variant[1], variant[2], 0)]
                new_scores = spherical_scores[(variant[2], variant[3])]
                assert(scores.jod == new_scores["base_score"])
                spherical_indexed_scores.append((
                    variant[0], variant[1], variant[2], variant[3],
                    # Use the new interpolated JOD score.
                    new_scores["score"],
                    scores.jod, scores.jod_low, scores.jod_high, scores.jod_var
                ))
            loading_progress.next(1)
        loading_progress.finish()

        spherical_scores_indexed = pd.DataFrame(
            data=spherical_indexed_scores,
            columns=(
                "tree_id", "tree_variant_id", "view_id", "view_variant_id",
                "jod", "base_jod", "jod_low", "jod_high", "jod_var"
            )
        )
        spherical_scores_indexed.set_index(["tree_id", "tree_variant_id", "view_id", "view_variant_id"], inplace=True)
        spherical_scores_indexed.sort_index(inplace=True)

        self.__l.info(f"\tDone, prepared {len(spherical_indexed_scores)} spherical scores.")

        return spherical_scores_indexed

    def _load_tree_data(self, base_path: str = "",
                        tree_catalogue: pd.DataFrame = pd.DataFrame(),
                        load_node_data: bool = True,
                        allow_preloaded: bool = False) -> Dict[Tuple[int, int], TreeFile]:
        """ Load all tree data files from given catalogue. """

        self.__l.info(f"Loading tree data from {len(tree_catalogue)} .tree files...")

        tree_data = { }
        parsing_progress = ParsingBar("", max=len(tree_catalogue))
        for index, tree in tree_catalogue.iterrows():
            if allow_preloaded and tree.data is not None:
                tree_data[index] = tree.data
            else:
                tree_data[index] = TreeFile(
                    file_path=f"{base_path}/{tree.path}",
                    load_node=load_node_data,
                ) if tree.path else None
            parsing_progress.next(1)
        parsing_progress.finish()

        self.__l.info(f"\tDone, loaded {len(tree_data)} tree files.")

        return tree_data

    def _determine_available_features(self,
                                      view_catalogue: pd.DataFrame,
                                      tree_data: Dict[Tuple[int, int], TreeFile],
                                      load_node_data: bool) -> dict:
        """ Create a hierarchy of feature names available for use. """

        self.__l.info(f"Determining available features...")

        available_features = {
            "stat": np.unique([
                name
                for tree in tree_data.values()
                if tree is not None and "stats" in tree.dynamic_meta_data
                for name, item in tree.dynamic_meta_data["stats"].items()
                if TreeStatistic.is_stat_dict(item)
            ]),
            "image": np.unique([
                name
                for tree in tree_data.values()
                if tree is not None and "stats" in tree.dynamic_meta_data and "visual" in tree.dynamic_meta_data["stats"]
                for name, item in tree.dynamic_meta_data["stats"]["visual"].items()
                if TreeImage.is_image_dict(item)
            ]),
            "other": dict_of_lists([
                v.split(".") for v in np.unique([
                    f"{name}.{element}"
                    for tree in tree_data.values()
                    if tree is not None and "stats" in tree.dynamic_meta_data
                    for name, item in tree.dynamic_meta_data["stats"].items()
                    if not TreeStatistic.is_stat_dict(item) and name != "visual"
                    for element in item.keys()
                ])
            ]),
            "view": view_catalogue.reset_index().view_type.unique(),
            # TODO - Detect available skeleton features?
            "skeleton": [ "segment", "position", "thickness" ] if load_node_data else [ ]
        }

        totals = { name: len(features) for name, features in available_features.items() }
        self.__l.info(f"\tDone, found { totals } available features.")

        return available_features

    def _load_empty(self):
        """ Load empty data definitions. """

        self._full_results = self._create_empty_results()
        self._results = self._generate_reduced_results(
            full_results=self._full_results
        )
        self._users = self._create_empty_users()
        self._full_scores = self._create_empty_scores()
        self._scores = self._generate_reduced_scores(
            full_scores=self._full_scores
        )
        self._full_scores_indexed = self._create_empty_indexed_scores()
        self._scores_indexed = self._generate_reduced_scores_indexed(
            full_scores_indexed=self._full_scores_indexed
        )
        self._full_view_catalogue = self._create_empty_view_catalogue()
        self._view_catalogue = self._generate_reduced_view_catalogue(
            full_view_catalogue=self._full_view_catalogue
        )
        self._full_tree_catalogue = self._create_empty_tree_catalogue()
        self._tree_catalogue = self._generate_reduced_tree_catalogue(
            full_tree_catalogue=self._full_tree_catalogue
        )
        self._spherical_scores_indexed = self._create_empty_spherical_indexed_scores()
        self._tree_data = self._create_empty_tree_data()
        self._available_features = self._create_empty_available_features()
        self._view_base_path = self._create_empty_dataset_path()
        self._dataset_meta = self._create_empty_dataset_meta()

    def _load_as_dataset(self, dataset_path: str, use_dithered: bool,
                         use_augment: bool, use_augment_variants: Optional[int]):
        """ Load data as a pre-exported data-set. """

        results_path = f"{dataset_path}/results.csv"
        if not os.path.isfile(results_path):
            raise RuntimeError(f"Dataset at \"{dataset_path}\" does not contain results.csv!")

        users_path = f"{dataset_path}/users.csv"
        if not os.path.isfile(users_path):
            raise RuntimeError(f"Dataset at \"{dataset_path}\" does not contain users.csv!")

        scores_path = f"{dataset_path}/scores.csv"
        if not os.path.isfile(scores_path):
            raise RuntimeError(f"Dataset at \"{dataset_path}\" does not contain scores.csv!")

        scores_indexed_path = f"{dataset_path}/scores_indexed.csv"
        if not os.path.isfile(scores_indexed_path):
            self.__l.warning(f"Dataset at \"{dataset_path}\" does not contain scores_indexed.csv, using a dummy!")
            dummy_df = BaseDataLoader._create_empty_indexed_scores()
            dummy_df.to_csv(scores_indexed_path, sep=";", index=True)

        spherical_scores_indexed_path = f"{dataset_path}/spherical_scores_indexed.csv"
        if not os.path.isfile(spherical_scores_indexed_path):
            self.__l.warning(f"Dataset at \"{dataset_path}\" does not contain spherical_scores_indexed.csv, using a dummy!")
            dummy_df = BaseDataLoader._create_empty_spherical_indexed_scores()
            dummy_df.to_csv(spherical_scores_indexed_path, sep=";", index=True)

        view_catalogue_path = f"{dataset_path}/view_catalogue.csv"
        if not os.path.isfile(view_catalogue_path):
            raise RuntimeError(f"Dataset at \"{dataset_path}\" does not contain view_catalogue.csv!")

        tree_catalogue_path = f"{dataset_path}/tree_catalogue.csv"
        if not os.path.isfile(tree_catalogue_path):
            raise RuntimeError(f"Dataset at \"{dataset_path}\" does not contain tree_catalogue.csv!")

        dataset_meta_path = f"{dataset_path}/dataset_meta.json"
        if not os.path.isfile(dataset_meta_path):
            raise RuntimeError(f"Dataset at \"{dataset_path}\" does not contain dataset_meta.json!")

        results = pd.read_csv(results_path, sep=";")
        if "first_view_variant_id" not in results:
            # Old-style dataset -> Add new columns:
            results["first_tree_variant_id"] = 0
            results["first_view_variant_id"] = 0
            results["second_tree_variant_id"] = 0
            results["second_view_variant_id"] = 0

        users = pd.read_csv(users_path, sep=";")
        users.set_index(["worker_id"], inplace=True)

        scores = pd.read_csv(scores_path, sep=";")
        if "tree_variant_id" not in scores:
            # Old-style dataset -> Add new columns:
            scores["tree_variant_id"] = 0
        scores["tree_id"] = scores["tree_id"].astype(int)
        scores.set_index(["tree_id", "tree_variant_id"])

        view_catalogue = pd.read_csv(view_catalogue_path, sep=";")
        view_catalogue["data"] = None
        if "view_variant_id" not in view_catalogue:
            # Old-style dataset -> Add new columns:
            view_catalogue["tree_variant_id"] = 0
            view_catalogue["view_variant_id"] = 0
            view_catalogue["json_path"] = ""
        view_catalogue.loc[pd.isna(view_catalogue.json_path), "json_path"] = ""
        view_catalogue.set_index(["tree_id", "tree_variant_id", "view_id", "view_variant_id", "view_type"], inplace=True)

        tree_catalogue = pd.read_csv(tree_catalogue_path, sep=";")
        tree_catalogue["data"] = None
        if "tree_variant_id" not in tree_catalogue:
            # Old-style dataset -> Add new columns:
            tree_catalogue["tree_variant_id"] = 0
            tree_catalogue["json_path"] = ""
        tree_catalogue.loc[pd.isna(tree_catalogue.json_path), "json_path"] = ""
        tree_catalogue.set_index(["tree_id", "tree_variant_id"], inplace=True)

        with open(dataset_meta_path, "r") as f:
            dataset_meta = json.load(f)

        scores_indexed = pd.read_csv(scores_indexed_path, sep=";")
        scores_indexed.set_index(["tree_id", "tree_variant_id", "view_id", "view_variant_id"], inplace=True)
        if len(scores_indexed) == 0:
            # Try re-calculating empty scores.
            self.__l.warning("Indexed scores not cached, re-calculating!")
            scores_indexed = self._index_scores(
                scores=scores
            )
            if len(scores_indexed) != 0:
                self.__l.warning("New indexed scores are valid, saving!")
                scores_indexed.to_csv(scores_indexed_path, sep=";", index=True)

        if not self._check_view_tree_catalogue(
                base_path=dataset_path,
                view_catalogue=view_catalogue,
                tree_catalogue=tree_catalogue,
                scores_indexed=scores_indexed):
            raise RuntimeError("Invalid data-set specified!")

        spherical_scores_indexed = pd.read_csv(spherical_scores_indexed_path, sep=";")
        spherical_scores_indexed["tree_id"] = spherical_scores_indexed["tree_id"].astype(int)
        spherical_scores_indexed["tree_variant_id"] = spherical_scores_indexed["tree_variant_id"].astype(int)
        spherical_scores_indexed["view_id"] = spherical_scores_indexed["view_id"].astype(int)
        spherical_scores_indexed["view_variant_id"] = spherical_scores_indexed["view_variant_id"].astype(int)
        spherical_scores_indexed.set_index(["tree_id", "tree_variant_id", "view_id", "view_variant_id"], inplace=True)
        if len(spherical_scores_indexed) == 0:
            # Try re-calculating empty scores.
            self.__l.warning("Spherical indexed scores not cached, re-calculating!")
            spherical_scores_indexed = self._prepare_spherical_indexed_scores(
                base_path=dataset_path,
                view_catalogue=view_catalogue,
                tree_catalogue=tree_catalogue,
                scores_indexed=scores_indexed
            )
            if len(spherical_scores_indexed) != 0:
                self.__l.warning("New spherical indexed scores are valid, saving!")
                spherical_scores_indexed.to_csv(spherical_scores_indexed_path, sep=";", index=True)

        # Filter dithered views if disabled.
        if not use_dithered:
            results = results[
                (results.first_view_variant_id == 0) & (results.second_view_variant_id == 0)
            ]
            view_catalogue = view_catalogue[
                (view_catalogue.index.get_level_values("view_variant_id") == 0)
            ]
            scores_indexed = scores_indexed[
                (scores_indexed.index.get_level_values("view_variant_id") == 0)
            ]
            spherical_scores_indexed = spherical_scores_indexed[
                (spherical_scores_indexed.index.get_level_values("view_variant_id") == 0)
            ]

        # Filter augmented trees if disabled.
        total_augment_variants = len(tree_catalogue.index.unique(level="tree_variant_id"))
        if use_augment_variants is None:
            use_augment_variants = total_augment_variants
        if not use_augment:
            use_augment_variants = 0

        if use_augment_variants < total_augment_variants:
            results = results[
                (results.first_tree_variant_id <= use_augment_variants) & \
                (results.second_tree_variant_id <= use_augment_variants)
            ]
            scores = scores[scores.tree_variant_id <= use_augment_variants]
            view_catalogue = view_catalogue[
                (view_catalogue.index.get_level_values("tree_variant_id") <= use_augment_variants)
            ]
            tree_catalogue = tree_catalogue[
                (tree_catalogue.index.get_level_values("tree_variant_id") <= use_augment_variants)
            ]
            scores_indexed = scores_indexed[
                (scores_indexed.index.get_level_values("tree_variant_id") <= use_augment_variants)
            ]
            spherical_scores_indexed = spherical_scores_indexed[
                (spherical_scores_indexed.index.get_level_values("tree_variant_id") <= use_augment_variants)
            ]

        tree_data = self._load_tree_data(
            base_path=dataset_path,
            tree_catalogue=tree_catalogue,
            load_node_data=self.c.load_node_data
        )

        available_features = self._determine_available_features(
            view_catalogue=view_catalogue,
            tree_data=tree_data,
            load_node_data=self.c.load_node_data
        )

        self._full_results = results
        self._results = self._generate_reduced_results(
            full_results=self._full_results
        )
        self._users = users
        self._full_scores = scores
        self._scores = self._generate_reduced_scores(
            full_scores=self._full_scores
        )
        self._full_scores_indexed = scores_indexed
        self._scores_indexed = self._generate_reduced_scores_indexed(
            full_scores_indexed=self._full_scores_indexed
        )
        self._full_view_catalogue = view_catalogue
        self._view_catalogue = self._generate_reduced_view_catalogue(
            full_view_catalogue=self._full_view_catalogue
        )
        self._full_tree_catalogue = tree_catalogue
        self._tree_catalogue = self._generate_reduced_tree_catalogue(
            full_tree_catalogue=self._full_tree_catalogue
        )
        self._spherical_scores_indexed = spherical_scores_indexed
        self._tree_data = tree_data
        self._available_features = available_features
        self._view_base_path = dataset_path
        self._dataset_meta = dataset_meta

    def _extract_results(self, raw_results: pd.DataFrame) -> pd.DataFrame:
        """
        Convert given raw results data into reduced results date-frame.

        :param raw_results: Raw input results data-frame.

        :return: Returns reduced results data-frame with following columns:
            * index - Integer without any ordering.
            * first_tree_id, first_view_id - Integer values.
            * second_tree_id, second_view_id - Integer values.
            * worker_id - String unique identifier of the worker.
            * choice - Integer 1 for first or 2 for second.
        """

        self.__l.info(f"Extracting results from {len(raw_results)} raw results...")

        #self.__l.error("SKIPPING RESULT EXTRACTION!")
        #return self._create_empty_results()

        def convert_row(row):
            first_groups = re.match(".*/tree([0-9]+)_view([0-9]+).*.png", row.first_url)
            second_groups = re.match(".*/tree([0-9]+)_view([0-9]+).*.png", row.second_url)
            return pd.Series({
                # TODO - Add support for tree/view variants.
                "first_tree_id": int(first_groups.group(1)),
                "first_tree_variant_id": 0,
                "first_view_id": int(first_groups.group(2)),
                "first_view_variant_id": 0,
                "second_tree_id": int(second_groups.group(1)),
                "second_tree_variant_id": 0,
                "second_view_id": int(second_groups.group(2)),
                "second_view_variant_id": 0,
                "worker_id": row.worker_id,
                "choice": int(row.choice)
            })

        results = raw_results.swifter.apply(func=convert_row, axis=1)

        self.__l.info(f"\tExtraction complete, resulting in {len(results)} records.")

        return results

    @staticmethod
    def _create_empty_users() -> pd.DataFrame:
        """ Create empty results dataframe as per _extract_users return. """

        return pd.DataFrame({
            "worker_id": str(),
            "task_count": int(),
            "average_choice": float(),
            "gender": str(),
            "age": str(),
            "education": str()
        }, index=[ ]).set_index("worker_id")

    def _extract_users(self, raw_results: pd.DataFrame) -> pd.DataFrame:
        """
        Convert given raw results data into a list of users.

        :param raw_results: Raw input results data-frame.

        :return: Returns users data-frame with following columns:
            * worker_id - String unique identifier of the worker.
            * task_count - Number of tasks filled by this user.
            * average_choice - Average choice picked by this user.
            * gender - Gender chosen by the user.
            * age - Age group chosen by the user.
            * education - Education chosen by the user.
        """

        self.__l.info(f"Extracting users from {len(raw_results)} raw results...")

        users = raw_results.groupby("worker_id").agg(
            {
                "mturk": "count", "choice": "mean",
                "gender": lambda x: x.mode(),
                "age": lambda x: x.mode(),
                "education": lambda x: x.mode(),
                "comment": lambda x: len(','.join(list(np.unique([ str(y) for y in x if str(y) != "nan" ])))),
            }).rename(
            {
                "mturk": "task_count",
                "choice": "average_choice"
            }, axis=1)

        self.__l.info(f"\tExtraction complete, resulting in {len(users)} users.")

        return users

    def _create_results_users(self,
                              tree_comparisons: List[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]], int]]
                              ) -> (pd.DataFrame, pd.DataFrame):
        """ Create results dataframe, along with dummy user information from given pairwise data. """

        empty_results = BaseDataLoader._create_empty_results()
        empty_users = BaseDataLoader._create_empty_users()

        if len(tree_comparisons) == 0:
            return empty_results, empty_users

        worker_id = "dummyworker"

        # Create pairwise records and assign them to the dummy user.
        results = pd.DataFrame([
            ( fti, fvi, sti, svi, worker_id, c )
            for ( ( fti, fvi ), ( sti, svi ), c ) in tree_comparisons
        ], columns=empty_results.columns)

        # Create dummy user record.
        users = pd.DataFrame([
            ( worker_id, len(results), results.choice.mean(), "M", "20-30", "Bachelors" )
        ])

        return results, users


    def _compile_scores(self, tree_scores: pd.DataFrame,
                        view_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Compile scores from given tree and view scores into one data-frame.

        :param tree_scores: Data-frame containing raw tree scores.
        :param view_scores: Data-frame containing raw view scores.

        :return: Returns scores data-frame with following columns:
            * index - Integer tree identifier.
            * tree_jod, tree_jod_low, tree_jod_high, tree_jod_var -
                Floating point values representing JOD properties for
                the whole tree.
            * view#_jod, view#_jod_low, view#_jod_high, view#_jod_var -
                Floating point values representing JOD properties for
                the #th view. # goes from 0 to number of views - 1.
        """

        self.__l.info(f"Compiling scores from {len(tree_scores)} trees "
                      f"and {len(view_scores)} views...")

        def extract_identifiers(row):
            groups = re.match(".*/tree([0-9]+)_view([0-9])+.*.png", row.condition)
            tree_id = int(groups.group(1))
            view_id = int(groups.group(2))
            return pd.Series({
                "tree_id": tree_id,
                "view_id": view_id
            })

        view_scores_indexed = view_scores
        identifiers = view_scores_indexed.apply(func=extract_identifiers, axis=1)
        view_scores_indexed["tree_id"] = identifiers.tree_id
        view_scores_indexed["view_id"] = identifiers.view_id
        view_scores_indexed.set_index(["tree_id", "view_id"], inplace=True)

        def convert_row(row):
            tree_id = int(re.match("tree([0-9]+)", row.condition).group(1))
            views = view_scores_indexed.loc[[tree_id, ]]
            return pd.Series({
                # TODO - Add support for tree variants.
                "tree_id": tree_id,
                "tree_variant_id": 0,
                "tree_jod": row["jod"],
                "tree_jod_low": row["jod_low"],
                "tree_jod_high": row["jod_high"],
                "tree_jod_var": row["var"],
                **dict(itertools.chain.from_iterable(
                    [[
                        (f"view{index[1]}_jod", view["jod"]),
                        (f"view{index[1]}_jod_low", view["jod_low"]),
                        (f"view{index[1]}_jod_high", view["jod_high"]),
                        (f"view{index[1]}_jod_var", view["var"]),
                    ] for index, view in views.iterrows() ]
                ))
            })

        scores = tree_scores.apply(func=convert_row, axis=1)
        scores["tree_id"] = scores["tree_id"].astype("int64")
        scores["tree_variant_id"] = scores["tree_variant_id"].astype("int64")
        scores.set_index(["tree_id", "tree_variant_id"], inplace=True)

        if scores.count().min() != len(scores):
            raise RuntimeError("Some trees do not contain the same amount of views!")

        self.__l.info(f"\tCompilation complete, resulting in {len(scores)} records.")

        return scores

    def _recover_scores(self, tree_info: Dict[Tuple[int, int], dict]) -> pd.DataFrame:
        """ Recover scores from tree information structure. See _load_additional_data and _compile_scores."""

        self.__l.info(f"Recovering scores for {len(tree_info)} trees...")

        tree_score_dicts = [ ]
        for tree_id, tree_data in tree_info.items():
            if "score" not in tree_data or tree_data["score"] is None:
                continue

            tree_score_dict = {
                key: value
                for view_id, view_data in tree_data["score"].items()
                for prefix in [
                    "tree" if view_id[0] < 0 else
                    f"view{view_id[0]}"
                ]
                for key, value in [
                    (f"{prefix}_jod", view_data["jod"]),
                    (f"{prefix}_jod_low", view_data["jod_low"]),
                    (f"{prefix}_jod_high", view_data["jod_high"]),
                    (f"{prefix}_jod_var", view_data["jod_var"]),
                ]
            }

            tree_score_dict["tree_id"] = tree_id[0]
            tree_score_dict["tree_variant_id"] = tree_id[1]
            tree_score_dicts.append(tree_score_dict)

        if len(tree_score_dicts) <= 0:
            self.__l.info("No tree scores found, returning empty!")
            return BaseDataLoader._create_empty_scores()

        scores = pd.DataFrame(tree_score_dicts)
        scores["tree_id"] = scores["tree_id"].astype("int64")
        scores["tree_variant_id"] = scores["tree_variant_id"].astype("int64")

        if scores.count().min() != len(scores):
            raise RuntimeError("Some trees do not contain the same amount of views!")

        self.__l.info(f"\tCompilation complete, resulting in {len(scores)} records.")

        return scores

    def _prepare_views_trees(self, tree_image_path: str, load_dither: bool
                             ) -> (pd.DataFrame, pd.DataFrame):
        """
        Create catalogue of views and trees available from given path.

        :param tree_image_path: Path to the root of the view data-set.
        :param load_dither: Load view variants from tree#/dither?

        :return: Returns data-frame containing catalogue of all
            located views. Resulting data-frame has following columns:
            * tree_id - Integer identifier of the tree.
            * view_id - Integer identifier of the view.
            * view_variant_id - Integer identifier of the view variant.
            * view_type - String specification of the view type.
            * path - Path to the view file, relative to the data-set base.
            * json_path - Path to the description json file, if available,
                relative to the data-set base.
            * data - Pre-loaded data, in this case None.
            and data-frame containing all located trees:
            * tree_id - Integer identifier of the tree.
            * tree_variant_id - Integer identifier of the tree variant.
            * path - Path to the tree file, relative to the data-set base.
            * json_path - Path to the description json file, if available,
                relative to the data-set base.
            * data - Pre-loaded data, in this case None.
        """

        self.__l.info(f"Preparing views from the base folder \"{tree_image_path}\"...")

        tree_folder_re = re.compile("[tT]ree([0-9]+)")
        tree_re = re.compile("[tT]ree([0-9]+)_variant([0-9]+).tree$")
        view_re = re.compile(".*_(?:screen|view)_([0-9]+)(?:_variant_([0-9]+))?(?:_([^_]+))?.png$")
        tree_folder_list = list(filter(tree_folder_re.match, os.listdir(tree_image_path)))

        tree_view_data = [ ]
        tree_data = [ ]

        for tree_folder in tree_folder_list:
            if load_dither:
                tree_view_folder = f"{tree_image_path}/{tree_folder}/dither"
            else:
                tree_view_folder = f"{tree_image_path}/{tree_folder}/views"
            if not os.path.exists(tree_view_folder):
                self.__l.warning(f"Tree \"{tree_folder}\" does NOT have any views associated, skipping!")
                continue

            tree_folder_match = re.match(tree_folder_re, tree_folder)
            tree_folder_id = int(tree_folder_match.group(1))
            tree_folder_name = f"tree{tree_folder_id}"

            tree_list = list(filter(tree_re.match, os.listdir(f"{tree_image_path}/{tree_folder}")))
            for tree in tree_list:
                tree_match = re.match(tree_re, tree)
                tree_id = int(tree_match.group(1))
                if tree_id != tree_folder_id:
                    self.__l.warning(f"Tree \"{tree_folder}\" contains misidentified tree \"{tree}\"!")
                if tree_match.group(2):
                    tree_variant_id = int(tree_match.group(2))
                else:
                    tree_variant_id = 0

                tree_path = pathlib.Path(f"{tree_image_path}/{tree_folder}/{tree}").absolute()
                tree_json_path = tree_path.with_suffix(".json")

                tree_data.append((
                    tree_id, tree_variant_id,
                    f"{tree_folder_name}/{tree_path.name}",
                    f"{tree_folder_name}/{tree_json_path.name}" \
                        if tree_json_path.is_file() \
                        else "",
                    None
                ))

            view_list = list(filter(view_re.match, os.listdir(tree_view_folder)))
            for view in view_list:
                view_match = re.match(view_re, view)
                view_id = int(view_match.group(1))
                if view_match.group(2):
                    view_variant_id = int(view_match.group(2))
                else:
                    view_variant_id = 0
                if view_match.group(3):
                    view_type = view_match.group(3)
                else:
                    view_type = "base"

                view_path = pathlib.Path(f"{tree_view_folder}/{view}").absolute()
                view_json_path = view_path.with_suffix(".json")

                tree_view_data.append((
                    tree_id, tree_variant_id, view_id, view_variant_id, view_type,
                    f"{tree_folder_name}/{view_path.parts[-2]}/{view_path.name}",
                    f"{tree_folder_name}/{view_path.parts[-2]}/{view_json_path.name}" \
                        if view_json_path.is_file() \
                        else "",
                    None
                ))

        view_catalogue = pd.DataFrame(
            data=tree_view_data,
            columns=("tree_id", "tree_variant_id", "view_id", "view_variant_id", "view_type", "path", "json_path", "data")
        )
        view_catalogue.set_index(["tree_id", "tree_variant_id", "view_id", "view_variant_id", "view_type"], inplace=True)
        view_catalogue.sort_index(inplace=True)

        tree_catalogue = pd.DataFrame(
            data=tree_data,
            columns=("tree_id", "tree_variant_id", "path", "json_path", "data")
        )
        tree_catalogue.set_index(["tree_id", "tree_variant_id"], inplace=True)
        tree_catalogue.sort_index(inplace=True)

        self.__l.info(f"\tView preparation complete, resulting in {len(view_catalogue)} view and {len(tree_catalogue)} tree records.")

        return view_catalogue, tree_catalogue

    @staticmethod
    def _create_empty_views() -> pd.DataFrame:
        """ Create empty results dataframe as per _prepare_views_trees return. """

        return BaseDataLoader._create_empty_view_catalogue()

    @staticmethod
    def _create_empty_trees() -> pd.DataFrame:
        """ Create empty results dataframe as per _prepare_views_trees return. """

        return BaseDataLoader._create_empty_tree_catalogue()

    def _recover_views_trees(self, tree_info: Dict[Tuple[int, int], dict]) -> (pd.DataFrame, pd.DataFrame):
        """ Recover views and tree data from given tree information. See _load_additional_data and _prepare_views_trees. """

        self.__l.info(f"Recovering views and trees from {len(tree_info)} trees...")

        tree_data = [ ]
        tree_view_data = [ ]

        for tree_id, tree_spec in tree_info.items():
            tree_file = tree_spec.get("data", None)
            tree_data.append((
                tree_id[0], tree_id[1],
                "", "", tree_file
            ))

            tree_views = tree_spec.get("view", None)
            if tree_views is not None:
                for tree_view_spec, tree_view_image in tree_views.items():
                    tree_view_data.append((
                        tree_id[0], tree_id[1],
                        tree_view_spec[0][0], tree_view_spec[0][1], tree_view_spec[1],
                        "", "", tree_view_image
                    ))

        if len(tree_view_data) <= 0:
            self.__l.info("No tree views found, using empty data-frame!")
            view_catalogue = BaseDataLoader._create_empty_views()
        else:
            view_catalogue = pd.DataFrame(
                data=tree_view_data,
                columns=("tree_id", "tree_variant_id", "view_id", "view_variant_id", "view_type", "path", "json_path", "data")
            )
            view_catalogue.set_index(["tree_id", "tree_variant_id", "view_id", "view_variant_id", "view_type"], inplace=True)
            view_catalogue.sort_index(inplace=True)

        if len(tree_data) <= 0:
            self.__l.info("No trees found, using empty data-frame!")
            tree_catalogue = BaseDataLoader._create_empty_trees()
        else:
            tree_catalogue = pd.DataFrame(
                data=tree_data,
                columns=("tree_id", "tree_variant_id", "path", "json_path", "data")
            )
            tree_catalogue.set_index(["tree_id", "tree_variant_id"], inplace=True)
            tree_catalogue.sort_index(inplace=True)

        self.__l.info(f"\tView preparation complete, resulting in {len(view_catalogue)} view and {len(tree_catalogue)} tree records.")

        return view_catalogue, tree_catalogue

    def _prepare_dataset_meta(self,
                              results: pd.DataFrame, users: pd.DataFrame, scores: pd.DataFrame,
                              view_catalogue: pd.DataFrame,
                              tree_catalogue: pd.DataFrame,
                              tree_data: dict) -> dict:
        """ Prepare meta-data structure for given data. """

        unique_id = secrets.token_hex(16)

        return {
            "unique_id": unique_id
        }

    def _load_as_raw_data(self, result_file: str, tree_scores_file: str,
                          view_scores_file: str, tree_image_path: str,
                          load_dither: bool, load_node_data: bool):
        """ Load data from raw structures. """

        raw_results = pd.read_csv(result_file, sep=";")

        tree_scores = pd.read_csv(tree_scores_file, sep=";")
        view_scores = pd.read_csv(view_scores_file, sep=";")

        results = self._extract_results(
            raw_results=raw_results
        )
        users = self._extract_users(
            raw_results=raw_results
        )
        scores = self._compile_scores(
            tree_scores=tree_scores,
            view_scores=view_scores
        )
        scores_indexed = self._index_scores(
            scores=scores
        )

        view_catalogue, tree_catalogue = self._prepare_views_trees(
            tree_image_path=tree_image_path,
            load_dither=load_dither
        )

        if not self._check_view_tree_catalogue(
                base_path=tree_image_path,
                view_catalogue=view_catalogue,
                tree_catalogue=tree_catalogue,
                scores_indexed=scores_indexed):
            raise RuntimeError("Invalid data-set specified!")

        spherical_scores_indexed = self._prepare_spherical_indexed_scores(
            base_path=tree_image_path,
            view_catalogue=view_catalogue,
            tree_catalogue=tree_catalogue,
            scores_indexed=scores_indexed
        )

        tree_data = self._load_tree_data(
            base_path=tree_image_path,
            tree_catalogue=tree_catalogue,
            load_node_data=load_node_data
        )

        dataset_meta = self._prepare_dataset_meta(
            results=results, users=users, scores=scores,
            view_catalogue=view_catalogue,
            tree_catalogue=tree_catalogue,
            tree_data=tree_data
        )

        available_features = self._determine_available_features(
            view_catalogue=view_catalogue,
            tree_data=tree_data,
            load_node_data=load_node_data
        )

        self._full_results = results
        self._results = self._generate_reduced_results(
            full_results=self._full_results
        )
        self._users = users
        self._full_scores = scores
        self._scores = self._generate_reduced_scores(
            full_scores=self._full_scores
        )
        self._full_scores_indexed = scores_indexed
        self._scores_indexed = self._generate_reduced_scores_indexed(
            full_scores_indexed=self._full_scores_indexed
        )
        self._full_view_catalogue = view_catalogue
        self._view_catalogue = self._generate_reduced_view_catalogue(
            full_view_catalogue=self._full_view_catalogue
        )
        self._full_tree_catalogue = tree_catalogue
        self._tree_catalogue = self._generate_reduced_tree_catalogue(
            full_tree_catalogue=self._full_tree_catalogue
        )
        self._spherical_scores_indexed = spherical_scores_indexed
        self._tree_data = tree_data
        self._available_features = available_features
        self._view_base_path = tree_image_path
        self._dataset_meta = dataset_meta

    def _load_additional_data(self, tree_info: Dict[Tuple[int, int], dict],
                              tree_comparisons: List[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]], int]] = [],
                              node_data_loaded: bool = False):
        """
        Load data from prepared structures.

        :param tree_info: Pre-loaded tree data, each element
            must contain following keys:
            - "data": Optional[TreeFile] - Prepared tree information.
            - "view": Optional[Dict[(Tuple[int, int], str), TreeImage]] - Prepared
                view information, mapping ((view id, view variant id), view_type) to loaded
                image.
            - "score": Optional[Dict[Tuple[int, int], Dict[str, float]]] - prepared
                tree scores, indexed by (view, variant) (-1 for the whole tree).
                Each internal dictionary must contain:
                    "jod", "jod_low", "jod_high" and "jod_var"
        :param tree_comparisons: Pre-loaded tree comparison data.
        :param node_data_loaded: Does the tree_info structure contain
            pre-loaded node data as well?
        """

        # Create empty versions for unavailable data.
        results, users = self._create_results_users(
            tree_comparisons=tree_comparisons
        )

        scores = self._recover_scores(
            tree_info=tree_info
        )
        scores_indexed = self._index_scores(
            scores=scores
        )

        view_catalogue, tree_catalogue = self._recover_views_trees(
            tree_info=tree_info
        )

        # TODO - Check the user-provided data?

        tree_data = self._load_tree_data(
            tree_catalogue=tree_catalogue,
            allow_preloaded=True,
            load_node_data=node_data_loaded
        )

        dataset_meta = self._prepare_dataset_meta(
            results=results, users=users, scores=scores,
            view_catalogue=view_catalogue,
            tree_catalogue=tree_catalogue,
            tree_data=tree_data
        )

        available_features = self._determine_available_features(
            view_catalogue=view_catalogue,
            tree_data=tree_data,
            load_node_data=node_data_loaded
        )

        self._full_results = results
        self._results = self._generate_reduced_results(
            full_results=self._full_results
        )
        self._users = users
        self._full_scores = scores
        self._scores = self._generate_reduced_scores(
            full_scores=self._full_scores
        )
        self._full_scores_indexed = scores_indexed
        self._scores_indexed = self._generate_reduced_scores_indexed(
            full_scores_indexed=self._full_scores_indexed
        )
        self._full_view_catalogue = view_catalogue
        self._view_catalogue = self._generate_reduced_view_catalogue(
            full_view_catalogue=self._full_view_catalogue
        )
        self._full_tree_catalogue = tree_catalogue
        self._tree_catalogue = self._generate_reduced_tree_catalogue(
            full_tree_catalogue=self._full_tree_catalogue
        )
        self._spherical_scores_indexed = scores_indexed
        self._tree_data = tree_data
        self._available_features = available_features
        self._view_base_path = ""
        self._dataset_meta = dataset_meta

    @property
    def full_results(self) -> pd.DataFrame:
        """
        Get results data-frame.

        :return: Returns results data-frame with following columns:
            * index - Integer without any ordering.
            * first_tree_id, first_tree_variant_id,
                first_view_id, first_view_variant_id - Integer values.
            * second_tree_id, second_tree_variant_id,
                second_view_id, second_view_variant_id - Integer values.
            * worker_id - String unique identifier of the worker.
            * choice - Integer 1 for first or 2 for second.
        """

        return self._full_results

    @property
    def results(self) -> pd.DataFrame:
        """
        Get results data-frame.

        :return: Returns results data-frame with following columns:
            * index - Integer without any ordering.
            * first_tree_id, first_view_id - Integer values.
            * second_tree_id, second_view_id - Integer values.
            * worker_id - String unique identifier of the worker.
            * choice - Integer 1 for first or 2 for second.
        """

        return self._results

    @property
    def users(self) -> pd.DataFrame:
        """
        Get users data-frame.

        :return: Returns users data-frame with following columns:
            * worker_id - String unique identifier of the worker.
            * task_count - Number of tasks filled by this user.
            * average_choice - Average choice picked by this user.
            * gender - Gender chosen by the user.
            * age - Age group chosen by the user.
            * education - Education chosen by the user.
        """

        return self._users

    @property
    def full_scores(self) -> pd.DataFrame:
        """
        Get scores data-frame.

        :return: Returns scores data-frame with following columns:
            * tree_id - Integer tree identifier.
            * tree_variant_id - Integer tree variant identifier.
            * tree_jod, tree_jod_low, tree_jod_high, tree_jod_var -
                Floating point values representing JOD properties for
                the whole tree.
            * view#_jod, view#_jod_low, view#_jod_high, view#_jod_var -
                Floating point values representing JOD properties for
                the #th view. # goes from 0 to number of views - 1.
        """

        return self._full_scores

    @property
    def scores(self) -> pd.DataFrame:
        """
        Get scores data-frame.

        :return: Returns scores data-frame with following columns:
            * index - Integer tree identifier.
            * tree_jod, tree_jod_low, tree_jod_high, tree_jod_var -
                Floating point values representing JOD properties for
                the whole tree.
            * view#_jod, view#_jod_low, view#_jod_high, view#_jod_var -
                Floating point values representing JOD properties for
                the #th view. # goes from 0 to number of views - 1.
        """

        return self._scores

    @property
    def full_scores_indexed(self) -> pd.DataFrame:
        """
        Get indexed scores data-frame.

        :return: Returns indexed scores data-frame with following columns:
            * tree_id, tree_variant_id, view_id, view_variant_id - Integer
                index for unique tree/view, view_id == -1 for the complete tree.
            * jod, jod_low, jod_high, jod_var - JOD properties.
        """

        return self._full_scores_indexed

    @property
    def scores_indexed(self) -> pd.DataFrame:
        """
        Get indexed scores data-frame.

        :return: Returns indexed scores data-frame with following columns:
            * tree_id, view_id - Integer index for unique tree/view,
                view_id == -1 for the complete tree.
            * jod, jod_low, jod_high, jod_var - JOD properties.
        """

        return self._scores_indexed

    @property
    def spherical_scores_indexed(self) -> pd.DataFrame:
        """
        Get indexed spherically interpolated scores data-frame.

        :return: Returns indexed scores data-frame with following columns:
            * tree_id, view_id, view_variant_id - Integer index for unique
                tree/view variant, view_id == -1 for the complete tree.
            * jod, base_jod, jod_low, jod_high, jod_var - JOD properties.
        """

        return self._spherical_scores_indexed

    @property
    def full_view_catalogue(self) -> pd.DataFrame:
        """
        Get view catalogue data-frame.

        :return: Returns view catalogue data-frame with following columns:
            * tree_id - Integer identifier of the tree.
            * view_id - Integer identifier of the view.
            * view_variant_id - Integer identifier of the view variant.
            * view_type - String specification of the view type.
            * path - Path to the view file, relative to the data-set base.
                May be empty in which case data should contain pre-loaded
                data.
            * json_path - Path to the description file, relative to the data-set base.
                May be empty for old or pre-loaded data.
            * data - Pre-loaded data, may be None.
        """

        return self._full_view_catalogue

    @property
    def view_catalogue(self) -> pd.DataFrame:
        """
        Get view catalogue data-frame.

        :return: Returns view catalogue data-frame with following columns:
            * tree_id - Integer identifier of the tree.
            * view_id - Integer identifier of the view.
            * view_type - String specification of the view type.
            * path - Path to the view file, relative to the data-set base.
                May be empty in which case data should contain pre-loaded
                data.
            * data - Pre-loaded data, may be None.
        """

        return self._view_catalogue

    @property
    def full_tree_catalogue(self) -> pd.DataFrame:
        """
        Get tree catalogue data-frame.

        :return: Returns tree catalogue data-frame with following columns:
            * tree_id - Integer identifier of the tree.
            * tree_variant_id - Integer identifier of the tree variant.
            * path - Path to the tree file, relative to the data-set base.
                May be empty in which case data should contain pre-loaded
                data.
            * json_path - Path to the description file, relative to the data-set base.
                May be empty for old or pre-loaded data.
            * data - Pre-loaded data, may be None.
        """

        return self._full_tree_catalogue

    @property
    def tree_catalogue(self) -> pd.DataFrame:
        """
        Get tree catalogue data-frame.

        :return: Returns tree catalogue data-frame with following columns:
            * tree_id - Integer identifier of the tree.
            * path - Path to the tree file, relative to the data-set base.
                May be empty in which case data should contain pre-loaded
                data.
            * data - Pre-loaded data, may be None.
        """

        return self._tree_catalogue

    @property
    def tree_data(self) -> Dict[Tuple[int, int], TreeFile]:
        """
        Get dictionary containing per-tree meta-data and node data.

        :return: Returns dictionary with key being the index of the tree
            with loaded tree file data.
        """

        return self._tree_data

    @property
    def available_features(self) -> Dict[str, List[str]]:
        """
        Get dictionary containing available features per category.

        :return: Returns dictionary with key being the feature category
            which contains lists of feature names.
        """

        return self._available_features

    @property
    def tree_ids(self) -> List[Tuple[int, int]]:
        """
        Get list of all tree identifiers in the dataset.

        :return: Returns list of all tree identifiers in the dataset.
        """

        return tuple_array_to_numpy(list(self._tree_data.keys()))

    @property
    def view_base_path(self) -> str:
        """ Get path for the root view directory. """

        return self._view_base_path

    @property
    def dataset_meta(self) -> dict:
        """ Get dataset meta-data structure. """

        return self._dataset_meta

    @property
    def dataset_version(self) -> str:
        """ Get dataset version structure. """

        return self.dataset_meta["unique_id"]


class CustomDataLoader(BaseDataLoader):
    """ Loader used for loading of custom data from hand-crafted inputs. """

    def __init__(self, data: Optional[dict] = None):
        if data is not None:
            if "tree_files" in data and "tree_views" in data and "tree_scores" in data:
                self.load_data(**data)
            elif "predictions" in data:
                self.load_predictions(**data)
            else:
                raise RuntimeError(f"Unknown data provided to the data loader \"{list(data.keys())}\"!")

    def _format_data(self, tree_files: Dict[Tuple[int, int], TreeFile],
                     tree_views: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], str, TreeImage]]],
                     tree_scores: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float, float, float, float]]]
                     ) -> dict:
        """ Pre-format data using BaseDataLoader requirements. """

        tree_ids = np.unique(np.concatenate(
            [
                tuple_array_to_numpy(list(tree_files.keys())),
                tuple_array_to_numpy(list(tree_views.keys())),
                tuple_array_to_numpy(list(tree_scores.keys()))
            ]
        ))

        tree_info = {
            tree_id: {
                "data": tree_files.get(tree_id, None),
                "view": {
                    (view_id, view_type): view_image
                    for view_id, view_type, view_image in tree_views[tree_id]
                } if (tree_id in tree_views) else None,
                "score": {
                    view_id: {
                        "jod": jod,
                        "jod_low": jod_low,
                        "jod_high": jod_high,
                        "jod_var": jod_var
                    }
                    for view_id, jod, jod_low, jod_high, jod_var in tree_scores[tree_id]
                } if (tree_id in tree_scores) else None
            }
            for tree_id in tree_ids
        }

        return tree_info

    def load_data(self, tree_files: Dict[Tuple[int, int], TreeFile] = { },
                  tree_views: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], str, TreeImage]]] = { },
                  tree_scores: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float, float, float, float]]] = { },
                  tree_comparisons: List[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[Tuple[int, int], Tuple[int, int]], int]] = [ ],
                  verbose: bool = False, node_data_loaded: bool = False):
        """
        Load user-provided data into this loader.

        :param tree_files: Pre-loaded tree files, key is the tree id, tree variant id.
        :param tree_views: Pre-loaded tree views, key is the tree id, tree variant id, values
            contain:
                view_id: int - Identifier of the view.
                view_variant_id: int - Identifier of the view variant.
                view_type: str - String identifier of the view type.
                view_image: TreeImage - Pre-loaded view image.
        :param tree_scores: Pre-loaded tree scores, key is the tree id, values
            contain:
                view_id: int - Identifier of the view, -1 for the whole tree.
                view_variant_id: int - Identifier of the view variant.
                jod, jod_low, jod_high, jod_var: float - JOD score and its stats.
        :param tree_comparisons: Choice data from pairwise comparison experiments.
            Contains, in order:
                ((first_tree_id, first_tree_variant_id), (first_view_id, first_view_variant_id)) -
                    Identifier of the first tree/view combination.
                ((second_tree_is, second_tree_variant_id), (second_view_id, second_view_variant_id)) -
                    Identifier of the second tree/view combination.
                choice - Which was chosen, 1 or 2.
        :param verbose: Print information messages?
        :param node_data_loaded: Do provided tree_files contain node data as well?
        """

        if verbose:
            self.__l.info("Loading custom data...")

        self._load_additional_data(self._format_data(
            tree_files=tree_files, tree_views=tree_views,
            tree_scores=tree_scores
        ), tree_comparisons=tree_comparisons,
            node_data_loaded=node_data_loaded)

        if verbose:
            self.__l.info("\tCustom loading finished!")

    def _format_prediction_data(self, predictions: List["Prediction"]) -> dict:
        """ Pre-format prediction data using BaseDataLoader requirements. """

        tree_files = { }
        tree_views = { }
        tree_scores = { }
        tree_comparisons = [ ]

        for prediction in predictions:
            tree_id = prediction.tree_id
            if prediction.tree_file is not None:
                tree_files[tree_id] = prediction.tree_file
            if prediction.tree_views is not None:
                tree_views = update_dict_recursively(
                    tree_views, {
                        tree_id: {
                            (view_id, view_type): view_image
                            for view_id, view_data in prediction.tree_views.items()
                            for view_type, view_image in view_data.items()
                        }
                    }, create_keys=True
                )
            if prediction.score_expected is not None:
                tree_scores = update_dict_recursively(
                    tree_scores, {
                        tree_id: {
                            view_id: view_data
                            for view_id, view_data in prediction.score_expected.items()
                        }
                    }, create_keys=True
                )

        tree_views = {
            tree_id: [
                (view_id, view_type, view_image)
                for (view_id, view_type), view_image in tree_view_data.items()
            ]
            for tree_id, tree_view_data in tree_views.items()
        }

        tree_scores = {
            tree_id: [
                (view_id, ) + (view_data["jod"], view_data["jod_low"],
                               view_data["jod_high"], view_data["jod_var"])
                for view_id, view_data in tree_score_data.items()
            ]
            for tree_id, tree_score_data in tree_scores.items()
        }

        return self._format_data(
            tree_files=tree_files,
            tree_views=tree_views,
            tree_scores=tree_scores,
        )

    def load_predictions(self, predictions: List["Prediction"]):
        """
        Load data required by given predictions.

        :param predictions: List of predictions to load the data for.
        """

        self.__l.info("Loading custom data...")

        self._load_additional_data(self._format_prediction_data(
            predictions=predictions
        ))

        self.__l.info("\tCustom loading finished!")


class DataLoader(BaseDataLoader, Configurable):
    """
    Input file loading and caching system.
    """

    COMMAND_NAME = "Data"
    """ Name of this command, used for configuration. """

    def __init__(self, config: Config):
        super().__init__(config=config)
        self._set_instance()

        self.__l.info("Initializing data loading system...")

    @classmethod
    def register_options(cls, parser: Config.Parser):
        """ Register configuration options for this class. """

        option_name = cls._add_config_parameter("experiment_result_file")
        parser.add_argument("--experiment-result-file",
                            action="store",
                            default="", type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Path to the experiment results csv file.")

        option_name = cls._add_config_parameter("experiment_tree_scores_file")
        parser.add_argument("--experiment-tree-scores-file",
                            action="store",
                            default="", type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Path to the experiment tree scores csv file.")

        option_name = cls._add_config_parameter("experiment_view_scores_file")
        parser.add_argument("--experiment-view-scores-file",
                            action="store",
                            default="", type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Path to the experiment view scores csv file.")

        option_name = cls._add_config_parameter("tree_image_path")
        parser.add_argument("--tree-image-path",
                            action="store",
                            default="", type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Path to the folder containing tree#/views "
                                 "structure. Optionally also may contain "
                                 "tree#/dither directories for view variants.")

        option_name = cls._add_config_parameter("tree_load_dither")
        parser.add_argument("--tree-load-dither",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Load view variants from tree#/dither? "
                                 "Only applicable when loading or exporting "
                                 "a dataset!")

        option_name = cls._add_config_parameter("dataset_path")
        parser.add_argument("--dataset-path",
                            action="store",
                            default="", type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Path to the pre-exported dataset. No other "
                                 "result or image data necessary.")

        option_name = cls._add_config_parameter("dataset_use_dithered")
        parser.add_argument("--dataset-use-dithered",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Use dithered view variants within the loaded dataset?")

        option_name = cls._add_config_parameter("dataset_use_augment")
        parser.add_argument("--dataset-use-augment",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Use augmented skeleton variants within the loaded dataset?")

        option_name = cls._add_config_parameter("dataset_use_augment_variants")
        parser.add_argument("--dataset-use-augment-variants",
                            action="store",
                            default=None, type=int,
                            metavar=("Count"),
                            dest=option_name,
                            help="Number of augmented skeleton variants to use. Default is all available.")

        option_name = cls._add_config_parameter("additional_path")
        parser.add_argument("--additional-path",
                            action="append",
                            default=[ ], type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Path to the additional data.")

        option_name = cls._add_config_parameter("load_node_data")
        parser.add_argument("--load-node-data",
                            action="store",
                            default=True, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Perform tree node parsing? Takes 10-20 seconds "
                                 "but may be necessary for some models.")

        option_name = cls._add_config_parameter("display_user_statistics")
        parser.add_argument("--display-user-statistics",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Display user experiment statistics?")

    def _load_additional_paths(self, additional_paths: List[Union[str, pathlib.Path]],
                               load_node_data: bool):
        """ Load additional data from given list of paths. """

        # Get information about the base dataset.
        base_path = pathlib.Path(self.view_base_path).absolute()
        max_id = np.max(self.tree_ids)[0]
        view_count = len(self.full_view_catalogue.index.unique(level="view_id"))
        view_resolution = TreeImage(image_path=f"{base_path}/{self.view_catalogue.path.iloc[0]}").data.shape[0]

        # Aggregate the additional data.
        additional_data = [ ]
        for root_path in additional_paths:
            root_path = pathlib.Path(root_path).absolute()

            score_path = root_path / "score_results.csv"
            if not score_path.is_file():
                raise RuntimeError(f"Additional path \"{root_path}\" does not contain \"score_results.csv\"!")
            score_results = pd.read_csv(score_path, sep=";").set_index("condition")

            available_trees = np.unique(
                [
                    str(path.with_suffix(""))
                    for ext in [ "tree", "png" ]
                    for path in root_path.glob(f"**/*{ext}")
                ]
            )

            for base_tree_path in available_trees:
                base_tree_path = pathlib.Path(base_tree_path)
                tree_path = base_tree_path.with_suffix(".tree")
                view_path = base_tree_path.with_suffix(".png")
                score = score_results.loc[base_tree_path.name]

                additional_data.append({
                    "name": base_tree_path.name,
                    "tree_path": tree_path if tree_path.is_file() else None,
                    "view_path": view_path if view_path.is_file() else None,
                    "jod": score["jod"], "jod_low": score["jod_low"],
                    "jod_high": score["jod_high"], "jod_var": score["var"],
                })

        # Convert the additional data into a common format.
        tree_files, tree_views, tree_scores = { }, { }, { }
        tree_paths, view_paths = { }, { }
        self.__l.info(f"Loading additional data for {len(additional_data)} trees...")
        parsing_progress = ParsingBar("", max=len(additional_data))
        for idx, additional_info in enumerate(additional_data):
            tree_id = ( max_id + idx + 1, 0 )
            score = (
                additional_info["jod"],
                additional_info["jod_low"],
                additional_info["jod_high"],
                additional_info["jod_var"],
            )
            if additional_info["tree_path"] is not None:
                tree_files[tree_id] = TreeFile(
                    file_path=additional_info["tree_path"],
                    load_node=load_node_data,
                ) if additional_info["tree_path"] is not None else None
                tree_paths[tree_id] = os.path.relpath(additional_info["tree_path"], base_path)
            else:
                tree_files[tree_id] = None
                tree_paths[tree_id] = None

            if additional_info["view_path"] is not None:
                tree_views[tree_id] = [
                    (
                        ( view_id, 0 ), "base",
                        TreeImage(
                            image_path=additional_info["view_path"]
                        ).resize(
                            resolution=view_resolution,
                            interpolation="linear",
                        ),
                    )
                    for view_id in range(view_count)
                ] if additional_info["view_path"] is not None else None
                view_paths[tree_id] = os.path.relpath(additional_info["view_path"], base_path)
            else:
                tree_views[tree_id] = None
                view_paths[tree_id] = None
            tree_scores[tree_id] = [
                ( ( view_id, 0 ), ) + score
                for view_id in range(view_count)
            ] + [ ( ( -1, 0 ), ) + score ]
            parsing_progress.next(1)
        parsing_progress.finish()
        self.__l.info(f"\tDone, loaded {len(additional_data)} additional trees.")

        # Load the additional data.
        additional_loader = CustomDataLoader(
            data={
                "tree_files": tree_files,
                "tree_views": tree_views,
                "tree_scores": tree_scores,
            },
        )

        # Add paths to both trees and views.
        for tree_id, tree_path in tree_paths.items():
            additional_loader.full_tree_catalogue.loc[tree_id].path = tree_path or ""
            additional_loader.tree_catalogue.loc[tree_id[0]].path = tree_path or ""
        for tree_id, view_path in view_paths.items():
            additional_loader.full_view_catalogue.loc[tree_id].path = view_path or ""
            additional_loader.view_catalogue.loc[tree_id[0]].path = view_path or ""

        # Add additional data to the current dataset.
        self._full_scores = pd.concat([
            self.full_scores, additional_loader.full_scores
        ])
        self._scores = pd.concat([
            self.scores, additional_loader.scores
        ])
        self._full_scores_indexed = pd.concat([
            self.full_scores_indexed, additional_loader.full_scores_indexed
        ])
        self._scores_indexed = pd.concat([
            self.scores_indexed, additional_loader.scores_indexed
        ])
        self._full_view_catalogue = pd.concat([
            self.full_view_catalogue, additional_loader.full_view_catalogue
        ])
        self._view_catalogue = pd.concat([
            self.view_catalogue, additional_loader.view_catalogue
        ])
        self._full_tree_catalogue = pd.concat([
            self.full_tree_catalogue, additional_loader.full_tree_catalogue
        ])
        self._tree_catalogue = pd.concat([
            self.tree_catalogue, additional_loader.tree_catalogue
        ])
        self._spherical_scores_indexed = pd.concat([
            self.spherical_scores_indexed, additional_loader.spherical_scores_indexed
        ])
        self._tree_data = {
            **self.tree_data, **additional_loader.tree_data
        }

    def _display_user_statistics(self):
        """ Display user experiment statistics. """

        self.__l.info("User experiment statistics: ")

        #plt.figure(figsize=[ 3.2, 4.8])
        fig, ax = plt.subplots(figsize=( 4.2, 4.0 ))
        ax = sns.histplot(ax=ax, data=self.users.loc[(self.users.gender == "F") | (self.users.gender == "M")], x="gender")
        ax.set(title="Gender", xlabel="Gender", ylabel="Count")
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        GraphSaver.save_graph("UserStatisticsGender")

        perc_male = len(self.users[self.users.gender == "M"]) / len(self.users) * 100.0
        perc_female = len(self.users[self.users.gender == "F"]) / len(self.users) * 100.0
        self.__l.info(f"Gender: Male {perc_male}% Female {perc_female}%")

        ax = sns.displot(data=self.users, x="comment")
        ax.set(title="Comment Length", xlabel="Comment Length", ylabel="Count")
        _, xlabels = plt.xticks()
        ax.set_xticklabels(xlabels, size=9)
        _, ylabels = plt.yticks()
        ax.set_yticklabels(ylabels, size=9)
        GraphSaver.save_graph("UserStatisticsComment")

        comment_length_mean = self.users["comment"].mean()
        comment_length_var = self.users["comment"].var()
        self.__l.info(f"Comment Length: {comment_length_mean} +- {comment_length_var}")

        fig, ax = plt.subplots(figsize=( 4.2, 4.0 ))
        ax = sns.histplot(ax=ax, data=self.users, x="average_choice")
        ax.set(title="Average Choice", xlabel="Average Choice", ylabel="Count")
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        GraphSaver.save_graph("UserStatisticsAverageChoice")

        fig, ax = plt.subplots(figsize=( 4.2, 4.0 ))
        ax = sns.histplot(ax=ax, data=self.users.loc[~self.users.age.str.contains("\[")], x="age")
        ax.set(title="Age Group", xlabel="Age", ylabel="Count")
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        GraphSaver.save_graph("UserStatisticsAge")

        perc_middle_aged = (len(self.users[self.users.age == "20-30"]) +
                            len(self.users[self.users.age == "31-40"])) / len(self.users) * 100.0
        self.__l.info(f"Aged between 20-40 {perc_middle_aged}%")

        fig, ax = plt.subplots(figsize=( 4.2, 4.0 ))
        ax = sns.histplot(ax=ax, data=self.users, x="education")
        ax.set(title="Education", xlabel="Education", ylabel="Count")
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        GraphSaver.save_graph("UserStatisticsEducation")

        scores = self.full_scores.reset_index().sort_values(by="tree_id")
        max_score = scores.tree_jod.max()
        #scores.tree_jod /= max_score
        #scores.tree_jod_var /= max_score * max_score
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.lineplot(data=scores, x="tree_id", y="tree_jod", ax=ax)
        ax.errorbar(x=scores.tree_id, y=scores.tree_jod, yerr=np.sqrt(scores.tree_jod_var),
                    fmt="none", c="black", capsize=2)
        ax.fill_between(x=scores.tree_id,
                        y1=scores.tree_jod - scores.tree_jod_var,
                        y2=scores.tree_jod + scores.tree_jod_var,
                        alpha=0.5)
        ax.set(title="Tree Perceptual Realism", xlabel="Tree ID", ylabel="Score [JOD]")
        #_, xlabels = plt.xticks()
        #ax.set_xticklabels(xlabels, size=9)
        #_, ylabels = plt.yticks()
        #ax.set_yticklabels(ylabels, size=9)
        GraphSaver.save_graph("UserStatisticsScore")

        scores = self.scores.reset_index().sort_values(by="tree_id").copy()
        scores["tree_jod"] = (scores["tree_jod"] - scores["tree_jod"].min()) / \
                             (scores["tree_jod"].max() - scores["tree_jod"].min())
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.histplot(data=scores, x="tree_jod", bins=9, ax=ax)
        ax.set(title="Tree Perceptual Realism", xlabel="Score", ylabel="Count")
        ax.set_yticks(list(range(0, 21, 2)))
        GraphSaver.save_graph("UserStatisticsTreeScoreHist")

        scores = self.scores_indexed.loc[self.scores_indexed.index.get_level_values(level="view_id") >= 0]\
            .reset_index().sort_values(by="tree_id").copy()
        scores["jod"] = (scores["jod"] - scores["jod"].min()) / \
                        (scores["jod"].max() - scores["jod"].min())
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.histplot(data=scores, x="jod", bins=10, ax=ax)
        ax.set(title="View Perceptual Realism", xlabel="Score", ylabel="Count")
        ax.set_yticks(list(range(0, 81, 10)))
        GraphSaver.save_graph("UserStatisticsViewScoreHist")

        self.__l.info("\tUser Statistics Reported!")

    def process(self):
        """ Perform data loading operations. """

        self.__l.info("Starting data load operations...")

        if self.c.dataset_path:
            self._load_as_dataset(
                dataset_path=self.c.dataset_path,
                use_dithered=self.c.dataset_use_dithered,
                use_augment=self.c.dataset_use_augment,
                use_augment_variants=self.c.dataset_use_augment_variants,
            )
        elif self.c.experiment_result_file and \
                self.c.experiment_tree_scores_file and \
                self.c.experiment_view_scores_file and \
                self.c.tree_image_path:
            self._load_as_raw_data(
                result_file=self.c.experiment_result_file,
                tree_scores_file=self.c.experiment_tree_scores_file,
                view_scores_file=self.c.experiment_view_scores_file,
                tree_image_path=self.c.tree_image_path,
                load_dither=self.c.tree_load_dither,
                load_node_data=self.c.load_node_data
            )
        else:
            self.__l.warning("No input data-set or full raw data sources provided!")
            self._load_empty()

        if len(self.c.additional_path) > 0:
            self._load_additional_paths(
                additional_paths=self.c.additional_path,
                load_node_data=self.c.load_node_data,
            )

        if self.c.display_user_statistics:
            self._display_user_statistics()

        self.__l.info("\tLoading operations finished!")

