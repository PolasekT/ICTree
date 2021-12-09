# -*- coding: utf-8 -*-

"""
System utilizing the PyTreeIO library to generate tree data.
"""

import json
import os
import shutil
from typing import List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import pathlib
import shutil

from perceptree.common.util import parse_bool_string
from perceptree.common.configuration import Config
from perceptree.common.configuration import Configurable
from perceptree.common.logger import ProcessingBar
from perceptree.common.logger import Logger

from perceptree.data.treeio import TreeFile
from perceptree.data.treeio import TreeImage

from perceptree.common.native import C_TREEIO_AVAILABLE, treeio


class DataGenerator(Logger, Configurable):
    """
    System utilizing the PyTreeIO library to generate tree data.
    """

    COMMAND_NAME = "Generate"
    """ Name of this command, used for configuration. """

    def __init__(self, config: Config):
        super().__init__(config=config)
        self._set_instance()

        self.__l.info("Initializing data generator system...")

    @classmethod
    def register_options(cls, parser: Config.Parser):
        """ Register configuration options for this class. """

        option_name = cls._add_config_parameter("output_path")
        parser.add_argument("--output-path",
                            action="store",
                            default="", type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Base path to store the outputs into.")

        option_name = cls._add_config_parameter("input_path")
        parser.add_argument("--input-path",
                            action="append",
                            default=[ ], type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Load all trees from given input path. Supported "
                                 "formats include \".tree\" and \".fbx\".")

        option_name = cls._add_config_parameter("input_tree")
        parser.add_argument("--input-tree",
                            action="append",
                            default=[ ], type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Load a single tree for processing. Supported "
                                 "formats include \".tree\" and \".fbx\".")

        option_name = cls._add_config_parameter("do_feature")
        parser.add_argument("--do-feature",
                            action="store_true",
                            default=False,
                            dest=option_name,
                            help="Specify this flag to enable feature calculation.")

        option_name = cls._add_config_parameter("do_render")
        parser.add_argument("--do-render",
                            action="store_true",
                            default=False,
                            dest=option_name,
                            help="Specify this flag to enable view rendering.")

        option_name = cls._add_config_parameter("render_resolution")
        parser.add_argument("--render-resolution",
                            action="store",
                            default="1024x1024", type=str,
                            metavar=("<SIZE>|<WIDTH>x<HEIGHT>"),
                            dest=option_name,
                            help="Resolution of the rendered views. Either "
                                 "a single number (e.g., 1024), or width and "
                                 "height (e.g., 1024x1024).")

        option_name = cls._add_config_parameter("render_samples")
        parser.add_argument("--render-samples",
                            action="store",
                            default=4, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Number of samples to render for each pixel.")

        option_name = cls._add_config_parameter("render_views")
        parser.add_argument("--render-views",
                            action="store",
                            default=5, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Number of uniformly spaced views to render.")

        option_name = cls._add_config_parameter("render_camera_distance")
        parser.add_argument("--render-camera-distance",
                            action="store",
                            default=5.0, type=float,
                            metavar=("DISTANCE"),
                            dest=option_name,
                            help="Distance of the rendering camera from the origin.")

        option_name = cls._add_config_parameter("render_camera_height")
        parser.add_argument("--render-camera-height",
                            action="store",
                            default=5.0, type=float,
                            metavar=("HEIGHT"),
                            dest=option_name,
                            help="Height of the rendering camera from the ground plane.")

        option_name = cls._add_config_parameter("render_tree_normalize")
        parser.add_argument("--render-tree-normalize",
                            action="store",
                            default=True, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Normalize the tree before rendering?")

        option_name = cls._add_config_parameter("render_tree_scale")
        parser.add_argument("--render-tree-scale",
                            action="store",
                            default=1.0, type=float,
                            metavar=("SCALE"),
                            dest=option_name,
                            help="Scale the normalized tree to this size. Only works when "
                                 "the tree is normalized!")

        option_name = cls._add_config_parameter("do_augment")
        parser.add_argument("--do-augment",
                            action="store_true",
                            default=False,
                            dest=option_name,
                            help="Specify this flag to enable tree augmentation.")

        option_name = cls._add_config_parameter("augment_variants")
        parser.add_argument("--augment-variants",
                            action="store",
                            default=16, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Number of augmented skeleton variants to generate.")

        option_name = cls._add_config_parameter("augment_seed")
        parser.add_argument("--augment-seed",
                            action="store",
                            default=0, type=int,
                            metavar=("SEED"),
                            dest=option_name,
                            help="Numerical seed used for tree augmentation.")

        option_name = cls._add_config_parameter("augment_distribution")
        parser.add_argument("--augment-distribution",
                            action="store",
                            default="uniform", type=str,
                            metavar=("uniform|normal"),
                            dest=option_name,
                            help="Distribution function used for randomized values in "
                                 "tree augmentation.")

        option_name = cls._add_config_parameter("augment_node")
        parser.add_argument("--augment-node",
                            action="store",
                            default="-1.0;1.0;0.0", type=str,
                            metavar=("LOW;HIGH;STRENGTH"),
                            dest=option_name,
                            help="Specification of node tree augmentation as percentage "
                                 "of the base tree size.")

        option_name = cls._add_config_parameter("augment_branch")
        parser.add_argument("--augment-branch",
                            action="store",
                            default="-1.0;1.0;0.0", type=str,
                            metavar=("LOW;HIGH;STRENGTH"),
                            dest=option_name,
                            help="Specification of branch tree augmentation as percentage "
                                 "of the base tree size.")

        option_name = cls._add_config_parameter("augment_branch_skip_leaves")
        parser.add_argument("--augment-branch-skip-leaves",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Skip leaf nodes for the branch tree augmentation?")

        option_name = cls._add_config_parameter("do_dither")
        parser.add_argument("--do-dither",
                            action="store_true",
                            default=False,
                            dest=option_name,
                            help="Specify this flag to enable tree view dithering. Rendering "
                                 "options are shared.")

        option_name = cls._add_config_parameter("dither_variants")
        parser.add_argument("--dither-variants",
                            action="store",
                            default=16, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Number of dithered view variants to generate.")

        option_name = cls._add_config_parameter("dither_seed")
        parser.add_argument("--dither-seed",
                            action="store",
                            default=0, type=int,
                            metavar=("SEED"),
                            dest=option_name,
                            help="Numerical seed used for view dithering.")

        option_name = cls._add_config_parameter("dither_distance")
        parser.add_argument("--dither-distance",
                            action="store",
                            default=0.5, type=float,
                            metavar=("DISTANCE"),
                            dest=option_name,
                            help="Range of dithering with respect to the camera distance.")

        option_name = cls._add_config_parameter("dither_yaw")
        parser.add_argument("--dither-yaw",
                            action="store",
                            default="-1.0;1.0;65.0", type=str,
                            metavar=("LOW;HIGH;STRENGTH"),
                            dest=option_name,
                            help="Range of dithering with respect to the camera yaw.")

        option_name = cls._add_config_parameter("dither_pitch")
        parser.add_argument("--dither-pitch",
                            action="store",
                            default="-1.0;1.0;25.0", type=str,
                            metavar=("LOW;HIGH;STRENGTH"),
                            dest=option_name,
                            help="Range of dithering with respect to the camera pitch.")

        option_name = cls._add_config_parameter("dither_roll")
        parser.add_argument("--dither-roll",
                            action="store",
                            default="-1.0;1.0;10.0", type=str,
                            metavar=("LOW;HIGH;STRENGTH"),
                            dest=option_name,
                            help="Range of dithering with respect to the camera roll.")

    def _process_trees(self, output_path: str, input_paths: List[str], input_trees: List[str]):
        """
        Process input trees, storing the outputs into the provided output path.

        :param output_path: Directory to output the results to.
        :param input_paths: List of input paths to scour for .tree files.
        :param input_trees: List of input tree files.
        """

        # Prepare the output path and a list of inputs.
        output_path = pathlib.Path(output_path).absolute()
        input_list = [
            ( input_tree, input_tree.parent.relative_to(input_path) )
            for input_path in [ pathlib.Path(p).absolute() for p in input_paths ]
            for input_tree in [
                pathlib.Path(p).absolute()
                for ext in [ "tree", "fbx" ]
                for p in pathlib.Path(input_path).glob(f"**/*.{ext}")
            ]
        ] + [
            ( pathlib.Path(input_tree).absolute(), pathlib.Path("./") )
            for input_tree in input_trees
        ]

        self.__l.info(f"Generator processing {len(input_list)} input tree files to \"{output_path}\"...")

        # Disable verbose logging.
        treeio.TreeLogger.set_logging_level(treeio.LoggingLevel.Error)
        # Setup the statistics calculator, renderer, and augmenter.
        stats = treeio.TreeStats()
        render = treeio.TreeRenderer()
        augment = treeio.TreeAugmenter()

        # Prepare featurization parameters.
        do_feature = self.c.do_feature

        # Prepare rendering parameters.
        do_render = self.c.do_render
        if "x" in self.c.render_resolution.lower():
            width, height = map(int, self.c.render_resolution.lower().split("x"))
        else:
            width = height = int(self.c.render_resolution)
        samples = self.c.render_samples
        views = self.c.render_views
        camera_distance = self.c.render_camera_distance
        camera_height = self.c.render_camera_height
        tree_normalize = self.c.render_tree_normalize
        tree_scale = self.c.render_tree_scale

        # Prepare augmentation parameters.
        do_augment = self.c.do_augment
        augment_variants = self.c.augment_variants
        augment_seed = self.c.augment_seed
        augment_rng = np.random.default_rng(augment_seed)
        augment_normal = self.c.augment_distribution.lower() == "normal"
        augment_node = tuple(map(float, self.c.augment_node.split(";")))
        augment_branch = tuple(map(float, self.c.augment_branch.split(";")))
        augment_branch_skip_leaves = self.c.augment_branch_skip_leaves

        # Prepare view dithering parameters.
        do_dither = self.c.do_dither
        dither_variants = self.c.dither_variants
        dither_seed = self.c.dither_seed
        dither_distance = self.c.dither_distance
        dither_yaw = tuple(map(float, self.c.dither_yaw.split(";")))
        dither_pitch = tuple(map(float, self.c.dither_pitch.split(";")))
        dither_roll = tuple(map(float, self.c.dither_roll.split(";")))

        # Process input files.
        progress = ProcessingBar("", max=len(input_list))
        for ( input_file, input_relative ) in input_list:
            # Prepare name identifier.
            base_name = input_file.with_suffix("").name

            # Prepare the output directory structure.
            base_output = output_path / input_relative
            if base_output.name != base_name:
                base_output /= base_name
            tree_output = base_output
            tree_output.mkdir(parents=True, exist_ok=True)
            view_output = base_output / "views"
            view_output.mkdir(parents=True, exist_ok=True)
            augment_output = base_output / "skeleton"
            augment_output.mkdir(parents=True, exist_ok=True)
            dither_output = base_output / "dither"
            dither_output.mkdir(parents=True, exist_ok=True)

            # Load the tree.
            tree = treeio.ArrayTree(path=str(input_file))

            # Calculate its features.
            if do_feature:
                stats.calculate_statistics(tree=tree)
                stats.save_statistics(tree=tree)

            # Save the tree with features calculated, along with the original.
            shutil.copyfile(input_file, tree_output / f"{input_file.name}.orig")
            tree.to_path(tree_output / f"{base_name}.tree")

            # Render the views and dithered variants.
            if do_render or do_dither:
                render.render_dither_tree(
                    tree=tree, output_path=view_output, base_name=base_name,
                    width=width, height=height, samples=samples, view_count=views,
                    camera_distance=camera_distance, camera_height=camera_height,
                    tree_normalize=tree_normalize, tree_scale=tree_scale,
                    dither_output_path=dither_output, dither_seed=dither_seed,
                    dither_count=dither_variants if do_dither else 0,
                    cam_distance_var=dither_distance,
                    cam_yaw_dither=dither_yaw[0],
                    cam_yaw_dither_low=dither_yaw[1], cam_yaw_dither_high=dither_yaw[2],
                    cam_pitch_dither=dither_pitch[0],
                    cam_pitch_dither_low=dither_pitch[1], cam_pitch_dither_high=dither_pitch[2],
                    cam_roll_dither=dither_roll[0],
                    cam_roll_dither_low=dither_roll[1], cam_roll_dither_high=dither_roll[2],
                )

            # Augment the skeleton.
            if do_augment:
                for variant_id in range(augment_variants):
                    variant_seed = augment_rng.integers(np.iinfo(np.int32).max)
                    tree_variant = treeio.ArrayTree.create_copy(tree)
                    augment.augment_tree(
                        tree=tree_variant, seed=variant_seed, normal=augment_normal,
                        n_low=augment_node[0], n_high=augment_node[1], n_strength=augment_node[2],
                        b_low=augment_branch[0], b_high=augment_branch[1], b_strength=augment_branch[2],
                        skip_leaves=augment_branch_skip_leaves,
                    )
                    tree_variant.to_path(augment_output / f"{base_name}_variant_{variant_id}.tree")

            progress.next(1)

        progress.finish()

        self.__l.info(f"\tGenerating completed!")

    def process(self):
        """ Perform data export operations. """

        self.__l.info("Starting data generating operations...")

        if C_TREEIO_AVAILABLE:
            self.__l.warning("\tPyTreeIO library IS available!")
            if len(self.c.input_path) > 0 or len(self.c.input_tree) > 0:
                self._process_trees(
                    output_path=self.c.output_path,
                    input_paths=self.c.input_path,
                    input_trees=self.c.input_tree,
                )
        else:
            self.__l.warning("\tPyTreeIO library is NOT available, skipping!")

        self.__l.info("\tGenerating operations finished!")


