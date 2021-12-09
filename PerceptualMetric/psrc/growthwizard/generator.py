# -*- coding: utf-8 -*-

"""
Growth parameter file generator.
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
from perceptree.common.logger import Logger
from perceptree.common.util import parse_bool_string

from growthwizard.parameter import GrowthParameters


class GrowthModelGenerator(Logger, Configurable):
    """
    Generator of growth models
    """

    COMMAND_NAME = "Generator"
    """ Name of this command, used for configuration. """

    def __init__(self, config: Config):
        super().__init__(config=config)
        self._set_instance()

        self.__l.info("Initializing growth model generator...")

    @classmethod
    def register_options(cls, parser: Config.Parser):
        """ Register configuration options for this class. """

        option_name = cls._add_config_parameter("input_files")
        parser.add_argument("-i", "--input-files",
                            action="append",
                            default=[ ], type=str,
                            metavar=("GROWTH_FILE.XML/DIRECTORY"),
                            dest=option_name,
                            help="Provide input parameter files to work on. Provide "
                                 "a directory to include all of the .xml files.")

        option_name = cls._add_config_parameter("sample_around")
        parser.add_argument("-s", "--sample-around",
                            action="store_true",
                            default=False,
                            dest=option_name,
                            help="Sample the parameter space around provided growth model "
                                 "parameters.")

        option_name = cls._add_config_parameter("sample_around_min_delta")
        parser.add_argument("--sample-around-min-delta",
                            action="store",
                            default=True, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Use min_delta for sampling around?")

        option_name = cls._add_config_parameter("sample_around_max_delta")
        parser.add_argument("--sample-around-max-delta",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Use max_delta for sampling around?")

        option_name = cls._add_config_parameter("sample_between")
        parser.add_argument("-b", "--sample-between",
                            action="store_true",
                            default=False,
                            dest=option_name,
                            help="Sample the parameter space between pairs of base model "
                                 "parameters.")

        option_name = cls._add_config_parameter("sample_between_samples")
        parser.add_argument("--sample-between-samples",
                            action="store",
                            default=1024, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Create given number of samples between each model pair.")

        option_name = cls._add_config_parameter("parameter_template")
        parser.add_argument("-t", "--parameter-template",
                            action="store",
                            default="", type=str,
                            metavar=("PATH.JSON"),
                            dest=option_name,
                            help="Provide parameter template used for setting the growth "
                                 "parameter values. Set to non-existent path to generate "
                                 "a default template.")

        option_name = cls._add_config_parameter("output_dir")
        parser.add_argument("-o", "--output-dir",
                            action="store",
                            default="./", type=str,
                            metavar=("DIRECTORY"),
                            dest=option_name,
                            help="Directory to place the outputs to.")

        option_name = cls._add_config_parameter("output_parameters")
        parser.add_argument("-p", "--output-parameters",
                            action="store",
                            default="Seed,Age,LateralBudPerNode,VarianceApicalAngle,BranchingAngleMean,"
                                    "BranchingAngleVariance,RollAngleMean,RollAngleVariance,ApicalBudKillProbability,"
                                    "LateralBudKillProbability,ApicalDominanceBase,ApicalDominanceDistanceFactor,"
                                    "ApicalDominanceAgeFactor,GrowthRate,InternodeLengthBase,InternodeLengthAgeFactor,"
                                    "ApicalControlBase,ApicalControlAgeFactor,ApicalControlLevelFactor,MaxBudAge,"
                                    "InternodeSize,Phototropism,GravitropismBase,GravitropismLevelFactor,PruningFactor,"
                                    "LowBranchPruningFactor,ThicknessRemovalFactor,GravityBendingStrength,"
                                    "GravityBendingAngleFactor,ApicalBudLightingFactor,LateralBudLightingFactor,"
                                    "EndNodeThickness,ThicknessControlFactor,CrownShynessBase,CrownShynessFactor,"
                                    "FoliageType", type=str,
                            metavar=("PARAM1,PARAM2,..."),
                            dest=option_name,
                            help="List of parameters to use.")

    def _get_flat_file_list(self, input_list: List[str], ext: str) -> List[str]:
        """ Get flattened file list, automatically crawling through directories. """

        flat_file_list = [ ]

        for input_path in input_list:
            path = pathlib.Path(input_path)
            if path.is_file():
                flat_file_list.append(str(path))
            elif path.is_dir():
                for file_path in path.glob(f"**/*.{ext}"):
                    flat_file_list.append(str(file_path))

        return flat_file_list

    def _load_input_files(self, input_list: List[str]) -> List[GrowthParameters]:
        """ Load all of the input files and return a lis of parameters. """

        input_files = self._get_flat_file_list(input_list=input_list, ext="xml")
        return [ GrowthParameters(file_path=file_path) for file_path in input_files ]

    def _generate_common_parameter_df(self,
                                      input_parameters: List[GrowthParameters]) -> pd.DataFrame:
        """ Generate dataframe containing all of the common growth parameters. """

        return pd.DataFrame(
            data=[ p.parameters for p in input_parameters ],
            index=[ p.name for p in input_parameters ]
        ).dropna(axis=1)

    def _generate_load_parameter_template(self,
                                          input_parameters: List[GrowthParameters],
                                          parameter_template: str) -> dict:
        """ Generate a new or load existing parameter template. """

        template = { }
        if pathlib.Path(parameter_template).exists():
            # Template already exists -> Use it.
            try:
                with open(parameter_template, "r") as f:
                    template = json.load(f)
            except:
                self.__l.warning(f"Failed to load existing parameter template "
                                 f"from \"{parameter_template}\", using default!")

        if len(template) == 0:
            # Generate a new template using input parameters.
            parameter_df = self._generate_common_parameter_df(
                input_parameters=input_parameters)

            for parameter_type in parameter_df.columns:
                enabled = True
                diff = np.diff(np.sort(np.unique(parameter_df[parameter_type])))

                if len(diff) == 0:
                    enabled = False
                    diff = [ parameter_df.dtypes[parameter_type].type(0) ]

                min_val = parameter_df[parameter_type].min().item()
                max_val = parameter_df[parameter_type].max().item()
                min_delta = np.min(diff).item()
                max_delta = np.max(diff).item()

                sample_count = int((max_val - min_val) / min_delta) if min_delta else 0

                template[parameter_type] = {
                    "enabled": enabled,
                    "min": min_val,
                    "max": max_val,
                    "min_delta": min_delta,
                    "max_delta": max_delta,
                    "samples": sample_count
                }

            # Save template for later use.
            with open(parameter_template, "w") as f:
                json.dump(template, f, indent=4)

        return template

    def _recursive_for_parameters(self, parameter_template: dict, range_fun: Callable, fun: Callable):
        """ Perform recursive for cycle over all parameters. """

        parameter_list = list(parameter_template.keys())
        range_limits = [
            range_fun(parameter_template[parameter_name], parameter_name)
            for parameter_name in parameter_list
        ]
        total_combinations = np.prod([ len(r) for r in range_limits ])

        for parameter_tuple in itertools.product(*range_limits):
            parameter_pack = {
                parameter_name: parameter_tuple[parameter_idx]
                for parameter_idx, parameter_name in enumerate(parameter_list)
            }
            fun(parameter_pack)

    def _sample_around_parameters(self, parameter_template: dict,
                                  base_parameters: dict,
                                  use_min_delta: bool,
                                  use_max_delta: bool) -> List[dict]:
        """ Sample parameter space around given base parameter set. """

        parameters = [ ]

        def range_fun(parameter_spec: dict, parameter_name: str):
            base_value = base_parameters[parameter_name]
            if parameter_spec["enabled"]:
                values = [ ]
                if use_min_delta:
                    values += [ base_value - parameter_spec["min_delta"], base_value + parameter_spec["min_delta"] ]
                if use_max_delta:
                    values += [ base_value - parameter_spec["max_delta"], base_value + parameter_spec["max_delta"] ]
                return values
            else:
                return [ base_parameters[parameter_name] ]

        def fun(current_parameters: dict):
            parameters.append(current_parameters)

        self._recursive_for_parameters(
            parameter_template=parameter_template,
            range_fun=range_fun, fun=fun,
        )

        return parameters

    def _sample_around_files(self, parameter_template: dict,
                             input_parameters: List[GrowthParameters],
                             output_dir: str):
        """ Sample parameter space around provided files and generate results. """

        for base_model in input_parameters:
            parameters = self._sample_around_parameters(
                parameter_template=parameter_template,
                base_parameters=base_model.parameters,
                use_min_delta=self.c.sample_around_min_delta,
                use_max_delta=self.c.sample_around_max_delta
            )
            min_max_spec = ("min" if self.c.sample_around_min_delta else "") + \
                           ("max" if self.c.sample_around_max_delta else "" )
            min_max_spec = f"{min_max_spec}_" if min_max_spec else ""
            parameters_df = pd.DataFrame(data=parameters)
            parameters_df.to_csv(f"{output_dir}/{base_model.name}"
                                 f"_sample_around_{min_max_spec}{len(parameters_df)}.csv",
                                 sep=",", index=False)

    def _sample_between_parameters(self, parameter_template: dict,
                                   base_parameters1: dict,
                                   base_parameters2: dict,
                                   sample_count: int) -> List[dict]:
        """ Sample parameter space between given base parameter sets. """

        parameter_names = list(parameter_template.keys())
        parameter_types = [ type(base_parameters1[parameter_name]) for parameter_name in parameter_names ]

        t = np.linspace(0.0, 1.0, sample_count).reshape((-1, 1))
        parameters1 = np.array([ base_parameters1[parameter_name] for parameter_name in parameter_names ])
        parameters2 = np.array([ base_parameters2[parameter_name] for parameter_name in parameter_names ])

        interpolated = np.multiply(parameters2, t) + np.multiply(parameters1, 1.0 - t)

        parameters = [
            {
                parameter_name: parameter_type(parameter_value)
                for parameter_name, parameter_type, parameter_value in
                    zip(parameter_names, parameter_types, parameter_vector)
            }
            for parameter_vector in interpolated
        ]

        return parameters

    def _sample_between_files(self, parameter_template: dict,
                              input_parameters: List[GrowthParameters],
                              output_dir: str):
        """ Sample parameter space between provided files and generate results. """

        for base_model1, base_model2 in itertools.combinations(input_parameters, 2):
            parameters = self._sample_between_parameters(
                parameter_template=parameter_template,
                base_parameters1=base_model1.parameters,
                base_parameters2=base_model2.parameters,
                sample_count=self.c.sample_between_samples
            )
            parameters_df = pd.DataFrame(data=parameters)
            parameters_df.to_csv(f"{output_dir}/{base_model1.name}_{base_model2.name}"
                                 f"_sample_between_{len(parameters_df)}.csv",
                                 sep=",", index=False)

    def process(self):
        """ Perform data export operations. """

        self.__l.info("Starting growth model generation operations...")

        if self.c.input_files:
            input_parameters = self._load_input_files(
                input_list=self.c.input_files
            )
        else:
            input_parameters = [ ]

        if self.c.parameter_template:
            parameter_template = self._generate_load_parameter_template(
                input_parameters=input_parameters,
                parameter_template=self.c.parameter_template
            )
        else:
            parameter_template = { }

        if self.c.sample_around:
            self._sample_around_files(
                parameter_template=parameter_template,
                input_parameters=input_parameters,
                output_dir=self.c.output_dir
            )

        if self.c.sample_between:
            self._sample_between_files(
                parameter_template=parameter_template,
                input_parameters=input_parameters,
                output_dir=self.c.output_dir
            )

        self.__l.info("\tGrowth model generation operations finished!")

