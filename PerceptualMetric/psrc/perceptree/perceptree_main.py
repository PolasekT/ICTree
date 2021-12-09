#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script of the PerceptualMetric for automated tree evaluation.
"""

__author__ = 'Tomas Polasek'
__copyright__ = 'Copyright (C) 2020, Tomas Polasek'
__credits__ = ["Tomas Polasek"]
__license__ = 'MIT'
__version__ = '0.1'
__maintainer__ = 'Tomas Polasek'
__email__ = 'ipolasek@vutbr.cz'
__status__ = 'Development'

import logging
import matplotlib.pyplot
import os
import sys
from typing import List, Optional
import traceback

# Utilities:
from perceptree.common.configuration import Config
from perceptree.common.configuration import Configurable
from perceptree.common.logging import Logger
from perceptree.common.pytorch_safe import *

# Systems:
from perceptree.common.logging import LoggingConfigurator
from perceptree.common.serialization import Serializer
from perceptree.common.tensorboard import TensorBoardLogger
from perceptree.data.generator import DataGenerator
from perceptree.data.loader import DataLoader
from perceptree.data.featurizer import DataFeaturizer
from perceptree.data.exporter import DataExporter
from perceptree.data.evaluate import EvaluationProcessor
from perceptree.data.predict import PredictionProcessor

# Models:
from perceptree.model.base import BaseModel
from perceptree.model.feature_predictor import FeaturePredictor
from perceptree.model.image_predictor import ImagePredictor
from perceptree.model.combined_predictor import CombinedPredictor


class PerceptreeMain(Logger, Configurable):
    """ Wrapper around the main function. """

    COMMAND_NAME = "Options"
    """ Name of this command, used for configuration. """

    def __init__(self):
        pass

    @classmethod
    def register_options(cls, parser: Config.Parser):
        """ Register configuration options for this class. """

        option_name = cls._add_config_parameter("save_cfg_path")
        parser.add_argument("-s", "--save-cfg",
                            action="store",
                            default="", type=str,
                            metavar=("FILENAME"),
                            dest=option_name,
                            help="Save the configuration into loadable file.")

        option_name = cls._add_config_parameter("load_cfg_path")
        parser.add_argument("-l", "--load-cfg",
                            action="store",
                            default="", type=str,
                            metavar=("FILENAME"),
                            dest=option_name,
                            help="Load the configuration from saved file.")

    def initialize_libraries(self):
        """
        Perform library initialization.
        """

        self.__l.info("Initializing libraries...")

        if "DISPLAY" in os.environ and os.environ["DISPLAY"] == "False":
            matplotlib.pyplot.switch_backend("agg")

        initialize_pytorch()

    def save_config(self, config: Config, save_cfg_path: str):
        """
        Save the config to given file.

        :param config: The configuration.
        :param save_cfg_path: Path to the target file.
        """

        self.__l.info("Saving config to file \"{}\" ."
                           .format(save_cfg_path))
        config.save_cache_yaml_to_file(save_cfg_path)

    def load_config(self, config: Config, load_cfg_path: str):
        """
        Load the config from given file.

        :param config: The configuration.
        :param load_cfg_path: Path to the target file.
        """

        self.__l.info("Loading config from file \"{}\" ."
                           .format(load_cfg_path))
        config.load_cache_yaml_from_file(load_cfg_path)

    def save_load_config(self, config: Config):
        """
        Check if saving/loading of config is requested and
        perform the operation.

        :param config: The configuration.
        """

        save_cfg_path = config["options.save_cfg_path"]
        load_cfg_path = config["options.load_cfg_path"]

        if save_cfg_path:
            # Saving is requested -> save the config.
            self.save_config(config, save_cfg_path)
        elif load_cfg_path:
            # Loading is requested -> load the config.
            self.load_config(config, load_cfg_path)

    def print_banner(self):
        """ Print the application banner. """

        print("""
                                                                                                        
                                                                                                
            `7MM\"""Mq.                                       MMP""MM""YMM                       
              MM   `MM.                                      P'   MM   `7                       
              MM   ,M9 .gP"Ya `7Mb,od8 ,p6"bo   .gP"Ya `7MMpdMAo. MM  `7Mb,od8 .gP"Ya   .gP"Ya  
              MMmmdM9 ,M'   Yb  MM' "'6M'  OO  ,M'   Yb  MM   `Wb MM    MM' "',M'   Yb ,M'   Yb 
              MM      8M\"""\"""  MM    8M       8M\"""\"""  MM    M8 MM    MM    8M\"""\""" 8M\"""\""" 
              MM      YM.    ,  MM    YM.    , YM.    ,  MM   ,AP MM    MM    YM.    , YM.    , 
            .JMML.     `Mbmmd'.JMML.   YMbmd'   `Mbmmd'  MMbmmd'.JMML..JMML.   `Mbmmd'  `Mbmmd' 
                                                         MM                                     
                                                       .JMML.
            
            Perceptual tree evaluation based on machine learning models.
            
            
            Authors:    Tomas Polasek
            E-mails:    ipolasek@vutbr.cz
            Version:    0.1
            Maintainer: Tomas Polasek
            Status:     Development
            
            For further information about script parameters, please see the --help or -h parameter.
            """)

    def process_model(self, config: Config, model_name: str, model: object) -> BaseModel:
        """
        Process given model and all of its requested actions.

        :param config: Configurator.
        :param model_name: Name of the model.
        :param model: Class of the model.

        :return: Returns prepared instance of the model.
        """

        self.__l.info("Preparing model {} ...".format(model_name))

        model_instance = model(config)

        load_filename = config["model." + model_name + ".load"]
        save_filename = config["model." + model_name + ".save"]
        train = config["model." + model_name + ".train"]

        serializer = self.get_instance(Serializer)

        if load_filename:
            serializer.deserialize_model(
                config=config, instance=model_instance,
                file_path=load_filename
            )
        elif train:
            self.__l.info("\tTraining model from current data...")

            model_instance.train()

            self.__l.info("\t\tTraining done!")
        else:
            self.__l.error("\tNo fitting has been performed on this model (Missing load or train)!")

        if save_filename:
            serializer.serialize_model(
                config=config, instance=model_instance,
                file_path=save_filename
            )

        self.__l.info("\tDone!".format(model_name))

        return model_instance

    def main(self, argv: List[str]) -> int:
        """
        Main function which contains:
            * Parameter processing
            * Calling inner functions according to the parameters
            * Error reporting

        :param argv: Argument vector including the app name.

        :return: Returns success code.
        """

        # Initialize configuration.
        config = Config()

        # Register systems.
        PerceptreeMain.register_config(config)
        LoggingConfigurator.register_config(config)
        Serializer.register_config(config)
        TensorBoardLogger.register_config(config)
        DataGenerator.register_config(config)
        DataLoader.register_config(config)
        DataFeaturizer.register_config(config)
        DataExporter.register_config(config)
        EvaluationProcessor.register_config(config)
        PredictionProcessor.register_config(config)

        # Register models.
        FeaturePredictor.register_model(config)
        ImagePredictor.register_model(config)
        CombinedPredictor.register_model(config)

        config.init_options()

        # Parse arguments passed from the command line.
        argv = argv[1:]
        config.parse_args(argv)

        # Initialize configuration of this application.
        super().__init__(config=config)
        self._set_instance()

        # Check if save/load of config is requested.
        self.save_load_config(config)

        # Enable requested logging.
        logging_config = LoggingConfigurator(config)

        # Display banner if verbose.
        if logging_config.logging_level >= logging.WARNING:
            self.print_banner()

        # Initialize library and configure.
        self.initialize_libraries()

        # Initialize systems.
        try:
            serializer = Serializer(config)
            data_generator = DataGenerator(config)
            data_loader = DataLoader(config)
            data_featurizer = DataFeaturizer(config)
            data_exporter = DataExporter(config)
            prediction_processor = PredictionProcessor(config)
            evaluation_processor = EvaluationProcessor(config)
        except Exception as e:
            self.__l.error(f"Exception occurred when initializing systems! : " \
                           f"\n{e}\n{traceback.format_exc()}")
            self.__m.report_exception("Initialize Systems", e)
            return -1

        # Generate data.
        try:
            data_generator.process()
        except Exception as e:
            self.__l.error(f"Exception occurred when generating data! : " \
                           f"\n{e}\n{traceback.format_exc()}")
            self.__m.report_exception("Generate Data", e)
            return -1

        # Load data.
        try:
            data_loader.process()
        except Exception as e:
            self.__l.error(f"Exception occurred when loading data! : " \
                           f"\n{e}\n{traceback.format_exc()}")
            self.__m.report_exception("Load Data", e)
            return -1

        # Prepare TensorBoard logging.
        TensorBoardLogger.initialize(config)

        # Train models.
        try:
            requested_models = config.get_requested_models()
            models = {}
            for model_name, model in requested_models.items():
                models[model_name] = self.process_model(config, model_name, model)
            config["models"] = models
        except Exception as e:
            self.__l.error(f"Exception occurred when training! : " \
                           f"\n{e}\n{traceback.format_exc()}")
            self.__m.report_exception("Perform Training", e)
            return -1

        # Predict results.
        try:
            predictions = prediction_processor.process(models)
        except Exception as e:
            self.__l.error(f"Exception occurred when predicting! : " \
                           f"\n{e}\n{traceback.format_exc()}")
            self.__m.report_exception("Prediction Processing", e)
            return -1

        # Perform evaluation.
        try:
            evaluation_processor.process(predictions)
        except Exception as e:
            self.__l.error(f"Exception occurred when evaluating! : " \
                           f"\n{e}\n{traceback.format_exc()}")
            self.__m.report_exception("Perform Evaluations", e)
            return -1

        # Export results.
        try:
            data_featurizer.process()
            data_exporter.process()
        except Exception as e:
            self.__l.error(f"Exception occurred when exporting! : " \
                           f"\n{e}\n{traceback.format_exc()}")
            self.__m.report_exception("Export Data", e)
            return -1

        return 0


def main(argv: Optional[List[str]] = None):
    perceptree_main = PerceptreeMain()
    perceptree_main.main(sys.argv if argv is None else argv)


if __name__ == "__main__":
    main()
