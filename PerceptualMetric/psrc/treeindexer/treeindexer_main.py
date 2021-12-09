#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script of the tree indexer.
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
import sys
from typing import List
import traceback

# Utilities:
from perceptree.common.configuration import Config
from perceptree.common.configuration import Configurable
from perceptree.common.logging import Logger

# Systems:
from perceptree.common.logging import LoggingConfigurator
from perceptree.common.serialization import Serializer
from treeindexer.indexer import TreeIndexer


class TreeIndexerMain(Logger, Configurable):
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

        # http://www.network-science.de/ascii/ Roman 160
        print("""
                                                                                                        
                                                                                                
                                                                                                
            ooooooooooooo                              ooooo                   .o8                                           
            8'   888   `8                              `888'                  "888                                           
                 888      oooo d8b  .ooooo.   .ooooo.   888  ooo. .oo.    .oooo888   .ooooo.  oooo    ooo  .ooooo.  oooo d8b 
                 888      `888""8P d88' `88b d88' `88b  888  `888P"Y88b  d88' `888  d88' `88b  `88b..8P'  d88' `88b `888""8P 
                 888       888     888ooo888 888ooo888  888   888   888  888   888  888ooo888    Y888'    888ooo888  888     
                 888       888     888    .o 888    .o  888   888   888  888   888  888    .o  .o8"'88b   888    .o  888     
                o888o     d888b    `Y8bod8P' `Y8bod8P' o888o o888o o888o `Y8bod88P" `Y8bod8P' o88'   888o `Y8bod8P' d888b    
                                                                                                             
            
            Tree indexer utility.
            
            
            Authors:    Tomas Polasek
            E-mails:    ipolasek@vutbr.cz
            Version:    0.1
            Maintainer: Tomas Polasek
            Status:     Development
            
            For further information about script parameters, please see the --help or -h parameter.
            """)

    def main(self, argv: List[str]):
        """
        Main function which contains:
            * Parameter processing
            * Calling inner functions according to the parameters
            * Error reporting

        :param argv: Argument vector including the app name.
        """

        # Initialize configuration.
        config = Config()

        # Register systems.
        TreeIndexerMain.register_config(config)
        LoggingConfigurator.register_config(config)
        Serializer.register_config(config)
        TreeIndexer.register_config(config)

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
            indexer = TreeIndexer(config)
        except Exception as e:
            self.__l.error(f"Exception occurred when initializing systems! : " \
                           f"\n{e}\n{traceback.format_exc()}")
            self.__m.report_exception("Initialize Systems", e)
            return

        # Perform operations.
        try:
            indexer.process()
        except Exception as e:
            self.__l.error(f"Exception occurred when indexing! : " \
                           f"\n{e}\n{traceback.format_exc()}")
            self.__m.report_exception("Load Data", e)
            return


def main():
    tree_indexer_main = TreeIndexerMain()
    tree_indexer_main.main(sys.argv)


if __name__ == "__main__":
    main()
