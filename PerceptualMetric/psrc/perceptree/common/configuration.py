# -*- coding: utf-8 -*-

"""
Configuration loading and processing.
"""

import argparse
import datetime
import logging
import pickle as pk
from enum import Enum
from io import StringIO
from typing import Any, Dict, Union, TextIO, TypeVar, List, Optional

from ruamel.yaml import YAML

from perceptree.common.logger import Logger
from perceptree.common.cache import Cache


def split_with_keywords(to_split: [str], keywords: [str]) -> [[str]]:
    """
    Split given list of strings into sub-lists which
    contain only one keyword each.

    :param to_split: Vector of strings to split.
    :param keywords: Keywords to split the list by.

    :return: Returns list of list of lists split by keywords.
    """

    result = []
    used_keywords = []
    last_string_idx = 0

    for idx, val in enumerate(to_split):
        if val not in keywords or val in used_keywords:
            continue

        if last_string_idx != idx:
            result.append(to_split[last_string_idx:idx])

        last_string_idx = idx
        used_keywords.append(val)

    result.append(to_split[last_string_idx:])

    return result


class Config(Logger, Cache):
    """
    Container for application configuration. Default options
    are automatically configured.
    """

    DEFAULT_OPTIONS = {}
    """ Default option values. """

    Parser = argparse.ArgumentParser
    """ Shortcut for the argument parser. """

    def __init__(self):
        super().__init__()

        self.parser = argparse.ArgumentParser()
        self.sub_commands = self.parser.add_subparsers()
        self.sub_commands_specified = []
        self.models = {}

        self._start_time = datetime.datetime.now()
        self._runtime_arguments = "BEFORE_INITIALIZATION"

    @property
    def start_time(self):
        """ Get time of starting the application. """
        return self._start_time

    @property
    def runtime_arguments(self):
        """ Get command line arguments for the current runtime. """
        return self._runtime_arguments

    def init_options(self):
        """
        Initialize all options to default values.
        """
        self.cache = Config.DEFAULT_OPTIONS.copy()

        # Include arguments from the main parser.
        sub_commands = {"main": self.parser}
        sub_commands.update(self.sub_commands.choices)

        # Go through all commands and initialize config options.
        for name, sub in sub_commands.items():
            for action in sub._actions:
                # Skip options with no default.
                if action.default is None or \
                        action.default == argparse.SUPPRESS:
                    continue

                self.set_path(action.dest, action.default, create=True)

    def add_subcommand(self, name: str) -> argparse.ArgumentParser:
        """
        Add new sub-command for argument parsing.

        :param name: Name of the sub-command.

        :return: Returns parser for the sub-command.
        """

        return self.sub_commands.add_parser(name)

    def add_model(self, name: str, cls: object) -> argparse.ArgumentParser:
        """
        Add new model and corresponding sub-command for
        argument parsing.

        :param name: Name of the sub-command.
        :param cls: Model class.

        :return: Returns parser for the sub-command.
        """

        if name in self.models:
            raise ValueError("Provided model name is already registered!")

        self.models[name] = cls
        parser = self.add_subcommand(name)

        parser.add_argument("-s", "--save",
                            action="store",
                            default="", type=str,
                            metavar=("FILENAME"),
                            dest="model." + name + ".save",
                            help="Save the trained model to given file.")

        prep_group = parser.add_mutually_exclusive_group(required=True)
        prep_group.add_argument("-l", "--load",
                                action="store",
                                default="", type=str,
                                metavar=("FILENAME"),
                                dest="model." + name + ".load",
                                help="Load the trained model from given file.")

        prep_group.add_argument("-t", "--train",
                                action="store_true",
                                default=False,
                                dest="model." + name + ".train",
                                help="Specify to train the model on current data.")

        return parser

    def get_requested_models(self) -> dict:
        """
        Get list of models which were specified on the
        command line.

        :return: Returns dictionary containing pairs
            of model name and its class.
        """

        return {model_name: self.models[model_name]
                for model_name in self.sub_commands_specified
                if model_name in self.models}

    def get_arg_parser(self) -> argparse.ArgumentParser:
        """
        Access the main argument parser, which can be used
        to add more arguments.

        :return: Returns the main argument parser.
        """

        return self.parser

    def parse_args(self, argv: [str]):
        """
        Parse command line arguments and fill corresponding
        options.

        :param argv: Vector of command line arguments.
        """

        self._runtime_arguments = argv

        # Split command line into sub-command lines.
        subcommand_argvs = self._split_subcommand_argvs(argv)
        parsed = []

        # Parse each sub-command line.
        for subcommand_argv in subcommand_argvs:
            subcommand = subcommand_argv[0] if subcommand_argv else None
            parser = self.parser

            if subcommand in self.sub_commands.choices:
                self.sub_commands_specified.append(subcommand)
                parser = self.sub_commands.choices[subcommand]
                subcommand_argv = subcommand_argv[1:]

            parsed.append(parser.parse_args(subcommand_argv))

        # Set corresponding config options.
        for namespace in parsed:
            for var, val in vars(namespace).items():
                self.set_path(var, val, create=True)

    def subcommand_arguments(self, subcommand: str,
                             argv: Optional[List[str]] = None) -> List[str]:
        """
        Get list of arguments for given subcommand.

        :param subcommand: Subcommand name to get.
        :param argv: Optional argument vector to use. Set to None
            to use current runtime arguments.

        :return: Returns list of arguments for given subcommand.
        """

        subcommand_argvs = {
            commands[0]: commands[1:]
            for commands in self._split_subcommand_argvs(argv or self._runtime_arguments)
            if commands
        }

        subcommand_argvs.get(subcommand, [ ])

    def subcommand_arguments_equal(self, argv1: [str], argv2: [str],
                                   subcommand: Optional[str]):
        """
        Compare given argument vectors and return whether they have
        the same options for given sub-command.

        :param argv1: First argument vector being compared.
        :param argv2: Second argument vector being compared.
        :param subcommand: Subcommand to check. When None, all arguments
            are checked.

        :return: Returns True if both argument vectors are the same.
        """

        if subcommand is None:
            return argv1 == argv2

        subcommand_argvs1 = {
            commands[0]: commands[1:]
            for commands in self._split_subcommand_argvs(argv1)
            if commands
        }
        subcommand_argvs2 = {
            commands[0]: commands[1:]
            for commands in self._split_subcommand_argvs(argv2)
            if commands
        }

        if subcommand not in subcommand_argvs1 or \
           subcommand not in subcommand_argvs2:
            return False

        return subcommand_argvs1[subcommand] == subcommand_argvs2[subcommand]

    def _split_subcommand_argvs(self, argv: [str]) -> [[str]]:
        """
        Split given argument vector into sub-vectors which
        contain only one sub-command each.

        :param argv: Vector of command line arguments.

        :return: Returns list of command line argument vectors.
        """

        sub_commands = self.sub_commands.choices.keys()
        sub_command_names = [sub for sub in sub_commands]

        return split_with_keywords(argv, sub_command_names)

    T = TypeVar("T")

    def get_instance(self, cls: T) -> T:
        """ Get the main instance of given class. """
        if not hasattr(cls, f"COMMAND_PATH"):
            raise RuntimeError(f"Unable to get instance of unregistered class "
                               f"{cls.__name__}, did you forget to register_config()?")

        return self.__getitem__(cls.COMMAND_PATH + ".instance")


class ConfigTemplate:
    """
    Helper class used for wrapping configuration parameters.
    """

    def __init__(self):
        self.managed_parameters = {}

        self.clear_parameters()

    def copy(self) -> "ConfigTemplate":
        """
        Create a copy of this config.

        :return: Returns the new copy.
        """

        result = ConfigTemplate()
        result.managed_parameters = self.managed_parameters.copy()

        return result

    def clear_parameter_values(self):
        """
        Clear only parameter values, not the list
        of managed parameters. All parameters will
        be set to None.
        """

        for param_name in self.managed_parameters:
            self.managed_parameters[param_name] = None

    def clear_parameters(self):
        """
        Clear all managed parameters and their values.
        """

        self.managed_parameters = {}

    def add_parameter(self,
                      param_name: str):
        """
        Add a new managed parameter to this config.

        :param param_name: Name of the parameter.
        :raises AttributeError: Raised when parameter
          with given name already exists.
        """

        if param_name in self.managed_parameters:
            raise AttributeError("Given parameter already exists!")

        self.managed_parameters[param_name] = None

    def set_parameters_from_config(self,
                                   config: Config,
                                   var_getter: Optional[object] = None):
        """
        Get parameter values from given config.

        :param config: Config to get the values from.
        :param var_getter: Optional object with var_path
          method, which returns path to variable within
          the config.
        """

        for param_name in self.managed_parameters:
            config_name = param_name
            if var_getter is not None:
                config_name = var_getter.var_path(param_name)

            try:
                param_value = config.get_path(
                    config_name,
                    create=False,
                    none_when_missing=False
                )
                self.managed_parameters[param_name] = param_value
            except KeyError:
                # Missing parameter -> Do nothing!
                pass

    def serialize(self) -> bytes:
        """
        Serialize all of the parameters.

        :return: Returns string representing the
          parameters.
        """

        return pk.dumps(self.managed_parameters)

    def deserialize(self, serialized: bytes):
        """
        Deserialize all of the parameters.

        :param serialized: Serialized string containing
          the parameters.
        """

        self.managed_parameters = pk.loads(serialized)

    def __getattr__(self,
                    param_name: str):
        """
        Lookup parameter within this config.

        :param param_name:

        :raises AttributeError: Raised when parameter
          with given name does not exist.

        :return: Value of the parameter.
        """

        if param_name not in self.managed_parameters:
            raise AttributeError("Given parameter ({}) does NOT exists!".format(param_name))

        return self.managed_parameters[param_name]


class ConfigurableMeta(type):
    """
    Meta-class which generates makes class configurable.

    Inspired by: https://stackoverflow.com/a/50731615 .
    """

    def __init__(cls, *args):
        super().__init__(*args)

        if not hasattr(cls, "COMMAND_NAME"):
            cls.COMMAND_NAME = cls.__name__

        # Explicit name mangling...
        # Set default configuration template.
        setattr(cls, f"_{cls.__name__}__ct", ConfigTemplate())


class Configurable(object):
    """
    Inheritable helper, which allows any class to
    become configurable.
    """

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        self.c = self._get_class_config_template().copy()
        self.c.set_parameters_from_config(self.config, self)

    @classmethod
    def register_options(cls, parser: Config.Parser):
        """ Dummy version of register options which should be overriden. """
        pass

    @classmethod
    def _add_config_parameter(cls, var_name: str) -> str:
        """ Add given option to the configuration template and return full name. """
        ct = cls._get_class_config_template()
        ct.add_parameter(var_name)
        return cls.var_path_name(cls.COMMAND_PATH, var_name)

    @classmethod
    def _get_class_config_template(cls) -> ConfigTemplate:
        """ Get configuration template for this class. """
        if not hasattr(cls, f"_{cls.__name__}__ct"):
            raise RuntimeError(f"No configuration template found for class "
                               f"{cls.__name__}, did you forget to register_config()?")
        return getattr(cls, f"_{cls.__name__}__ct")

    @classmethod
    def _initialize_class(cls):
        """ Initialize this class with required members. """

        if not hasattr(cls, "COMMAND_NAME"):
            cls.COMMAND_NAME = cls.__name__

        if not hasattr(cls, "COMMAND_PATH"):
            cls.COMMAND_PATH = cls.COMMAND_NAME.lower()

        # Explicit name mangling...
        # Set default configuration template.
        setattr(cls, f"_{cls.__name__}__ct", ConfigTemplate())

    @classmethod
    def register_config(cls, config: Config):
        """ Register class configuration in provided config. """

        cls._initialize_class()
        parser = config.add_subcommand(cls.COMMAND_NAME)
        cls.register_options(parser)

    @classmethod
    def register_model(cls, config: Config):
        """ Register model configuration in provided config. """

        cls._initialize_class()
        parser = config.add_model(cls.COMMAND_NAME, cls)
        cls.register_options(parser)

    @classmethod
    def var_path_name(cls, command_path: str, var_name: str):
        """
        Get path to the given variable in the configuration
        system.

        :param command_path: Name of the command.
        :param var_name: Name / path of the variable.

        :return: Fully qualified variable name.
        """

        return command_path + "." + var_name

    def serialize_config(self) -> dict:
        """ Serialize configuration for this object. """
        return {
            "config_data": self.c.serialize()
        }

    def deserialize_config(self, cfg: dict):
        """ Deserialize configuration for this object from given dictionary. """
        self.c.deserialize(cfg["config_data"])

    def _set_instance(self):
        """ Use the self instance as the main instance for this class. """
        self.config[self.var_path("instance")] = self

    T = TypeVar("T")

    def get_instance(self, cls: T) -> T:
        """ Get the main instance of given class. """
        if not hasattr(cls, f"COMMAND_PATH"):
            raise RuntimeError(f"Unable to get instance of unregistered class "
                               f"{cls.__name__}, did you forget to register_config()?")

        return self.config[self.var_path_name(cls.COMMAND_PATH, "instance")]

    def var_path(self, var_name: str) -> str:
        """
        Get path to the given variable in the configuration
        system.

        :param var_name: Name / path of the variable.

        :return: Fully qualified variable name.
        """

        return self.var_path_name(self.COMMAND_PATH, var_name)

    def get_var(self, var_name: str) -> Any:
        """
        Get variable by name.

        :param var_name: Variable name / path, which was
            provided to the var_path method.

        :return: Returns value of the variable.
        """

        return self.config[self.var_path(var_name)]
