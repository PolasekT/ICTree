# -*- coding: utf-8 -*-

"""
Serialization and deserialization utilities.
"""

import io
import pickle as pk

from perceptree.common.configuration import Config
from perceptree.common.configuration import Configurable
from perceptree.common.logger import Logger


class Serializer(Logger, Configurable):
    """ Helper class for serialization and deserialization of classes. """

    COMMAND_NAME = "Serializer"
    """ Name of this command, used for configuration. """

    DUMP_CONFIG = None
    """ Config used for Pickle dumping operations. """

    def __init__(self, config: Config):
        super().__init__(config=config)
        self._set_instance()

        self.__l.info("Initializing serialization system...")

    @classmethod
    def register_options(cls, parser: Config.Parser):
        """ Register configuration options for this class. """

        pass

    def serialize_model(self, config: Config, instance: object, file_path: str):
        """ Serialize model instance and save to given path. """

        self.__l.info("\tSaving model to file \"{}\" ."
                           .format(file_path))
        Serializer.DUMP_CONFIG = config
        serialized = instance.serialize()
        serialized = pk.dumps(serialized)
        Serializer.DUMP_CONFIG = None
        try:
            with open(file_path, "wb") as file:
                file.write(serialized)
        except (OSError, IOError) as e:
            self.__l.error("\tFailed to save the model to file: \"{}\""
                           .format(str(e)))

    def deserialize_model(self, config: Config, instance: object, file_path: str):
        """ De-serialize model instance from given path. """

        self.__l.info("\tLoading model from file \"{}\" ."
                           .format(file_path))

        serialized = ""
        try:
            with open(file_path, "rb") as file:
                serialized = file.read()
        except (OSError, IOError) as e:
            self.__l.error("\tFailed to load the model from file: \"{}\"\n\t Attempting to continue..."
                                .format(str(e)))

        # Fix for missing types in older library versions.
        class CustomUnpickler(pk.Unpickler):
            def find_class(self, module, name):
                if name == "InterpolationMode":
                    return str

                try:
                    return super().find_class(__name__, name)
                except AttributeError:
                    return super().find_class(module, name)
        def pk_loads(serialized: str) -> any:
            f = io.BytesIO(serialized)
            unpickler = CustomUnpickler(f)
            return unpickler.load()

        self.__l.info("\tDeserializing model from string (Length {})..."
                      .format(len(serialized)))
        Serializer.DUMP_CONFIG = config
        instance.deserialize(pk_loads(serialized))
        Serializer.DUMP_CONFIG = None
        self.__l.info("\tModel has been successfully deserialized!")

    def deserialize_model_torch(self, config: Config, instance: object, file_path: str):
        """ De-serialize model instance from given path. """

        Serializer.DUMP_CONFIG = config
        instance.deserialize_from_file(path=file_path)
        Serializer.DUMP_CONFIG = None

    def deserialize_model_dict(self, config: Config, instance: object, serialized: dict):
        """ De-serialize model instance from given dictionary. """

        Serializer.DUMP_CONFIG = config
        instance.deserialize(serialized)
        Serializer.DUMP_CONFIG = None

class Serializable(object):
    """ Base class for serializable classes. """

    def serialize(self) -> dict:
        """ Dummy version of serialize which should be overriden. Returns dictionary to be serialized."""
        pass

    def deserialize(self, data: dict):
        """ Dummy version of deserialize which should be overriden. Gets the same dictionary returned from serialize."""
        pass
