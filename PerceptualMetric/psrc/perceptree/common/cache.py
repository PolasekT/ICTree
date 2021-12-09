# -*- coding: utf-8 -*-

"""
Simple addressable cache.
"""

import argparse
import datetime
import logging
import os
import pathlib
import pickle as pk
import re
from enum import Enum
from io import StringIO
from typing import Any, Dict, Iterable, Optional, List, Union, TextIO, TypeVar

from ruamel.yaml import YAML


def deep_copy_dict(first: dict) -> dict:
    """ Perform a deep copy of all dictonaries on input and return result. """

    copy = first.copy()

    for key, value in copy.items():
        if isinstance(value, dict):
            copy[key] = deep_copy_dict(copy[key])

    return copy


DictT = TypeVar("DictT")
def update_dict_recursively(first: DictT, second: DictT,
                            create_keys: bool = False,
                            convert: bool = False) -> DictT:
    """
    Update the first dictionary with values from the second
    dictionary. Only keys common to both dictionaries will
    be used! The type of value from the second dictionary
    must also be the same or it will be skipped.
    Warning: The original dictionary (first) will be updated
    in place!

    :param first: Dictionary to update.
    :param second: Update with values from this dictionary
    :param create_keys: Create non-existent keys in first?
    :param convert: Convert dictionary instances in second
        into type used in first?

    :return: Returns updated first dictionary.
    """

    for key, value in second.items():
        if (not create_keys and key not in first) or \
                (key in first and not isinstance(first[key], type(value))):
            continue
        if isinstance(value, type(first)):
            first[key] = update_dict_recursively(
                first.get(key, type(first)()), value,
                create_keys=create_keys,
                convert=convert
            )
        elif convert and isinstance(value, dict):
            first[key] = update_dict_recursively(
                first.get(key, type(first)()), type(first)(value),
                create_keys=create_keys,
                convert=convert
            )
        else:
            first[key] = value

    return first


class CacheDict(dict):
    """ Special type of dictionary used in cache structure. """

    yaml_tag = u"tag:CacheDict"

    @classmethod
    def to_yaml(cls, representer, data):
        return representer.represent_mapping(cls.yaml_tag, data)

    @classmethod
    def from_yaml(cls, constructor, node):
        return CacheDict(constructor.construct_mapping(node, True))


class CommonDict(dict):
    """ Special type of dictionary usable to store data in cache structure. """

    yaml_tag = u"tag:CommonDict"

    @classmethod
    def to_yaml(cls, representer, data):
        return representer.represent_mapping(cls.yaml_tag, data)

    @classmethod
    def from_yaml(cls, constructor, node):
        return CommonDict(constructor.construct_mapping(node, True))


def prepare_cache_yaml() -> YAML:
    """ Prepare yaml usable with cache objects. """
    yaml = YAML(typ="safe")

    yaml.register_class(CacheDict)
    yaml.register_class(CommonDict)

    return yaml


class Cache(object):
    """
    Simple addressable cache.
    """

    CACHE_PATH_SEP = "."
    """
    Separator for cache paths.
    """

    CACHE_FILE_EXT = ".cache"
    """
    File extension used for filesystem cached values.
    """

    def __init__(self, initial: Optional[dict] = None):
        self.cache = CacheDict()

        if initial is not None:
            self.initialize_options(initial)

    def __getitem__(self, option: str) -> any:
        """
        Access configuration option by name.

        :param option: Path to the option.

        :return: Returns value of the option or None if
        requested option does not exist.
        """

        return self.get_path(option, create=False, none_when_missing=True)

    def __setitem__(self, option: str, value: any):
        """
        Change configuration option by name
        The key must already exist!

        :param option: Path to the option.
        :param value: New value for the option, must
        be of the same type as the old one.
        """

        self.set_path(option, value, create=True)

    def cache_path(self, path: str) -> [str]:
        """
        Transform given configuration path into a list
        of keys in the config dictionary.

        :param path: String representation of cache
        option path.

        :return: Returns list of keys to get to the
        option.
        """

        return [ name.replace("\\", "") for name in re.split(r"(?<!\\)\.", path) ]

    @staticmethod
    def _recurse_regex_path_generator(path_dict: Union[CacheDict, any], path: List[str], current_path: str) -> Iterable:
        """ Recursion helper for generating all matching regex paths. """

        if isinstance(path_dict, CacheDict):
            for name, name_dict in path_dict.items():
                match = re.fullmatch(path[0], name)
                if match and len(path) == 1:
                    yield path_dict, name, current_path
                elif match:
                    yield from Cache._recurse_regex_path_generator(
                        path_dict=name_dict, path=path[1:],
                        current_path=Cache.CACHE_PATH_SEP.join(filter(None, [ current_path, name ]))
                    )

    def get_path_dict_reg(self, regex_path: str) -> Iterable:
        """
        Generator for dictionaries and names matching given regex path.

        :param regex_path: Path to the value/dictionary possibly containing regex.

        :return: Yields dictionary, name, path pairs.
        """

        path_parts = self.cache_path(regex_path)
        path_generator = Cache._recurse_regex_path_generator(
            path_dict=self.cache, path=path_parts,
            current_path=""
        )

        for path_dict, name, path in path_generator:
            if isinstance(path_dict, CacheDict):
                yield path_dict, name, path

    def get_path_dict(self, path: str,
                      create: bool = False,
                      none_when_missing: bool = False) -> Optional[CacheDict]:
        """
        Get dictionary which contains given path.

        :param path: Path to the value.
        :param create: When set to true, the path will be
            created, else exception will be thrown when any
            dictionary in the path is missing.
        :param none_when_missing: Return None if there is no
            parameter with such name.

        :return: Returns the dictionary which should contain
        given path.
        """

        path_parts = self.cache_path(path)
        cache = self.cache

        # Iterate path only, skip the option itself.
        for key in path_parts[:-1]:
            if cache.get(key) is None:
                if create:
                    cache.setdefault(key, CacheDict())
                else:
                    if none_when_missing:
                        return None
                    else:
                        raise KeyError("Given key \"{}\" in path \"{}\" does not exist!"
                                       .format(key, path))
            if not isinstance(cache.get(key), CacheDict):
                raise KeyError("Given key \"{}\" in path \"{}\" does not lead to a dictionary!"
                               .format(key, path))

            cache = cache.get(key)

        return cache

    def get_path(self, path: str, create: bool = False,
                 none_when_missing: bool = False,
                 default: any = None) -> any:
        """
        Access value by path.

        :param path: Path to the value.
        :param create: Create any missing keys in the path.
            When set to False, this method will throw for
            missing keys!
        :param none_when_missing: Return None if there is no
            parameter with such name.
        :param default: Value returned by default.

        :return: Returns value of the option or None if the
        create False and the key does not exist.
        """

        if not path:
            return self.cache

        name = self.cache_path(path)[-1]
        cache = self.get_path_dict(path, create, none_when_missing)

        if cache is None:
            cache = CacheDict()

        result = cache.get(name)

        if name not in cache and not create:
            if none_when_missing:
                return None
            elif default is not None:
                return default
            else:
                raise KeyError("Given cache path \"{}\" does not exist!"
                               .format(path))

        return result

    def set_path(self, path: str, value: any,
                 create: bool = False):
        """
        Change value by path.

        :param path: Path to the value.
        :param value: New value for the option, must
        be of the same type as the old one.
        :param create: Create any missing keys in the path.
        When set to False, this method will throw for missing
        keys!
        """

        name = self.cache_path(path)[-1]
        cache = self.get_path_dict(path, create)

        if cache.get(name) is None:
            if not create:
                raise KeyError("Given option \"{}\" does not exist!"
                               .format(path))
        else:
            if not isinstance(value, type(cache[name])) and value is not None:
                raise ValueError("Value for given option \"{}\" must be {}!"
                                 .format(path, type(cache[name])))

        cache[name] = value

    def has_value(self, path: str) -> bool:
        """
        Does given path contain an end-point value?

        :param path: Path to the value.

        :return: Returns whether given path contains a value.
        """

        name = self.cache_path(path)[-1]
        path_dict = self.get_path_dict(
            path=path, create=False,
            none_when_missing=True
        )

        if path_dict is None or \
           name not in path_dict or \
           isinstance(path_dict[name], CacheDict):
            return False
        else:
            return True

    def update_options(self, cache: dict):
        """
        Update values with those in given dictionary.

        :param cache: Dictionary containing the options.
        """

        self.cache = update_dict_recursively(self.cache, cache, create_keys=False)

    def initialize_options(self, cache: dict):
        """
        Initialize values with those in given dictionary.

        :param cache: Dictionary containing the options.
        """

        self.cache = update_dict_recursively(self.cache, cache, create_keys=True, convert=True)

    def load_cache_path(self, cache_path: str, path: str) -> int:
        """
        Load cached data from given base path using provided data path.

        :param cache_path: Base path to the root of the cache in filesystem.
        :param path: Path to the root directory or end data, which should
            be loaded. Provide path = "" to load everything.

        :return: Returns number of values which were found and loaded successfully.
        """

        loaded_values = 0
        load_path = pathlib.Path(
            f"{cache_path}/{'/'.join(self.cache_path(path))}"
        ).absolute()
        load_end_data_path = load_path.with_suffix(self.CACHE_FILE_EXT)
        base_cache_path = path

        if load_end_data_path.exists():
            # Single end data.
            cache_files = [ load_end_data_path ]
            load_path = load_path.parent
            base_cache_path = self.CACHE_PATH_SEP.join(self.cache_path(path)[:-1])
        else:
            # Search directory structure.
            cache_files = load_path.rglob(f"*{self.CACHE_FILE_EXT}")

        for cached_file_path in cache_files:
            absolute_path = cached_file_path.absolute()
            relative_path = absolute_path.relative_to(load_path)

            # Get path to the current cache file within the cache structure.
            add_cache_path = self.CACHE_PATH_SEP.join(relative_path.with_suffix("").parts)
            if base_cache_path:
                value_cache_path = f"{base_cache_path}{self.CACHE_PATH_SEP}{add_cache_path}"
            else:
                value_cache_path = add_cache_path

            # Load the cache file.
            try:
                with open(cached_file_path, "rb") as f:
                    value = pk.load(f)
            except:
                continue

            # Add the value to the current cache.
            self.set_path(path=value_cache_path, value=value, create=True)
            loaded_values += 1

        return loaded_values

    def load_cache_yaml(self, config: Union[str, TextIO]):
        """
        Load the YAML formatted cache from given string and
        override all contained options within this config.

        :param config: YAML formatted configuration.
        """

        yaml = prepare_cache_yaml()
        data = { }

        # Load the configuration.
        try:
            data = yaml.load(config)
        except (TypeError, AttributeError, NotImplementedError) as e:
            raise AttributeError("Failed to parse YAML cache file: \"{}\""
                                 .format(str(e)))

        # Override any common attributes.
        self.initialize_options(data)

    def load_cache_yaml_from_file(self, path: str):
        """
        Load the YAML formatted cache from given path and
        set override all contained options within this config.

        :param path: Path to the config.
        """

        # Load the configuration file.
        try:
            with open(path, 'r') as f:
                self.load_cache_yaml(f)
        except (OSError, IOError) as e:
            raise RuntimeError("Failed to load YAML cache file: \"{}\""
                               .format(str(e)))
        except (TypeError, AttributeError, NotImplementedError) as e:
            raise AttributeError("Failed to parse YAML cache file: \"{}\""
                                 .format(str(e)))

    def _save_cache_path_recurse(self, current: any, current_path: pathlib.Path) -> int:
        """ Recursively save all data from current position. """

        saved_count = 0

        if isinstance(current, CacheDict):
            # Found a dictionary -> Continue with recursion.
            for name, value in current.items():
                saved_count += self._save_cache_path_recurse(
                    value, current_path / pathlib.Path(name))
        elif current is None:
            # Found dead end -> Skip it.
            pass
        else:
            # Found end data -> Save it.
            save_path = current_path.absolute().with_suffix(self.CACHE_FILE_EXT)
            os.makedirs(save_path.parent, exist_ok=True)
            with open(save_path, "wb") as f:
                pk.dump(current, f)
            saved_count += 1

        return saved_count

    def save_cache_path(self, cache_path: str, path: str) -> int:
        """
        Save cached data to given base path using provided data path.

        :param cache_path: Base path to the root of the cache in filesystem.
        :param path: Path to the root directory or end data, which should
            be saved. Provide path = "" to save everything.

        :return: Returns number of values which were saved successfully.
        """

        save_path = pathlib.Path(
            f"{cache_path}/{'/'.join(self.cache_path(path))}"
        ).absolute()
        base_cache_path = path

        base_cache = self.get_path(
            path=base_cache_path,
            create=False,
            none_when_missing=True
        )

        saved_values = self._save_cache_path_recurse(
            current=base_cache,
            current_path=save_path
        )

        return saved_values

    def save_cache_yaml(self) -> str:
        """
        Save the current cache options into YAML formatted string.

        :return: Returns YAML formatted string containing
        the current configuration.
        """

        yaml = prepare_cache_yaml()
        string_stream = StringIO()

        yaml.dump(self.cache, string_stream)
        string_stream.seek(0)

        return string_stream.read()

    def save_cache_yaml_to_file(self, path: str):
        """
        Save YAML formatted cache to given path.

        :param path: Path to the cache.
        """

        # Save the configuration file.
        try:
            with open(path, 'w') as f:
                f.write(self.save_cache_yaml())
                f.flush()
        except (OSError, IOError) as e:
            raise RuntimeError("Failed to save YAML cache file: \"{}\""
                               .format(str(e)))
        except (TypeError, AttributeError, NotImplementedError) as e:
            raise AttributeError("Failed to parse YAML cache to file: \"{}\""
                                 .format(str(e)))

