# -*- coding: utf-8 -*-

"""
Helper used for saving files.
"""

import datetime
import os
import pathlib
import shutil

import pandas as pd


class FileSaver:
    """
    Helper class used for saving files in specified
    format.
    """

    enabled = True
    """
    Enable graph saving.
    """

    add_timestamp = False
    """
    Add timestamp to the filename?
    """

    file_prefix = ""
    """
    Prefix used for all of the saved files.
    """

    @classmethod
    def _generate_timestamp_str(cls) -> str:
        """
        Generate timestamp string to identify files.

        :return: Returns string representation of the timestamp.
        """

        now = datetime.datetime.now()

        return "{D}_{M}_{Y}-{h}-{m}-{s}-{ms}".format(
            h=now.hour,
            m=now.minute,
            s=now.second,
            ms=int(now.microsecond / 1000.0),
            D=now.day,
            M=now.month,
            Y=now.month
        )

    @classmethod
    def _generate_name_path(cls,
                            name: str,
                            override_add_timestamp=None) -> str:
        """ Generate path for given name and make sure it exists. """

        if override_add_timestamp is None:
            add_timestamp = cls.add_timestamp
        else:
            add_timestamp = override_add_timestamp

        if add_timestamp:
            name = f"{name}_{cls._generate_timestamp_str()}"

        output_path = pathlib.Path(f"{cls.file_prefix}/{name}")
        if output_path.exists():
            return cls._generate_name_path(name, override_add_timestamp=True)

        os.makedirs(output_path.absolute().parent, exist_ok=True)

        return output_path

    @classmethod
    def save_string(cls,
                    name: str,
                    obj: any,
                    override_add_timestamp=None):
        """
        Save given object using the logging directory.

        :param name: Name of the file.
        :param obj: Object to save as string.
        :param override_add_timestamp: Set to True of False to
            override the add_timestamp attribute.
        """

        if not cls.enabled:
            return

        output_path = cls._generate_name_path(name, override_add_timestamp)
        with open(f"{output_path}.txt", "w") as out:
            out.write(str(obj))

    @classmethod
    def save_csv(cls,
                 name: str,
                 obj: pd.DataFrame,
                 override_add_timestamp=None):
        """
        Save given frame as csv into the logging directory.

        :param name: Name of the file.
        :param obj: Object to save as string.
        :param override_add_timestamp: Set to True of False to
            override the add_timestamp attribute.
        """

        if not cls.enabled:
            return

        output_path = cls._generate_name_path(name, override_add_timestamp)
        obj.to_csv(f"{output_path}.csv", sep=";")

    @classmethod
    def save_binary(cls,
                    name: str,
                    obj: any,
                    override_add_timestamp=None):
        """
        Save given object using the logging directory.

        :param name: Name of the file.
        :param obj: Object to save as string.
        :param override_add_timestamp: Set to True of False to
            override the add_timestamp attribute.
        """

        if not cls.enabled:
            return

        output_path = cls._generate_name_path(name, override_add_timestamp)
        with open(f"{output_path}.bin", "wb") as out:
            out.write(obj)

    @classmethod
    def log_file(cls,
                 file_path: str,
                 name: str,
                 override_add_timestamp=None):
        """
        Copy file from given file_path to the logging directory.

        :param file_path: Path to the file.
        :param name: Name of the output file.
        :param override_add_timestamp: Set to True of False to
            override the add_timestamp attribute.
        """

        if not cls.enabled:
            return

        output_path = cls._generate_name_path(name, override_add_timestamp)
        input_suffix = pathlib.Path(file_path).suffix
        shutil.copyfile(file_path, f"{output_path}{input_suffix}")

