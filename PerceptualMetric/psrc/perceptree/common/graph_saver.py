# -*- coding: utf-8 -*-

"""
Helper used for saving graphs.
"""

import datetime
import os
import pathlib

import matplotlib.pyplot as plt


class GraphSaver:
    """
    Helper class used for saving graphs in specified
    format, or just displaying them to the user.
    """

    enabled = True
    """
    Enable graph saving.
    """

    save_as_file = False
    """
    Set to True to save the graph as a file.
    """

    show_graph = True
    """
    Set to True to display the graph using plt.show().
    """

    file_extension = ".png"
    """
    Extension used for the graph files.
    """

    add_timestamp = False
    """
    Add timestamp to the filename?
    """

    directory_prefix = ""
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

        return "{h}-{m}-{s}-{ms}-{D}_{M}_{Y}".format(
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

        output_path = pathlib.Path(f"{cls.directory_prefix}/{name}{cls.file_extension}")
        if output_path.exists():
            return cls._generate_name_path(name, override_add_timestamp=True)

        os.makedirs(output_path.absolute().parent, exist_ok=True)

        return output_path

    @classmethod
    def export_graph(cls,
                     path: str,
                     plot=plt):
        """
        Force export given graph to the path provided.

        :param path: Path to save the graph to.
        :param plot: The plot itself.
        """

        fig = plt.gcf()
        plt.rcParams["agg.path.chunksize"] = 10000
        fig.savefig(path)

    @classmethod
    def save_graph(cls,
                   name: str,
                   plot=plt,
                   override_add_timestamp=None,
                   override_show=None):
        """
        Save or display given plot.

        :param name: Name of the plot.
        :param plot: The plot itself.
        :param override_add_timestamp: Set to True of False to
            override the add_timestamp attribute.
        :param override_show: Override displaying of the graph.
        """

        if not cls.enabled:
            return

        try:
            if cls.save_as_file:
                if override_add_timestamp is None:
                    add_timestamp = cls.add_timestamp
                else:
                    add_timestamp = override_add_timestamp

                if add_timestamp:
                    name = f"{name}_{cls._generate_timestamp_str()}"

                output_path = cls._generate_name_path(name, override_add_timestamp)

                GraphSaver.export_graph(path=output_path, plot=plot)

            if (override_show is None or override_show) and cls.show_graph and \
                    ("DISPLAY" not in os.environ or os.environ["DISPLAY"] != "False"):
                plt.show()
        except Exception as e:
            print(f"Failed to save graph \"{e}\"")
