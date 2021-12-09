# -*- coding: utf-8 -*-

"""
Wrapper for testing tree file.
"""

import os
import pathlib


def dummy_tree_file_content() -> str:
    dummy_tree_file_path = pathlib.Path(os.path.realpath(__file__)).parent / "dummy_tree_file.tree"
    with open(dummy_tree_file_path, "r") as f:
        return f.read()
