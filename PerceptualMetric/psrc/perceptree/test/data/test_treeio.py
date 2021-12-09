# -*- coding: utf-8 -*-

"""
Testing of the TreeIO compatibility classes.
"""

import pytest

from perceptree.data.treeio import *

from perceptree.test.data.dummy_tree_file import dummy_tree_file_content
from perceptree.test.data.dummy_tree_image import dummy_tree_image_dict


def gen_dummy_tree_file(load_static: bool = True, load_dynamic: bool = True,
                    load_node: bool = True, calculate_stats: bool = True) -> TreeFile:
    return TreeFile(
        file_content=dummy_tree_file_content(),
        load_static=load_static, load_dynamic=load_dynamic,
        load_node=load_node, calculate_stats=calculate_stats
    )


def gen_dummy_tree_file_full() -> TreeFile:
    return gen_dummy_tree_file(
        load_static=True, load_dynamic=True,
        load_node=True, calculate_stats=True
    )


@pytest.fixture()
def dummy_tree_file_full() -> TreeFile:
    return gen_dummy_tree_file_full()


def gen_dummy_tree_image(resolution: int,
                         channels: int) -> TreeImage:
    return TreeImage(
        image_dict=dummy_tree_image_dict((resolution, resolution, channels))
    )


def gen_dummy_tree_image_full() -> TreeImage:
    return gen_dummy_tree_image(
        resolution=256, channels=3
    )

@pytest.fixture()
def dummy_tree_image_full() -> TreeImage:
    return gen_dummy_tree_image_full()


class TestTreeFile:

    def test_initialization(self):
        tree_file = TreeFile()
        tree_file = gen_dummy_tree_file(
            load_static=True, load_dynamic=True,
            load_node=True, calculate_stats=True
        )


class TestTreeImage:

    def test_initialization(self):
        tree_image = gen_dummy_tree_image(
            resolution=255, channels=3
        )

