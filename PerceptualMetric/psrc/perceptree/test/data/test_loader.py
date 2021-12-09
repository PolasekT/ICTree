# -*- coding: utf-8 -*-

"""
Testing of the featurizer system.
"""

import pytest

from perceptree.data.loader import *

from perceptree.test.common.test_configuration import *
from perceptree.test.data.test_treeio import *


def gen_data_loader_full(config_load: Config) -> BaseDataLoader:
    data_loader = DataLoader(config=config_load)
    data_loader.process()

    return data_loader


@pytest.fixture()
def data_loader_full(config_load) -> BaseDataLoader:
    return gen_data_loader_full(config_load)


def gen_data_loader_limited() -> BaseDataLoader:
    data_loader = CustomDataLoader()

    tree_count = 2
    view_count = 5
    view_types = [ "base", "albedo", "depth", "light", "normal", "shadow" ]

    data_loader.load_data(
        tree_files={
            tree_id: gen_dummy_tree_file_full()
            for tree_id in range(tree_count)
        },
        tree_views={
            tree_id: [
                ( view_id, view_type, gen_dummy_tree_image_full() )
                for view_id in range(view_count)
                for view_type in view_types
            ]
            for tree_id in range(tree_count)
        },
        tree_scores={
            tree_id: [
                ( view_id, view_id + 0.5, view_id - 0.5, view_id + 1.0, 0.5 )
                for view_id in range(view_count)
            ] + [
                ( -1, view_count / 2.0, view_count / 2.0 - 0.5, view_count / 2.0 + 0.5, 0.5 )
            ]
            for tree_id in range(tree_count)
        },
        tree_comparisons=[
            ( ( tree_id1, view_id1 ), ( tree_id2, view_id2 ), np.random.randint(1, 3))
            for tree_id1 in range(tree_count)
            for tree_id2 in range(tree_count)
            for view_id1 in range(view_count)
            for view_id2 in range(view_count)
            if tree_id1 != tree_id2
        ]
    )

    return data_loader


@pytest.fixture()
def data_loader_limited() -> BaseDataLoader:
    return gen_data_loader_limited()


class TestDataLoader:

    def test_initialize_empty(self):
        data_loader = BaseDataLoader()

    def test_initialize_full(self, data_loader_full: BaseDataLoader):
        pass

    def test_initialize_custom(self, data_loader_limited: BaseDataLoader):
        pass

