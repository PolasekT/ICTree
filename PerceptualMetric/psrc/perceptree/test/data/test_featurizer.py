# -*- coding: utf-8 -*-

"""
Testing of the featurizer system.
"""

import pytest

from perceptree.data.featurizer import *

from perceptree.test.data.test_loader import *


@pytest.fixture()
def featurizer_empty(config_load: Config) -> DataFeaturizer:
    return DataFeaturizer(config=config_load)


@pytest.fixture()
def featurizer_full(data_loader_full: DataLoader) -> DataFeaturizer:
    return DataFeaturizer(config=data_loader_full.config)


class TestFeaturizer:

    def test_initialize_empty(self, featurizer_empty: DataFeaturizer):
        pass

    def test_initialize_full(self, featurizer_full: DataFeaturizer):
        pass

    def test_calculate_data(self, featurizer_full: DataFeaturizer):
        features = featurizer_full.calculate_features()
        views = featurizer_full.calculate_views()
        scores = featurizer_full.calculate_scores()

        data_loader_custom = gen_data_loader_limited()
        features_custom = featurizer_full.calculate_features_for_data(
            data_loader=data_loader_custom, configuration=features["configuration"]
        )
        views_custom = featurizer_full.calculate_views_for_data(
            data_loader=data_loader_custom, configuration=views["configuration"]
        )
        scores_custom = featurizer_full.calculate_scores_for_data(
            data_loader=data_loader_custom, configuration=scores["configuration"]
        )

