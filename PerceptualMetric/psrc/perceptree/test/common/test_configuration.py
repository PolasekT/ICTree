# -*- coding: utf-8 -*-

"""
Testing of the configuration system.
"""

import pytest

from perceptree.perceptree_main import *


def prepare_config(argv: List[str]) -> Config:
    config = Config()

    # Register systems.
    PerceptreeMain.register_config(config)
    LoggingConfigurator.register_config(config)
    Serializer.register_config(config)
    TensorBoardLogger.register_config(config)
    DataLoader.register_config(config)
    DataFeaturizer.register_config(config)
    DataExporter.register_config(config)
    EvaluationProcessor.register_config(config)
    PredictionProcessor.register_config(config)

    # Register models.
    FeaturePredictor.register_model(config)
    ImagePredictor.register_model(config)
    CombinedPredictor.register_model(config)

    config.init_options()
    config.parse_args(argv)

    return config


@pytest.fixture()
def config_import() -> Config:
    argv = """
        Logging -v
        Data
        --experiment-result-file
        /projects/data/tturk_result_data.csv
        --experiment-tree-scores-file
        /projects/data/tturk_tree_scores.csv
        --experiment-view-scores-file
        /projects/data/tturk_view_scores.csv
        --tree-image-path
        /projects/data/tree/
    """.split()

    return prepare_config(argv=argv)


@pytest.fixture()
def config_load() -> Config:
    argv = """
        Logging -v
        Data
        --dataset-path
        /projects/data/dataset/
        --load-node-data
        False 
    """.split()

    return prepare_config(argv=argv)


class TestConfiguration:

    def test_init(self):
        config = Config()
        config.init_options()
        config.parse_args([ ])

    def test_import(self, config_import):
        pass

    def test_load(self, config_load):
        pass
