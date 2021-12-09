# -*- coding: utf-8 -*-

"""
Model used for feature and image combined predictions.
"""

from perceptree.model.base import *


class CombinedPredictor(BaseModel):
    """
    Model used for feature and image combined predictions.

    :param config: Application configuration.
    """

    MODEL_NAME = "cp"
    """ Name of this prediction model. """
    CATEGORY_NAME = "CombinedPredictor"
    """ Category as visible to the user. """

    def __init__(self, config: Config):
        super().__init__(config, self.MODEL_NAME)

    @classmethod
    def register_options(cls, parser: Config.Parser):
        """ Register configuration options for this class. """

        super().register_common_options(parser)

    def serialize(self) -> dict:
        """ Get data to be serialized for this model. """
        pass

    def deserialize(self, data: dict):
        """ Deserialize data for this model. """
        pass

    def train(self):
        """ Train this model. """

        features = self._featurizer.calculate_features()

    def predict(self):
        """ Predict results for provided trees. """
        pass

