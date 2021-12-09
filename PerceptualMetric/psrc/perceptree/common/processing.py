# -*- coding: utf-8 -*-

"""
Utilities and helpers used for data pre-processing.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sklearn.preprocessing as skp

from perceptree.common.logger import Logger
from perceptree.common.serialization import Serializable


class Scaler(Logger, Serializable):
    """
    Wrapper around any type of scaler with common interface.

    :param scaler_type: Scaler to use, must be one of:
        "standard", "minmax", "maxabs", "robust"
        "probust", "quantile", "power", "normal",
        "none"
    """

    def __init__(self, scaler_type: str = "none"):
        self._scaler_name = scaler_type
        self._base_class = Scaler.scaler_constructor_factory(
            scaler_type=scaler_type
        )
        self._scaler = None

    @staticmethod
    def scaler_constructor_factory(scaler_type: str) -> any:
        """ Factory used for construction scaler types. """
        scalers = {
            "standard": skp.StandardScaler,
            "minmax":   skp.MinMaxScaler,
            "maxabs":   skp.MaxAbsScaler,
            "robust":   skp.RobustScaler,
            "probust":  PositiveRobustScaler,
            "quantile": skp.QuantileTransformer,
            "power":    skp.PowerTransformer,
            "normal":   skp.Normalizer,
            "none":     None
        }

        return scalers.get(scaler_type, skp.StandardScaler)

    @property
    def scaler(self) -> any:
        """ Make sure we have instance of requested scaler and return it. """
        if self._scaler is not None:
            self._scaler = self._base_class()
        return self._scaler

    @property
    def scaler_type(self) -> any:
        """ Get type of the currently used scaler. """
        return self._base_class

    @property
    def scaler_name(self):
        """ Get name of the currently used scaler. """
        return self._scaler_name

    def fit(self, x: any, y: Optional[any] = None) -> "Scaler":
        """ Fit scaler to given data. """

        if self.scaler:
            self.scaler.fit(x=x, y=y)

        return self

    def transform(self, x: any) -> any:
        """ Transform input data. """

        return self.scaler.transform(x=x) if self.scaler else x

    def inverse_transform(self, x: any) -> any:
        """ Inverse transform input data. """

        return self.scaler.inverse_transform(x=x) if self.scaler else x

    def serialize(self) -> dict:
        """ Serialize the Scaler. """
        return {
            "scaler_name": self._scaler_name,
            "scaler": self._scaler
        }

    def deserialize(self, data: dict):
        """ Serialize the Scaler from provided data. """
        self.__init__(scaler_type=data["scaler_name"])
        self._scaler = data["scaler"]


class PositiveRobustScaler(skp.RobustScaler):
    """ Robust scaler which maps values to only positive numbers. """

    def __init__(self, with_centering=True, with_scaling=True,
                 quantile_range=(25.0, 75.0), copy=True):
        super(PositiveRobustScaler, self).__init__(
            with_centering=with_centering, with_scaling=with_scaling,
            quantile_range=quantile_range, copy=copy
        )

        self._offset = 0.0

    def fit(self, x, y = None):
        """ Fit scaler to given data. """

        super(PositiveRobustScaler, self).fit(x, y)
        if isinstance(x, pd.DataFrame):
            min_value = np.min(x)
            transformed = super(PositiveRobustScaler, self).transform([min_value])[0]
        else:
            min_value = np.min(x)
            transformed = super(PositiveRobustScaler, self).transform([[min_value]])[0][0]
        self._offset = -transformed

        return self

    def transform(self, x):
        """ Transform input data. """

        result = super(PositiveRobustScaler, self).transform(x)
        result += self._offset

        return result

    def inverse_transform(self, x):
        """ Inverse transform input data. """

        input_values = x - self._offset
        result = super(PositiveRobustScaler, self).inverse_transform(input_values)

        return result
