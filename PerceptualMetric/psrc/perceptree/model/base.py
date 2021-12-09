# -*- coding: utf-8 -*-

"""
Base model class used by all other models.
"""

import argparse
import datetime
import inspect
import os
import pathlib
import re
import time
from dateutil.relativedelta import relativedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from perceptree.common.pytorch_safe import *

from perceptree.common.cache import CacheDict
from perceptree.common.cache import update_dict_recursively
from perceptree.common.configuration import Config
from perceptree.common.configuration import Configurable
from perceptree.common.file_saver import FileSaver
from perceptree.common.graph_saver import GraphSaver
from perceptree.common.serialization import Serializable
from perceptree.common.serialization import Serializer
from perceptree.common.logger import FittingBar
from perceptree.common.logger import Logger
from perceptree.common.util import parse_bool_string
from perceptree.common.util import parse_list_string
from perceptree.common.util import recurse_dict
from perceptree.data.dataset import DatasetConfig
from perceptree.data.dataset import TreeDataset
from perceptree.data.featurizer import DataFeaturizer
from perceptree.data.loader import BaseDataLoader


class BaseNetwork(Logger):
    """ Base class for all networks - the internal predictors of a model. """

    SNAPSHOT_EXTENSION = "pth"
    """ File extension used for model snapshots. """

    def __init__(self, cfg: dict, model_names: List[str]):
        super().__init__()

        self._cfg = cfg
        self._model_names = model_names
        self._model = None
        self._data_configuration = None
        self._training_state = None

    @property
    def cfg(self) -> dict:
        """ Get the configuration dictionary. """
        return self._cfg

    def serialize(self) -> dict:
        """ Serialize current state of this model. """

        source_file = inspect.getfile(self.__class__)
        source_code = ""
        try:
            with open(source_file) as f:
                source_code = f.read()
            FileSaver.save_string("source", source_code)
        except:
            self.__l.error("Failed to recover source code for serialization!")

        return {
            "cfg": self._cfg,
            "model": {
                model_name: model.state_dict() if isinstance(model, tnn.Module) else model
                for model_name, model in self._model.items()
            } if self._model is not None else None,
            "data_configuration": self._data_configuration,
            "training_state": self._training_state,
            "src": source_code,
        }

    def deserialize(self, serialized: dict):
        """ Deserialize model from given dictionary. """

        self.__init__(**serialized["cfg"])

        for model_name, model_state in serialized["model"].items():
            self.__l.info(f"Restoring {model_name} parameters...")

            if isinstance(self._model[model_name], tnn.Module):
                result = self._model[model_name].load_state_dict(model_state)
                self._model[model_name].train()
            else:
                result = "success"
                self._model[model_name] = model_state

            self.__l.info(f"\tFinished with \"{result}\"!")

        self._data_configuration = serialized["data_configuration"]
        self._training_state = serialized["training_state"]

        if self._training_state["finalized"]:
            self.finalize(**self._data_configuration)

    def deserialize_from_file(self, path: str):
        """ Deserialize model from given path. """

        with open(path, "rb") as f:
            serialized = torch.load(f)

        self.deserialize(serialized=serialized)

    @classmethod
    def from_serialized(cls, serialized: dict) -> "BaseNetwork":
        """ Load model from serialized representation. """

        model = cls(**serialized["cfg"])
        model.deserialize(serialized)

        return model

    def _initialize(self):
        """ Initialize the networks to default state. """

        self._model = {
            model_name: None
            for model_name in self._model_names
        }

        self._training_state = {
            "init": True,
            "finalized": False,
            "runtime": {
                model_name: {
                    "trained_epochs": -1,
                }
                for model_name in self._model_names
            },
            "losses": {
                model_name: [ ]
                for model_name in self._model_names
            }
        }

    def fit(self, dataset: TreeDataset, model_type: str, *args, **kwargs) -> dict:
        """ Fit model_type model using provided parameters. """

        raise NotImplemented

    def finalize(self, *args, **kwargs):
        """ Finalize model fitting operations. """

        pass

    def predict(self, inputs: Union[np.array, TreeDataset], *args, **kwargs) -> np.array:
        """ Predict results for given inputs. """

        raise NotImplemented

    def cross_validate(self, dataset: TreeDataset, *args, **kwargs) -> np.array:
        """ Cross-validate results on given dataset. """

        raise NotImplemented

    @property
    def data_configuration(self) -> dict:
        """ Get data configuration used to train this model. """
        return self._data_configuration

    def _get_devices(self) -> (torch.device, torch.device):
        """ Get training device and prediction device handles. """

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cpu_device = torch.device("cpu")

        return device, cpu_device

    def _get_option_for_type(self, option: str, model_type: str) -> any:
        """ Get config option for given model type. """

        value = self._cfg[option]
        if isinstance(value, dict):
            return value.get(model_type, None)
        else:
            return value

    def _get_runtime_option_for_type(self, option: str, model_type: str, *args, **kwargs) -> any:
        """ Get runtime values for given fitting type - cmp or score. """

        option_cls = self._get_option_for_type(option, model_type)
        model_runtime = self._training_state["runtime"].get(model_type, { })

        if option in model_runtime:
            option_value = model_runtime[option]
        else:
            if isinstance(option_cls, tuple):
                # Unwrap provided arguments.
                args += tuple(option_cls[1:-1])
                kwargs.update(dict(option_cls[-1]))
                option_cls = option_cls[0]

            option_value = option_cls(*args, **kwargs) if option_cls is not None else None
            update_dict_recursively(self._training_state, {
                "runtime": { model_type: { option: option_value } }
            })

        return option_value

    def _generate_loss_graph(self, losses: list, save_path: Optional[str] = None) -> (plt.Figure, pd.DataFrame):
        """ Generate loss graph from given list of losses and return the figure and the df. """

        losses_df = pd.DataFrame(data=losses, columns=("epoch", "loss", "type"))

        steps = max(1, min(100, len(losses_df.epoch.unique())))
        step = (losses_df.epoch.max() + 1) / steps
        plot_df = losses_df.copy()
        plot_df.epoch = ((plot_df.epoch + 0.5 * step) // step) * step

        g = sns.lineplot(x="epoch", y="loss", hue="type", data=plot_df)
        g.set_yscale("log")

        if save_path is not None:
            FileSaver.save_csv(save_path, obj=losses_df)

        return g, losses_df

    def _epoch_callback(self, model_type: str, model_name: str,
                        epoch: int, total_epochs: int, losses: list, do_loss_graph: bool,
                        snapshot_path: Optional[str], epoch_stride: Optional[int] = None):
        """ Callback used when fitting. """
        self._training_state["runtime"][model_type]["trained_epochs"] = epoch

        last_epoch = (epoch + 1) == total_epochs
        work_epoch = epoch_stride is None or ((epoch + 1) % epoch_stride) == 0 or last_epoch

        if work_epoch:
            if snapshot_path is not None:
                pathlib.Path(snapshot_path).mkdir(parents=True, exist_ok=True)
                snapshot_save_path = f"{snapshot_path}/{model_name}_{model_type}_{epoch}_snapshot.{self.SNAPSHOT_EXTENSION}"
                serialized = self.serialize()

                torch.save(serialized, snapshot_save_path)

            if do_loss_graph and not last_epoch:
                self._generate_loss_graph(losses=losses, save_path=f"{model_name}_{model_type}_{epoch}_loss")
                GraphSaver.save_graph(f"{model_name}_{model_type}_{epoch}_loss")


class BaseModel(Logger, Configurable, Serializable):
    """
    Base class for all prediction models.

    :param config: Application configuration.
    :param model_name: Name of the model, without spaces.
    """

    def __init__(self, config: Config, model_name: str):
        super().__init__(config=config)

        self.model_name = model_name
        self._featurizer = self.get_instance(DataFeaturizer)

    @classmethod
    def register_common_options(cls, parser: argparse.ArgumentParser):
        """
        Register common options for all models.

        :param parser: Parser for the model.
        """

        pass

    def _enable_feature_types(self,
                              config: DatasetConfig,
                              feature_types: Optional[List[str]] = None):
        """ Enable all feature types on given list. """

        feature_enablers = {
            "hist": lambda x: x.enable_hist(),
            "stat": lambda x: x.enable_stat(),
            "other": lambda x: x.enable_other(),
            "image": lambda x: x.enable_image(),
            "view": lambda x: x.enable_view(),
        }

        feature_types = feature_types or [ ]
        for feature_type in feature_types:
            feature_enablers[feature_type](config)

    def _enable_view_types(self,
                           config: DatasetConfig,
                           view_types: Optional[List[str]] = None):
        """ Enable all view types on given list. """

        view_types = view_types or [ ]
        for view_type in view_types:
            config.set(f"^view.{view_type}.\\.*$", True)

    def _enable_skeleton_types(self,
                               config: DatasetConfig,
                               skeleton_types: Optional[List[str]] = None):
        """ Enable all skeleton types on given list. """

        skeleton_types = skeleton_types or [ ]
        for skeleton_type in skeleton_types:
            config.set(f"^skeleton.{skeleton_type}.\\.*$", True)

    def _prepare_dataset_config(self,
                                split_name: Optional[List[str]] = None,
                                feature_types: Optional[List[str]] = None,
                                view_types: Optional[List[str]] = None,
                                skeleton_types: Optional[List[str]] = None,
                                data_loader: Optional[BaseDataLoader] = None,
                                data_configuration: dict = { },
                                tree_filter: Optional[Union[List[int], Dict[int, Set[int]]]] = None,
                                prediction_mode: bool = False,
                                feature_config: Optional[CacheDict] = None
                                ) -> DatasetConfig:
        """
        Generate initialized dataset configuration.

        :param split_name: Specification of which splits should be used.
        :param feature_types: Types of features to enable.
        :param view_types: Type of views to enable. Using "view" feature type enables all!
        :param skeleton_types: Type of data required for the skeleton/graph representation.
        :param data_loader: Optional data loader, containing the data. None leads
            to use of the default data loader.
        :param data_configuration: Optional initial config, containing "dataset" and
            "data" elements, gained from data_config() of existing TreeDataset.
        :param tree_filter: Filter for tree ids and views, useful for predictions.
        :param prediction_mode: Enable prediction mode to exclude scores and names.
        :param feature_config: Optional feature configuration overwriting the defaults.

        :return: Returns initialized dataset config, which can be further configured.
        """

        config = DatasetConfig(
            featurizer=self._featurizer,
            data_loader=data_loader,
            initial_config=data_configuration.get("dataset", None)
        )

        self._enable_feature_types(
            feature_types=feature_types,
            config=config
        )

        self._enable_view_types(
            view_types=view_types,
            config=config
        )

        self._enable_view_types(
            view_types=view_types,
            config=config
        )

        self._enable_skeleton_types(
            skeleton_types=skeleton_types,
            config=config
        )

        config.set_data_configuration(data_configuration.get("data", None))

        config.set_options(CacheDict({
            "featurizer": CacheDict({
                "splits": split_name,
                "tree_filter": tree_filter,
                "generate_names": not prediction_mode
            })
        }))
        config.enable_score(enabled=not prediction_mode)

        config.set_featurizer_option("verbose_loading", not prediction_mode)
        config.set_featurizer_option("resolve_indices", not prediction_mode)

        if feature_config is not None:
            config.set_options(CacheDict({ "featurizer": feature_config }))

        return config

    def train(self):
        """ Train this model on pre-configured data. """
        raise RuntimeError("Not implemented!")

    def predict(self, prediction: "Prediction", data: BaseDataLoader):
        """ Predict score for given prediction. """
        raise RuntimeError("Not implemented!")
