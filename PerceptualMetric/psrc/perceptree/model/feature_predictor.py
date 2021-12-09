# -*- coding: utf-8 -*-

"""
Model used for feature-only predictions.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
import sklearn.metrics as sklm

from perceptree.model.base import *
from perceptree.data.predict import Prediction


class DeepFeatureNetwork(BaseNetwork):
    """
    Simple predictor using dense layers.

    :param batch_size: Size of batches or dictionary specifying for each model.
    :param optimizer: Optimizer factory taking list of parameters or dictionary specifying for each model.
    :param loss: Loss factory taking optimizer or dictionary specifying for each model.
    :param lr_scheduler: Optional lr scheduler factory taking optimizer or dictionary specifying for each model.
    :param model_parameters: Parameters for each model.
    :param input_features: Optional nubmer of input features. If both input/output are specified, model
        is initialized immediately.
    :param output_features: Optional nubmer of output features. If both input/output are specified, model
        is initialized immediately.

    :usage:
        >>> network = DeepFeatureNetwork(
        >>>     epochs=2000,
        >>>     batch_size={ "feature": 256 },
        >>>     optimizer=(torch.optim.Adam, { "lr": 0.0001, "momentum": 0.95 }),
        >>>     loss={ "feature": tnn.MSELoss },
        >>>     lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        >>>     model_parameters={ }
        >>>)
    """

    def __init__(self, epochs: int, batch_size: Union[int, Dict[str, int]],
                 optimizer: Union[any, Dict[str, any]], loss: Union[any, Dict[str, any]],
                 lr_scheduler: Optional[Union[Any, Dict[str, any]]] = None,
                 model_parameters: Optional[Dict[str, Dict[str, any]]] = None,
                 input_features: Optional[int] = None, output_features: Optional[int] = None):
        super().__init__(cfg={
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "loss": loss,
            "lr_scheduler": lr_scheduler,
            "model_parameters": model_parameters
        }, model_names=[ "feature", "cmpb", "cmpd", "score" ])

        self._initialize()

        if input_features is not None and output_features is not None:
            self._initialize_empty_models(
                input_features=input_features,
                output_features=output_features
            )

    def deserialize(self, serialized: dict):
        """ Deserialize model from given dictionary. """

        if serialized["model"]["feature"]:
            serialized["cfg"]["input_features"] = serialized["model"]["feature"]["fcf.weight"].shape[1]
        if serialized["model"]["score"]:
            serialized["cfg"]["output_features"] = serialized["model"]["score"]["fc.weight"].shape[0]

        super().deserialize(serialized=serialized)

    class DNNFeature(tnn.Module):
        def __init__(self, input_features: int):
            super(DeepFeatureNetwork.DNNFeature, self).__init__()

            #sizes = [ 256, 64, 32 ]
            #sizes = [ 512, 256, 64, 32 ]
            sizes = [ 2048, 1024, 512, 256, 64, 32 ]
            #sizes = [ 8192 * 2, 8192, 4096, 2048, 1024, 512, 256, 64, 32 ]

            #sizes = [ 200, 100, 50, 25 ]
            #sizes = [ 100, 50, 25, 10 ]
            #sizes = [ 50, 25, 15, 10 ]
            #sizes = [ 200, 100, 50, 100, 150, 100, 50, 25 ]

            # TODO - Add BN?
            self.bn = tnn.BatchNorm1d(input_features)
            self.fcf = tnn.Linear(input_features, sizes[0], bias=True)
            self.fc_middle = tnn.ModuleList([
                tnn.Linear(sizes[idx - 1], sizes[idx], bias=True)
                for idx in range(1, len(sizes) - 1)
            ])
            self.bn_middle = tnn.ModuleList([
                tnn.BatchNorm1d(sizes[idx])
                for idx in range(1, len(sizes) - 1)
            ])
            self.dropout = tnn.Dropout(p=0.1)
            self.fcl = tnn.Linear(sizes[-2], sizes[-1], bias=True)

            #self.act = tnn.LeakyReLU(0.1)
            self.act = tnn.ReLU()
            #self.act = tnn.Sigmoid()

            self.output_features = sizes[-1]

        def forward(self, x):

            x = self.bn(x)
            x = self.fcf(x)
            x = self.act(x)

            for fc, bn in zip(self.fc_middle, self.bn_middle):
                x = fc(x)
                x = bn(x)
                x = self.act(x)

            x = self.dropout(x)
            x = self.fcl(x)
            x = self.act(x)

            return x

    class DNNCmp(tnn.Module):
        def __init__(self, input_features: int, output_features: int, differential: bool = True):
            super(DeepFeatureNetwork.DNNCmp, self).__init__()

            # Prepare output layer for comparison output.
            self.fc = tnn.Linear(input_features, output_features)
            self.act = tnn.LeakyReLU(0.1) if differential else tnn.Sigmoid()

        def forward(self, x):
            return self.act(self.fc(x))

    class DNNScore(tnn.Module):
        def __init__(self, input_features: int, output_features: int):
            super(DeepFeatureNetwork.DNNScore, self).__init__()

            # Prepare output layer for comparison output.
            self.fc = tnn.Linear(input_features, output_features)
            self.dropout = tnn.Dropout(p=0.1)
            self.act = tnn.LeakyReLU(0.1)
            #self.act = tnn.Sigmoid()
            #self.act = tnn.ReLU()

        def forward(self, x):
            x = self.fc(x)
            x = self.dropout(x)
            x = self.act(x)

            return x

    def _initialize(self):
        """ Initialize the networks to default state. """

        super()._initialize()

    def _initialize_empty_models(self, input_features: int, output_features: int):
        """ Initialize currently unavailable models. """

        device, cpu_device = self._get_devices()

        if self._model["feature"] is None:
            self._model["feature"] = DeepFeatureNetwork.DNNFeature(
                input_features=input_features,
                **self._cfg["model_parameters"].get("feature", { })
            ).train().to(device=device)
        if self._model["cmpb"] is None:
            self._model["cmpb"] = DeepFeatureNetwork.DNNCmp(
                # Getting features from both branches -> 2x.
                input_features=2 * self._model["feature"].output_features,
                output_features=output_features,
                **self._cfg["model_parameters"].get("cmpb", { })
            ).train().to(device=device)
        if self._model["cmpd"] is None:
            self._model["cmpd"] = DeepFeatureNetwork.DNNCmp(
                # Getting features from both branches -> 2x.
                input_features=2 * self._model["feature"].output_features,
                output_features=output_features,
                **self._cfg["model_parameters"].get("cmpd", { })
            ).train().to(device=device)
        if self._model["score"] is None:
            self._model["score"] = DeepFeatureNetwork.DNNScore(
                # Getting features from single branch only -> no multiplication.
                input_features=self._model["feature"].output_features,
                output_features=output_features,
                **self._cfg[ "model_parameters" ].get("score", { })
            ).train().to(device=device)

    def reset(self):
        """ Re-initialize the model to default state. """
        self._initialize()

    def fit(self, dataset: TreeDataset, model_type: str, *args, **kwargs) -> dict:
        """ Fit model_type model using provided parameters. """

        if model_type.startswith("cmp"):
            return self.fit_cmp(dataset=dataset, *args, estimator=model_type, **kwargs)
        elif model_type == "score":
            return self.fit_score(dataset=dataset, *args, **kwargs)

    def is_fitting_necessary(self, estimator: str, epochs: int) -> bool:
        """ Is fitting of given estimator stage necessary? """

        model_start_epoch = self._training_state["runtime"][estimator]["trained_epochs"] + 1

        return epochs > model_start_epoch or self._data_configuration is None or \
               ("score_config" if estimator == "score" else "cmp_config") not in self._data_configuration

    def fit_cmp(self, dataset: TreeDataset, estimator: str, epochs: int, fit_feature: bool = True, verbose: bool = True,
                snapshot_path: Optional[str] = None, save_epoch_stride: Optional[int] = None) -> dict:
        """ Fit pairwise comparison part of the model. """

        self._initialize_empty_models(
            input_features=dataset.inputs_shape()[-1],
            output_features=dataset.outputs_shape()[-1]
        )

        if self._data_configuration is None:
            self._data_configuration = { }
        self._data_configuration[ "cmp_config" ] = dataset.data_config

        self._fit(
            dataset=dataset, verbose=verbose,
            model_type=estimator, epochs=epochs,
            feature=self._model["feature"],
            estimate=self._model[estimator],
            fit_feature=fit_feature,
            epoch_callback=lambda e, te, l: self._epoch_callback(
                model_type=estimator, model_name="DeepNeuralNetworkPredictor",
                epoch=e, total_epochs=te, losses=l, do_loss_graph=verbose,
                snapshot_path=snapshot_path, epoch_stride=save_epoch_stride
            )
        )
        torch.cuda.empty_cache()

        return dataset.data_config

    def fit_score(self, dataset: TreeDataset, epochs: int, fit_feature: bool = True, verbose: bool = True,
                  snapshot_path: Optional[str] = None, save_epoch_stride: Optional[int] = None) -> dict:
        """ Fit score estimation part of the model. """

        self._initialize_empty_models(
            input_features=dataset.inputs_shape()[-1],
            output_features=dataset.outputs_shape()[-1]
        )

        if self._data_configuration is None:
            self._data_configuration = { }
        self._data_configuration[ "score_config" ] = dataset.data_config

        self._fit(
            dataset=dataset, verbose=verbose,
            model_type="score", epochs=epochs,
            feature=self._model["feature"],
            estimate=self._model["score"],
            fit_feature=fit_feature,
            epoch_callback=lambda e, te, l: self._epoch_callback(
                model_type="score", model_name="DeepNeuralNetworkPredictor",
                epoch=e, total_epochs=te, losses=l, do_loss_graph=verbose,
                snapshot_path=snapshot_path, epoch_stride=save_epoch_stride)
        )
        torch.cuda.empty_cache()

        return dataset.data_config

    def finalize(self, cmp_config: dict, score_config: dict):
        """ Finalize model fitting. """

        device, cpu_device = self._get_devices()

        self._model = {
            "feature": self._model["feature"].to(device=cpu_device),
            "cmpb": self._model["cmpb"].to(device=cpu_device),
            "cmpd": self._model["cmpd"].to(device=cpu_device),
            "score": self._model["score"].to(device=cpu_device)
        }

        for model in self._model.values():
            model.eval()

        self._data_configuration = {
            "cmp_config": cmp_config,
            "score_config": score_config
        }
        self._training_state["finalized"] = True

    def _fit(self, dataset: TreeDataset, verbose: bool,
             model_type: str, epochs: int,
             feature: tnn.Module, estimate: tnn.Module,
             fit_feature: bool,
             epoch_callback: Optional[callable]):

        """ Fit model_type model using provided parameters. """

        # Prepare model properties:
        model_batch_size = self._get_option_for_type("batch_size", model_type)
        model_epochs = epochs
        model_start_epoch = self._training_state["runtime"][model_type]["trained_epochs"] + 1

        # Prepare optional validation data:
        if "train" in dataset.splits and "valid" in dataset.splits:
            do_validation = True
            dataset.set_current_splits(splits="train")
            valid_dataset = dataset.duplicate_for_splits(splits="valid")
            valid_data = td.DataLoader(dataset=valid_dataset, batch_size=model_batch_size // 2, shuffle=True)
        else:
            do_validation = False

        # Prepare the training data:
        data = td.DataLoader(dataset=dataset, batch_size=model_batch_size, shuffle=True)

        # Prepare model training utilities.
        parameters = list(feature.parameters()) if fit_feature else [ ] + \
                     list(estimate.parameters())
        optimizer = self._get_runtime_option_for_type("optimizer", model_type, parameters)
        lr_scheduler = self._get_runtime_option_for_type("lr_scheduler", model_type, optimizer)
        loss_fun = self._get_runtime_option_for_type("loss", model_type)

        # Prepare loss calculation helper:
        def calculate_loss(x, y, dev):
            if not isinstance(x, list):
                x = [ x ]

            batch_size = x[0].shape[0]
            feature_size = x[0].shape[1]

            x = torch.cat(x, 1).reshape((-1, feature_size))
            features = feature(x.float().to(device=dev))
            features = features.reshape((batch_size, -1,) + features.shape[2:])

            prediction = estimate(features)
            loss_grad = loss_fun(prediction, y.float().to(device=dev))

            return loss_grad

        # Prepare for training:
        if "losses" not in self._training_state:
            self._training_state["losses"] = { }
        if model_type not in self._training_state["losses"]:
            self._training_state["losses"][model_type] = [ ]
        device, cpu_device = self._get_devices()

        for epoch in range(model_start_epoch, model_epochs):
            epoch_losses = [ ]

            if verbose:
                self.__l.info(f"Epoch #{epoch}:")

            fitting_bar = FittingBar(max=len(data))

            feature.train()
            estimate.train()

            for batch_x, batch_y in data:
                optimizer.zero_grad()

                loss = calculate_loss(x=batch_x, y=batch_y, dev=device)
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                self._training_state["losses"][model_type].append((epoch, loss_value, "train"))
                epoch_losses.append(loss_value)

                mean_loss_value = np.mean(epoch_losses)

                if verbose:
                    fitting_bar.loss = f"{loss_value:.5f}|{mean_loss_value:.5f}"
                    fitting_bar.next(1)

            if verbose:
                fitting_bar.finish()

            epoch_loss = np.mean(epoch_losses)
            if verbose:
                self.__l.info(f"Epoch #{epoch} TLoss : {epoch_loss}")

            if do_validation:
                valid_losses = [ ]

                # Clean the cache to make space for validation data.
                loss = None
                torch.cuda.empty_cache()

                if verbose:
                    self.__l.info(f"Epoch #{epoch} Validation:")

                validation_bar = FittingBar(max=len(valid_data))

                with torch.no_grad():
                    for batch_x, batch_y in valid_data:
                        loss = calculate_loss(x=batch_x, y=batch_y, dev=device)

                        loss_value = loss.item()
                        self._training_state["losses"][model_type].append((epoch, loss_value, "valid"))
                        valid_losses.append(loss_value)

                        mean_loss_value = np.mean(valid_losses)

                        if verbose:
                            validation_bar.loss = f"{loss_value:.5f}|{mean_loss_value:.5f}"
                            validation_bar.next(1)

                if verbose:
                    validation_bar.finish()

                # Clean the cache to make space for training data.
                loss = None
                torch.cuda.empty_cache()

                epoch_valid_loss = np.mean(valid_losses)
                if verbose:
                    self.__l.info(f"Epoch #{epoch} VLoss : {epoch_valid_loss}")

            # Use last mean loss value calculated - either train or validation.
            if lr_scheduler is not None:
                lr_scheduler.step(mean_loss_value)

            if epoch_callback is not None:
                epoch_callback(epoch, model_epochs, self._training_state["losses"][model_type])

        if verbose:
            self._generate_loss_graph(losses=self._training_state["losses"][model_type],
                                      save_path=f"FeaturePredictor_{model_type}_loss_final")
            GraphSaver.save_graph(f"FeaturePredictor_{model_type}_loss_final")

    def predict(self, inputs: Union[np.array, TreeDataset]) -> np.array:
        model_inputs = torch.from_numpy(inputs.inputs() if isinstance(inputs, TreeDataset) else inputs).float()
        if len(model_inputs) > 0:
            with torch.no_grad():
                return self._model["score"](self._model["feature"](model_inputs)).numpy()
        else:
            return np.array([ ])

    def cross_validate(self, dataset: TreeDataset, verbose: bool) -> np.array:
        raise RuntimeError("Cross validation for DNN not implemented!")


class ForestFeatureNetwork(BaseNetwork):
    """
    Simple predictor using RandomForestRegressor.

    :param model_parameters: Parameters for each model.

    :usage:
        >>> network = ForestFeatureNetwork(
        >>>     model_parameters={
        >>>         "n_estimators": 1000,
        >>>         "bootstrap": False,
        >>>         "max_depth": None,
        >>>         "max_samples": None,
        >>>         "max_features": None,
        >>>         "criterion": "mae",
        >>>         "random_state": None,
        >>>         "n_jobs": -1,
        >>>      }
        >>>)
    """

    def __init__(self, model_parameters: Optional[ Dict[ str, Dict[ str, any ] ] ] = None):
        super().__init__({
            "model_parameters": model_parameters
        }, model_names=[ "feature" ])

        self._initialize()

    def fit(self, dataset: TreeDataset, model_type: str, verbose: bool):
        """ Fit model_type model using provided parameters. """

        forest = self.create_forest(verbose=verbose)
        forest.fit(X=dataset.inputs(), y=dataset.outputs())

        self._model["feature"] = forest
        self._data_configuration = dataset.data_config

    def predict(self, inputs: Union[np.array, TreeDataset]) -> np.array:
        if isinstance(inputs, TreeDataset):
            return self._model["feature"].predict(inputs.inputs())
        else:
            return self._model["feature"].predict(inputs)

    def cross_validate(self, dataset: TreeDataset, groups: int, verbose: bool) -> np.array:
        forest = RandomForestRegressor(
            *self._model_args, **self._model_kwargs,
            verbose=verbose
        )
        return cross_val_score(
            estimator=forest, X=dataset.inputs(), y=dataset.outputs(),
            n_jobs=-1, verbose=verbose,
            cv=groups
        )

    def create_forest(self, verbose: bool = False) -> RandomForestRegressor:
        """ Create model of the underlying type. """
        return RandomForestRegressor(
            **self._cfg["model_parameters"],
            verbose=verbose
        )


class LinearFeatureNetwork(BaseNetwork):
    """
    Simple predictor using LinearRegression.

    :param model_parameters: Parameters for each model.

    :usage:
        >>> network = LinearFeatureNetwork(
        >>>     model_parameters={
        >>>         "fit_intercept": True,
        >>>         "normalize": False,
        >>>         "positive": False,
        >>>         "n_jobs": -1,
        >>>      }
        >>>)
    """

    def __init__(self, model_parameters: Optional[ Dict[ str, Dict[ str, any ] ] ] = None):
        super().__init__({
            "model_parameters": model_parameters
        }, model_names=[ "feature" ])

        self._initialize()

    def fit(self, dataset: TreeDataset, model_type: str, verbose: bool):
        """ Fit model_type model using provided parameters. """

        model = LinearRegression(**self._cfg["model_parameters"])
        model.fit(X=dataset.inputs(), y=dataset.outputs())

        self._model["feature"] = model
        self._data_configuration = dataset.data_config

    def predict(self, inputs: Union[np.array, TreeDataset]) -> np.array:
        if isinstance(inputs, TreeDataset):
            return self._model["feature"].predict(inputs.inputs())
        else:
            return self._model["feature"].predict(inputs)


class LassoFeatureNetwork(BaseNetwork):
    """
    Simple predictor using Lasso.

    :param model_parameters: Parameters for each model.

    :usage:
        >>> network = LassoFeatureNetwork(
        >>>     model_parameters={
        >>>         "alpha": 1.0,
        >>>         "max_iter": 1000,
        >>>         "tol": 1e-4,
        >>>         "fit_intercept": True,
        >>>         "normalize": False,
        >>>         "precompute": False,
        >>>         "selection": "cyclic",
        >>>         "random_state": None
        >>>      }
        >>>)
    """

    def __init__(self, model_parameters: Optional[ Dict[ str, Dict[ str, any ] ] ] = None):
        super().__init__({
            "model_parameters": model_parameters
        }, model_names=[ "feature" ])

        self._initialize()

    def fit(self, dataset: TreeDataset, model_type: str, verbose: bool):
        """ Fit model_type model using provided parameters. """

        model = Lasso(**self._cfg["model_parameters"])
        model.fit(X=dataset.inputs(), y=dataset.outputs())

        self._model["feature"] = model
        self._data_configuration = dataset.data_config

    def predict(self, inputs: Union[np.array, TreeDataset]) -> np.array:
        if isinstance(inputs, TreeDataset):
            return self._model["feature"].predict(inputs.inputs())
        else:
            return self._model["feature"].predict(inputs)


class RidgeFeatureNetwork(BaseNetwork):
    """
    Simple predictor using Ridge.

    :param model_parameters: Parameters for each model.

    :usage:
        >>> network = RidgeFeatureNetwork(
        >>>     model_parameters={
        >>>         "alpha": 1.0,
        >>>         "max_iter": None,
        >>>         "tol": 1e-3,
        >>>         "fit_intercept": True,
        >>>         "normalize": False,
        >>>         "solver": "auto",
        >>>         "random_state": None
        >>>      }
        >>>)
    """

    def __init__(self, model_parameters: Optional[ Dict[ str, Dict[ str, any ] ] ] = None):
        super().__init__({
            "model_parameters": model_parameters
        }, model_names=[ "feature" ])

        self._initialize()

    def fit(self, dataset: TreeDataset, model_type: str, verbose: bool):
        """ Fit model_type model using provided parameters. """

        model = Ridge(**self._cfg["model_parameters"])
        model.fit(X=dataset.inputs(), y=dataset.outputs())

        self._model["feature"] = model
        self._data_configuration = dataset.data_config

    def predict(self, inputs: Union[np.array, TreeDataset]) -> np.array:
        if isinstance(inputs, TreeDataset):
            return self._model["feature"].predict(inputs.inputs())
        else:
            return self._model["feature"].predict(inputs)


class ElasticFeatureNetwork(BaseNetwork):
    """
    Simple predictor using ElasticNet.

    :param model_parameters: Parameters for each model.

    :usage:
        >>> network = ElasticFeatureNetwork(
        >>>     model_parameters={
        >>>         "alpha": 1.0,
        >>>         "l1_ratio": 0.5,
        >>>         "max_iter": 1000,
        >>>         "tol": 1e-4,
        >>>         "fit_intercept": True,
        >>>         "normalize": False,
        >>>         "precompute": False,
        >>>         "positive": False,
        >>>         "selection": "cyclic",
        >>>         "random_state": None
        >>>      }
        >>>)
    """

    def __init__(self, model_parameters: Optional[ Dict[ str, Dict[ str, any ] ] ] = None):
        super().__init__({
            "model_parameters": model_parameters
        }, model_names=[ "feature" ])

        self._initialize()

    def fit(self, dataset: TreeDataset, model_type: str, verbose: bool):
        """ Fit model_type model using provided parameters. """

        model = ElasticNet(**self._cfg["model_parameters"])
        model.fit(X=dataset.inputs(), y=dataset.outputs())

        self._model["feature"] = model
        self._data_configuration = dataset.data_config

    def predict(self, inputs: Union[np.array, TreeDataset]) -> np.array:
        if isinstance(inputs, TreeDataset):
            return self._model["feature"].predict(inputs.inputs())
        else:
            return self._model["feature"].predict(inputs)


class LarsFeatureNetwork(BaseNetwork):
    """
    Simple predictor using Lars.

    :param model_parameters: Parameters for each model.

    :usage:
        >>> network = LarsFeatureNetwork(
        >>>     model_parameters={
        >>>         "fit_intercept": True,
        >>>         "normalize": True,
        >>>         "precompute": "auto",
        >>>         "n_nonzero_coefs": 500,
        >>>         "eps": np.finfo(float).eps,
        >>>         "fit_path": True,
        >>>         "jitter": None,
        >>>         "random_state": None
        >>>      }
        >>>)
    """

    def __init__(self, model_parameters: Optional[ Dict[ str, Dict[ str, any ] ] ] = None):
        super().__init__({
            "model_parameters": model_parameters
        }, model_names=[ "feature" ])

        self._initialize()

    def fit(self, dataset: TreeDataset, model_type: str, verbose: bool):
        """ Fit model_type model using provided parameters. """

        model = Lars(**self._cfg["model_parameters"], verbose=verbose)
        model.fit(X=dataset.inputs(), y=dataset.outputs())

        self._model["feature"] = model
        self._data_configuration = dataset.data_config

    def predict(self, inputs: Union[np.array, TreeDataset]) -> np.array:
        if isinstance(inputs, TreeDataset):
            return self._model["feature"].predict(inputs.inputs())
        else:
            return self._model["feature"].predict(inputs)


class LassoLarsFeatureNetwork(BaseNetwork):
    """
    Simple predictor using LassoLars.

    :param model_parameters: Parameters for each model.

    :usage:
        >>> network = LassoLarsFeatureNetwork(
        >>>     model_parameters={
        >>>         "alpha": 1.0,
        >>>         "max_iter": 500,
        >>>         "eps": np.finfo(float).eps,
        >>>         "fit_intercept": True,
        >>>         "fit_path": True,
        >>>         "normalize": True,
        >>>         "precompute": "auto",
        >>>         "positive": False,
        >>>         "jitter": None,
        >>>         "random_state": None
        >>>      }
        >>>)
    """

    def __init__(self, model_parameters: Optional[ Dict[ str, Dict[ str, any ] ] ] = None):
        super().__init__({
            "model_parameters": model_parameters
        }, model_names=[ "feature" ])

        self._initialize()

    def fit(self, dataset: TreeDataset, model_type: str, verbose: bool):
        """ Fit model_type model using provided parameters. """

        model = LassoLars(**self._cfg["model_parameters"], verbose=verbose)
        model.fit(X=dataset.inputs(), y=dataset.outputs())

        self._model["feature"] = model
        self._data_configuration = dataset.data_config

    def predict(self, inputs: Union[np.array, TreeDataset]) -> np.array:
        if isinstance(inputs, TreeDataset):
            return self._model["feature"].predict(inputs.inputs())
        else:
            return self._model["feature"].predict(inputs)


class OrthoMatchingFeatureNetwork(BaseNetwork):
    """
    Simple predictor using OrthogonalMatchingPursuit.

    :param model_parameters: Parameters for each model.

    :usage:
        >>> network = OrthoMatchingFeatureNetwork(
        >>>     model_parameters={
        >>>         "fit_intercept": True,
        >>>         "normalize": True,
        >>>         "precompute": "auto",
        >>>         "n_nonzero_coefs": 500
        >>>      }
        >>>)
    """

    def __init__(self, model_parameters: Optional[ Dict[ str, Dict[ str, any ] ] ] = None):
        super().__init__({
            "model_parameters": model_parameters
        }, model_names=[ "feature" ])

        self._initialize()

    def fit(self, dataset: TreeDataset, model_type: str, verbose: bool):
        """ Fit model_type model using provided parameters. """

        model = OrthogonalMatchingPursuit(**self._cfg["model_parameters"])
        model.fit(X=dataset.inputs(), y=dataset.outputs())

        self._model["feature"] = model
        self._data_configuration = dataset.data_config

    def predict(self, inputs: Union[np.array, TreeDataset]) -> np.array:
        if isinstance(inputs, TreeDataset):
            return self._model["feature"].predict(inputs.inputs())
        else:
            return self._model["feature"].predict(inputs)


class ARDFeatureNetwork(BaseNetwork):
    """
    Simple predictor using ARDRegression.

    :param model_parameters: Parameters for each model.

    :usage:
        >>> network = ARDFeatureNetwork(
        >>>     model_parameters={
        >>>         "n_iter": 300,
        >>>         "tol": 1e-3,
        >>>         "alpha_1": 1e-6,
        >>>         "alpha_2": 1e-6,
        >>>         "lambda_1": 1e-6,
        >>>         "lambda_2": 1e-6,
        >>>         "threshold_lambda": 10000.0,
        >>>         "fit_intercept": True,
        >>>         "compute_score": False,
        >>>         "normalize": False,
        >>>      }
        >>>)
    """

    def __init__(self, model_parameters: Optional[ Dict[ str, Dict[ str, any ] ] ] = None):
        super().__init__({
            "model_parameters": model_parameters
        }, model_names=[ "feature" ])

        self._initialize()

    def fit(self, dataset: TreeDataset, model_type: str, verbose: bool):
        """ Fit model_type model using provided parameters. """

        model = ARDRegression(**self._cfg["model_parameters"], verbose=verbose)
        model.fit(X=dataset.inputs(), y=dataset.outputs())

        self._model["feature"] = model
        self._data_configuration = dataset.data_config

    def predict(self, inputs: Union[np.array, TreeDataset]) -> np.array:
        if isinstance(inputs, TreeDataset):
            return self._model["feature"].predict(inputs.inputs())
        else:
            return self._model["feature"].predict(inputs)


class BayesRidgeFeatureNetwork(BaseNetwork):
    """
    Simple predictor using BayesianRidge.

    :param model_parameters: Parameters for each model.

    :usage:
        >>> network = BayesRidgeFeatureNetwork(
        >>>     model_parameters={
        >>>         "n_iter": 300,
        >>>         "tol": 1e-3,
        >>>         "alpha_1": 1e-6,
        >>>         "alpha_2": 1e-6,
        >>>         "lambda_1": 1e-6,
        >>>         "lambda_2": 1e-6,
        >>>         "alpha_init": None,
        >>>         "lambda_init": None,
        >>>         "fit_intercept": True,
        >>>         "compute_score": False,
        >>>         "normalize": False,
        >>>      }
        >>>)
    """

    def __init__(self, model_parameters: Optional[ Dict[ str, Dict[ str, any ] ] ] = None):
        super().__init__({
            "model_parameters": model_parameters
        }, model_names=[ "feature" ])

        self._initialize()

    def fit(self, dataset: TreeDataset, model_type: str, verbose: bool):
        """ Fit model_type model using provided parameters. """

        model = BayesianRidge(**self._cfg["model_parameters"], verbose=verbose)
        model.fit(X=dataset.inputs(), y=dataset.outputs())

        self._model["feature"] = model
        self._data_configuration = dataset.data_config

    def predict(self, inputs: Union[np.array, TreeDataset]) -> np.array:
        if isinstance(inputs, TreeDataset):
            return self._model["feature"].predict(inputs.inputs())
        else:
            return self._model["feature"].predict(inputs)


class FeaturePredictor(BaseModel):
    """
    Model used for feature-only predictions.

    :param config: Application configuration.
    """

    MODEL_NAME = "fop"
    """ Name of this prediction model. """
    CATEGORY_NAME = "FeaturePredictor"
    """ Category as visible to the user. """

    def __init__(self, config: Config):
        super().__init__(config, self.MODEL_NAME)

        self._model = None

    @classmethod
    def register_options(cls, parser: Config.Parser):
        """ Register configuration options for this class. """

        super().register_common_options(parser)

        option_name = cls._add_config_parameter("feature_types")
        parser.add_argument("--feature-types",
                            action="store",
                            default=[ ], type=parse_list_string(typ=str, sep=","),
                            metavar=("STR,LIST"),
                            dest=option_name,
                            help="List of feature types to use from { stat, image, hist, other }. "
                                 "Use empty list (',') to use all features.")

        option_name = cls._add_config_parameter("model_type")
        parser.add_argument("--model-type",
                            action="store",
                            default="forest", type=str,
                            metavar=("<forest|linear|lasso|ridge|elastic|lars|lassolars|ortho|ard|bayes|dnn>"),
                            dest=option_name,
                            help="Type of model to use - forest, dnn or other model.")

        option_name = cls._add_config_parameter("batch_size")
        parser.add_argument("--batch-size",
                            action="store",
                            default=64, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Mini-batch size used for training.")

        option_name = cls._add_config_parameter("score_epochs")
        parser.add_argument("--score-epochs",
                            action="store",
                            default=2000, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Number of epochs to train the score network for.")

        option_name = cls._add_config_parameter("feature_buckets")
        parser.add_argument("--feature-buckets",
                            action="store",
                            default=8, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Number of buckets used in histogram features.")

        option_name = cls._add_config_parameter("feature_resolution")
        parser.add_argument("--feature-resolution",
                            action="store",
                            default=32, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Resolution of image features.")

        option_name = cls._add_config_parameter("feature_normalize")
        parser.add_argument("--feature-normalize",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Normalize feature values?")

        option_name = cls._add_config_parameter("feature_standardize")
        parser.add_argument("--feature-standardize",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Standardize feature values?")

        option_name = cls._add_config_parameter("use_view_scores")
        parser.add_argument("--use-view-scores",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Use view scores during training?")

        option_name = cls._add_config_parameter("use_tree_variants")
        parser.add_argument("--use-tree-variants",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Use tree variants during training?")

        option_name = cls._add_config_parameter("use_view_variants")
        parser.add_argument("--use-view-variants",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Use view variants during training?")

        option_name = cls._add_config_parameter("pre_generate_variants")
        parser.add_argument("--pre-generate-variants",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Pre-generate training samples for all variants?")

        option_name = cls._add_config_parameter("binary_pretraining")
        parser.add_argument("--binary-pretraining",
                            action="store",
                            default=True, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Use binary data in the pretraining phase?")

        option_name = cls._add_config_parameter("binary_pretraining_ptg")
        parser.add_argument("--binary-pretraining-ptg",
                            action="store",
                            default=None, type=float,
                            metavar=("<0.0, 1.0>"),
                            dest=option_name,
                            help="Percentage of records to use in binary pre-training.")

        option_name = cls._add_config_parameter("binary_pretraining_epochs")
        parser.add_argument("--binary-pretraining-epochs",
                            action="store",
                            default=50, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Number of epochs in the binary pre-training phase")

        option_name = cls._add_config_parameter("differential_pretraining")
        parser.add_argument("--differential-pretraining",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Use score differences in the pretraining phase?")

        option_name = cls._add_config_parameter("differential_pretraining_ptg")
        parser.add_argument("--differential-pretraining-ptg",
                            action="store",
                            default=None, type=float,
                            metavar=("<0.0, 1.0>"),
                            dest=option_name,
                            help="Percentage of records to use in differential pre-training.")

        option_name = cls._add_config_parameter("differential_pretraining_epochs")
        parser.add_argument("--differential-pretraining-epochs",
                            action="store",
                            default=50, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Number of epochs in the differential pre-training phase")

        option_name = cls._add_config_parameter("forest_estimators")
        parser.add_argument("--forest-estimators",
                            action="store",
                            default=1000, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Number of estimators in the forest.")

        option_name = cls._add_config_parameter("verbose_fitting")
        parser.add_argument("--verbose-fitting",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Print verbose fitting information?")

        option_name = cls._add_config_parameter("do_cross_validate")
        parser.add_argument("--do-cross-validate",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Perform cross validation?")

        option_name = cls._add_config_parameter("cross_validate_groups")
        parser.add_argument("--cross-validate-groups",
                            action="store",
                            default=5, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Number of groups to use in cross-validation.")

        option_name = cls._add_config_parameter("do_fit")
        parser.add_argument("--do-fit",
                            action="store",
                            default=True, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Fit prediction model?")

        option_name = cls._add_config_parameter("fit_on")
        parser.add_argument("--fit-on",
                            action="store",
                            default=[ "train" ], type=parse_list_string(typ=str, sep=","),
                            metavar=("STR,LIST"),
                            dest=option_name,
                            help="Perform fitting on provided splits from { train, valid, test }. "
                                 "Use empty list (',') to use all splits.")

        option_name = cls._add_config_parameter("do_test")
        parser.add_argument("--do-test",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Test fitted model on test split?")

        option_name = cls._add_config_parameter("do_feature_importance")
        parser.add_argument("--do-feature-importance",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Test feature importance and report results?")

        option_name = cls._add_config_parameter("feature_importance_on")
        parser.add_argument("--feature-importance-on",
                            action="store",
                            default=[ "train" ], type=parse_list_string(typ=str, sep=","),
                            metavar=("STR,LIST"),
                            dest=option_name,
                            help="Perform feature importance testing on provided splits from { train, valid, test }. "
                                 "Use empty list (',') to use all splits.")

        option_name = cls._add_config_parameter("feature_importance_runs")
        parser.add_argument("--feature-importance-runs",
                            action="store",
                            default=5, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Number of runs to perform for feature importance scoring.")

        option_name = cls._add_config_parameter("feature_importance_top")
        parser.add_argument("--feature-importance-top",
                            action="store",
                            default=20, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Number of most important features to display details for.")

        option_name = cls._add_config_parameter("feature_importance_top_sort")
        parser.add_argument("--feature-importance-top-sort",
                            action="store",
                            default=True, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Sort the top features by their importance before displaying them?")

        option_name = cls._add_config_parameter("feature_importance_bottom")
        parser.add_argument("--feature-importance-bottom",
                            action="store",
                            default=20, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Number of least important features to display details for.")

        option_name = cls._add_config_parameter("feature_importance_bottom_sort")
        parser.add_argument("--feature-importance-bottom-sort",
                            action="store",
                            default=True, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Sort the worst features by their importance before displaying them?")

        option_name = cls._add_config_parameter("feature_importance_aggregate_category")
        parser.add_argument("--feature-importance-aggregate-category",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Aggregate feature names, including their category?")

        option_name = cls._add_config_parameter("feature_importance_aggregate_before")
        parser.add_argument("--feature-importance-aggregate-before",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Aggregate feature importance before mean, based on their base meaning?")

        option_name = cls._add_config_parameter("feature_importance_aggregate_after")
        parser.add_argument("--feature-importance-aggregate-after",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Aggregate feature importance after mean, based on their base meaning?")

        option_name = cls._add_config_parameter("feature_importance_gini")
        parser.add_argument("--feature-importance-gini",
                            action="store",
                            default=True, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Perform feature importance analysis by using Gini impurity?")

        option_name = cls._add_config_parameter("feature_importance_permutation")
        parser.add_argument("--feature-importance-permutation",
                            action="store",
                            default=True, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Perform feature importance analysis by using permutational analysis?")

        option_name = cls._add_config_parameter("feature_importance_permutation_repeats")
        parser.add_argument("--feature-importance-permutation-repeats",
                            action="store",
                            default=2, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Number of repeats performed for the permutation importance analysis.")

        option_name = cls._add_config_parameter("feature_importance_export")
        parser.add_argument("--feature-importance-export",
                            action="store",
                            default=None, type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Export feature importance results into given path.")

        option_name = cls._add_config_parameter("snapshot_path")
        parser.add_argument("--snapshot-path",
                            action="store",
                            default=None, type=str,
                            metavar=("PATH_TO_SNAPSHOTS"),
                            dest=option_name,
                            help="Save model snapshots to given directory, not applicable to most model types.")

        option_name = cls._add_config_parameter("load_pretrained")
        parser.add_argument("--load-pretrained",
                            action="store",
                            default=None, type=str,
                            metavar=("PATH_TO_PTH"),
                            dest=option_name,
                            help="Load pre-trained snapshot from given path, not applicable to mos model types.")

        option_name = cls._add_config_parameter("pretrained_continue")
        parser.add_argument("--pretrained-continue",
                            action="store",
                            default=None, type=str,
                            metavar=("PATH_TO_BASE_DIR"),
                            dest=option_name,
                            help="Find latest compatible snapshot in given path and continue training, not "
                                 "applicable to most model types.")

    def serialize(self) -> dict:
        """ Get data to be serialized for this model. """
        return {
            "model": self._model.serialize(),
            "model_type": type(self._model),
            "config": self.serialize_config()
        }

    def deserialize(self, data: dict):
        """ Deserialize data for this model. """
        self._model = data["model_type"].from_serialized(data["model"])
        self.deserialize_config(data["config"])

    def _prepare_model(self, type: str) -> BaseNetwork:
        """ Prepare requested model - "forest" or "dnn". """
        if type == "forest":
            self.__l.info("Using RandomForestRegressor model...")
            return ForestFeatureNetwork(
                model_parameters={
                    "n_estimators": self.c.forest_estimators,
                    "bootstrap": False,
                    "max_depth": None,
                    "max_samples": None,
                    "max_features": None,
                    #"criterion": "mae",
                    "criterion": "mse",
                    "random_state": None,
                    "n_jobs": -1,
                }
            )
        elif type == "linear":
            self.__l.info("Using LinearRegression model...")
            return LinearFeatureNetwork(
                model_parameters={
                    "fit_intercept": True,
                    "normalize": False,
                    "positive": False,
                    "n_jobs": -1,
                }
            )
        elif type == "lasso":
            self.__l.info("Using Lasso model...")
            return LassoFeatureNetwork(
                model_parameters={
                    "alpha": 1.0,
                    "max_iter": 1000,
                    "tol": 1e-4,
                    "fit_intercept": True,
                    "normalize": False,
                    "precompute": False,
                    "selection": "cyclic",
                    "random_state": None,
                }
            )
        elif type == "ridge":
            self.__l.info("Using Ridge model...")
            return RidgeFeatureNetwork(
                model_parameters={
                    "alpha": 1.0,
                    "max_iter": None,
                    "tol": 1e-3,
                    "fit_intercept": True,
                    "normalize": False,
                    "solver": "auto",
                    "random_state": None,
                }
            )
        elif type == "elastic":
            self.__l.info("Using ElasticNet model...")
            return ElasticFeatureNetwork(
                model_parameters={
                    "alpha": 1.0,
                    "l1_ratio": 0.5,
                    "max_iter": 1000,
                    "tol": 1e-4,
                    "fit_intercept": True,
                    "normalize": False,
                    "precompute": False,
                    "positive": False,
                    "selection": "cyclic",
                    "random_state": None,
                }
            )
        elif type == "lars":
            self.__l.info("Using Lars model...")
            return LarsFeatureNetwork(
                model_parameters={
                    "fit_intercept": True,
                    "normalize": True,
                    "precompute": "auto",
                    "n_nonzero_coefs": 128,
                    #"eps": np.finfo(float).eps,
                    "eps": 0.00001,
                    "fit_path": True,
                    "jitter": None,
                    "random_state": None,
                }
            )
        elif type == "lassolars":
            self.__l.info("Using LassoLars model...")
            return LassoLarsFeatureNetwork(
                model_parameters={
                    "alpha": 1.0,
                    "max_iter": 500,
                    "eps": np.finfo(float).eps,
                    "fit_intercept": True,
                    "fit_path": True,
                    "normalize": True,
                    "precompute": "auto",
                    "positive": False,
                    "jitter": None,
                    "random_state": None,
                }
            )
        elif type == "ortho":
            self.__l.info("Using OrthogonalMatchingPursuit model...")
            return OrthoMatchingFeatureNetwork(
                model_parameters={
                    "fit_intercept": True,
                    "normalize": True,
                    "precompute": "auto",
                    "n_nonzero_coefs": 128,
                }
            )
        elif type == "ard":
            self.__l.info("Using ARDRegression model...")
            return ARDFeatureNetwork(
                model_parameters={
                    "n_iter": 300,
                    "tol": 1e-3,
                    "alpha_1": 1e-6,
                    "alpha_2": 1e-6,
                    "lambda_1": 1e-6,
                    "lambda_2": 1e-6,
                    "threshold_lambda": 10000.0,
                    "fit_intercept": True,
                    "compute_score": False,
                    "normalize": False,
                }
            )
        elif type == "bayes":
            self.__l.info("Using BayesianRidge model...")
            return BayesRidgeFeatureNetwork(
                model_parameters={
                    "n_iter": 300,
                    "tol": 1e-3,
                    "alpha_1": 1e-6,
                    "alpha_2": 1e-6,
                    "lambda_1": 1e-6,
                    "lambda_2": 1e-6,
                    "alpha_init": None,
                    "lambda_init": None,
                    "fit_intercept": True,
                    "compute_score": False,
                    "normalize": False,
                }
            )
        elif type == "dnn":
            self.__l.info("Using DeepNeuralNetwork model...")
            return DeepFeatureNetwork(
                epochs=self.c.score_epochs,
                batch_size={ "cmpb": self.c.batch_size, "cmpd": self.c.batch_size, "score": self.c.batch_size },
                optimizer=(torch.optim.Adam, { "lr": 0.002, "weight_decay": 0.01, "amsgrad": True }),
                loss={ "cmpb": tnn.BCELoss, "cmpd": tnn.MSELoss, "score": tnn.MSELoss },
                lr_scheduler=(torch.optim.lr_scheduler.ReduceLROnPlateau,
                              { "patience": 10, "verbose": self.c.verbose_fitting }),
                model_parameters={
                    "cmpb": { "differential": False },
                    "cmpd": { "differential": True },
                }
            )
        else:
            raise RuntimeError(f"Unknown model requested: \"{type}\"!")

    def _get_current_feature_config(self) -> CacheDict:
        """ Get current configuration of the featurizer. """

        return CacheDict({
            "quant_start": 0.001,
            "quant_end": 0.999,
            "total_buckets": self.c.feature_buckets,
            "normalize_features": self.c.feature_normalize,
            "standardize_features": self.c.feature_standardize,
            "buckets_from_split": False,
            "image_resolution": self.c.feature_resolution,
            "image_interpolation": None,
        })

    def _prepare_data(self,
                      split_name: Optional[List[str]] = None,
                      feature_types: Optional[List[str]] = None,
                      data_loader: Optional[BaseDataLoader] = None,
                      data_configuration: dict = { },
                      tree_filter: Optional[Union[List[int], Dict[int, Set[int]]]] = None,
                      comparison_mode: bool = False,
                      comparison_score_difference: bool = False,
                      comparison_sample_ptg: Optional[float] = None,
                      use_view_scores: bool = False,
                      comparison_view_scores: bool = False,
                      use_tree_variants: bool = False,
                      use_view_variants: bool = False,
                      pre_generate_variants: bool = False,
                      prediction_mode: bool = False,
                      ) -> TreeDataset:
        """ Prepare data for given split, return generated dataset. """

        self.__l.info("Preparing dataset...")

        config = self._prepare_dataset_config(
            split_name=split_name,
            feature_types=feature_types,
            data_loader=data_loader,
            data_configuration=data_configuration,
            tree_filter=tree_filter,
            prediction_mode=prediction_mode,
            feature_config=self._get_current_feature_config()
        )

        config.set_featurizer_option("comparison_mode", comparison_mode)
        config.set_featurizer_option("comparison_score_difference", comparison_score_difference)
        config.set_featurizer_option("comparison_sample_ptg", comparison_sample_ptg)
        config.set_featurizer_option("comparison_view_scores", comparison_view_scores)
        config.set_featurizer_option("use_view_scores", use_view_scores)
        config.set_featurizer_option("use_tree_variants", use_tree_variants)
        config.set_featurizer_option("use_view_variants", use_view_variants)
        config.set_featurizer_option("pre_generate_variants", pre_generate_variants)

        return TreeDataset(config=config)

    def _cross_validate_model(self, model: any):
        """ Perform cross validation on provided model. """

        dataset = self._prepare_data(
            split_name=["train", "valid", "test"],
            feature_types=self.c.feature_types
        )

        self.__l.info(f"Cross-validating model on {dataset.inputs_shape()}, {dataset.outputs_shape()}...")
        cross_val_scores = model.cross_validate(
            dataset=dataset,
            groups=self.c.cross_validate_groups,
            verbose=self.c.verbose_fitting
        )
        self.__l.info(f"\tCross-validation completed, scores={cross_val_scores}")

    def _locate_snapshot_deserialize(self, model: BaseNetwork, base_path: str):
        """ Attempt to locate a compatible snapshot and load it into provided model. """

        # Make backup of the original model configuration in case of incompatibility.
        backup = model.serialize()

        # Get list of snapshots ordered by their time of creation - latest first.
        available_snapshots = [
            f.absolute()
            for f in pathlib.Path(base_path).glob(f"**/*.{BaseNetwork.SNAPSHOT_EXTENSION}")
        ]
        available_snapshots.sort(key=os.path.getctime, reverse=True)

        found_compatible = False
        serializer = self.get_instance(Serializer)
        model_dirty = len(available_snapshots) > 0
        for snapshot_path in available_snapshots:
            try:
                serializer.deserialize_model_torch(
                    config=self.config, instance=model,
                    file_path=str(snapshot_path.absolute()))
                found_compatible = True
                break
            except Exception as e:
                self.__l.warn(f"\tFailed to load \"{snapshot_path}\" with ({type(e)}): {e}")

        if not found_compatible:
            self.__l.warn(f"No compatible snapshots found in \"{base_path}\", training from scratch!")
            if model_dirty:
                serializer.deserialize_model_dict(
                    config=self.config, instance=model,
                    serialized=backup
                )

    def _fit_dnn(self, model: DeepFeatureNetwork):
        """ Perform model fitting for DNN models. """

        binary = self.c.binary_pretraining
        differential = self.c.differential_pretraining

        binary_epochs = self.c.binary_pretraining_epochs
        differential_epochs = self.c.differential_pretraining_epochs
        score_epochs = model.cfg["epochs"]

        if self.c.pretrained_continue:
            self._locate_snapshot_deserialize(model, self.c.pretrained_continue)

        if self.c.load_pretrained:
            serializer = self.get_instance(Serializer)
            serializer.deserialize_model(
                config=self.config, instance=model,
                file_path=self.c.load_pretrained
            )

        if binary and model.is_fitting_necessary(estimator="cmpb", epochs=binary_epochs):
            cmp_ds = self._prepare_data(
                split_name=self.c.fit_on,
                feature_types=self.c.feature_types,
                comparison_mode=True,
                comparison_score_difference=False,
                comparison_sample_ptg=self.c.binary_pretraining_ptg,
                comparison_view_scores=self.c.use_view_scores,
                use_tree_variants=self.c.use_tree_variants,
                use_view_variants=self.c.use_view_variants,
                pre_generate_variants=self.c.pre_generate_variants,
            )
            self.__l.info(f"Fitting binary comparison model on {cmp_ds.inputs_shape()}, {cmp_ds.outputs_shape()}...")
            cmp_config = model.fit_cmp(
                dataset=cmp_ds, estimator="cmpb", epochs=binary_epochs, fit_feature=True, verbose=True,
                snapshot_path=self.c.snapshot_path, save_epoch_stride=100
            )
            self.__l.info("\tFitting completed!")
            cmp_ds = None
        else:
            cmp_config = model.data_configuration["cmp_config"] if binary else None

        if differential and model.is_fitting_necessary(estimator="cmpd", epochs=differential_epochs):
            cmp_ds = self._prepare_data(
                split_name=self.c.fit_on,
                feature_types=self.c.feature_types,
                comparison_mode=True,
                comparison_score_difference=True,
                comparison_sample_ptg=self.c.differential_pretraining_ptg,
                comparison_view_scores=self.c.use_view_scores,
                use_tree_variants=self.c.use_tree_variants,
                use_view_variants=self.c.use_view_variants,
                pre_generate_variants=self.c.pre_generate_variants,
            )
            self.__l.info(
                f"Fitting differential comparison model on {cmp_ds.inputs_shape()}, {cmp_ds.outputs_shape()}...")
            cmp_config = model.fit_cmp(
                dataset=cmp_ds, estimator="cmpd", epochs=differential_epochs, fit_feature=True, verbose=True,
                snapshot_path=self.c.snapshot_path, save_epoch_stride=100
            )
            self.__l.info("\tFitting completed!")
            cmp_ds = None
        else:
            cmp_config = model.data_configuration["cmp_config"] if differential else None

        if model.is_fitting_necessary(estimator="score", epochs=score_epochs):
            score_ds = self._prepare_data(
                split_name=self.c.fit_on,
                feature_types=self.c.feature_types,
                comparison_mode=False,
                use_view_scores=self.c.use_view_scores,
                use_tree_variants=self.c.use_tree_variants,
                use_view_variants=self.c.use_view_variants,
                pre_generate_variants=self.c.pre_generate_variants,
            )

            self.__l.info(f"Fitting score model on {score_ds.inputs_shape()}, {score_ds.outputs_shape()}...")
            score_config = model.fit_score(
                dataset=score_ds, epochs=score_epochs, fit_feature=True, verbose=True,
                snapshot_path=self.c.snapshot_path, save_epoch_stride=3000
            )
            self.__l.info("\tFitting completed!")
        else:
            score_config = model.data_configuration["score_config"]

        model.finalize(cmp_config=cmp_config, score_config=score_config)

    def _fit_simple(self, model: any):
        """ Perform model fitting for simple models. """

        dataset = self._prepare_data(
            split_name=self.c.fit_on,
            feature_types=self.c.feature_types
        )

        self.__l.info(f"Fitting model on {dataset.inputs_shape()}, {dataset.outputs_shape()}...")
        model.fit(dataset=dataset, model_type="feature", verbose=self.c.verbose_fitting)
        self.__l.info("\tFitting completed!")

    def _fit_model(self, model: any):
        """ Perform model fitting on provided model. """

        if isinstance(model, DeepFeatureNetwork):
            self._fit_dnn(model=model)
        else:
            self._fit_simple(model=model)

    def _test_model(self, model: any):
        """ Perform model testing on provided pre-trained model. """

        dataset = self._prepare_data(
            split_name=[ "test" ],
            feature_types=self.c.feature_types
        )

        self.__l.info(f"Testing model on {dataset.inputs_shape()}, {dataset.outputs_shape()}...")
        predictions = model.predict(dataset)
        r2_score = sklm.r2_score(y_true=dataset.outputs(), y_pred=predictions)
        mse_score = sklm.mean_squared_error(y_true=dataset.outputs(), y_pred=predictions)
        mae_score = sklm.median_absolute_error(y_true=dataset.outputs(), y_pred=predictions)
        self.__l.info(f"\tTesting completed, R2={r2_score}; MSE={mse_score}; MAE={mae_score}")

    def _analyze_feature_importance(self, model_type: str):
        """ Perform feature importance analysis on provided model type. """

        model = self._prepare_model(type=model_type)

        do_gini = self.c.feature_importance_gini
        do_permutation = self.c.feature_importance_permutation
        permutation_repeats = self.c.feature_importance_permutation_repeats

        if not isinstance(model, ForestFeatureNetwork):
            raise RuntimeError("Feature importance analysis is only supported for forest predictor!")

        forest = model.create_forest(verbose=self.c.verbose_fitting)

        dataset = self._prepare_data(
            split_name=self.c.feature_importance_on,
            feature_types=self.c.feature_types
        )
        dataset_names = dataset.names()[:dataset.inputs_shape()[1]]
        if do_gini or do_permutation:
            results = cross_validate(
                estimator=forest, X=dataset.inputs(), y=dataset.outputs(),
                cv=self.c.feature_importance_runs,
                n_jobs=-1, verbose=self.c.verbose_fitting,
                return_estimator=True
            )

        if do_permutation:
            input_values = dataset.inputs()
            output_values = dataset.outputs()
            cv = KFold(n_splits=self.c.feature_importance_runs, random_state=42)
            rs = [
                permutation_importance(
                    estimator=model, X=input_values[test], y=output_values[test],
                    n_repeats=permutation_repeats,
                    random_state=42
                )
                for model, (train, test) in zip(
                    results["estimator"],
                    cv.split(X=input_values, y=output_values))
            ]

        if do_gini:
            importances = []
            for trained_model in results["estimator"]:
                importances.append([tree.feature_importances_ for tree in trained_model.estimators_])
            importances = np.concatenate(importances).transpose()
            self._analyze_feature_importances(importances=importances, dataset_names=dataset_names,
                                              importance_type="Gini")

        if do_permutation:
            importances = np.concatenate([ r.importances for r in rs ], axis=1)
            self._analyze_feature_importances(importances=importances, dataset_names=dataset_names,
                                              importance_type="Permutation")

    def _analyze_feature_importances(self, importances: np.array,
                                     dataset_names: List[str],
                                     importance_type: str):
        """ Analyze given feature importances
        """
        importance_data = importances.reshape((-1,))
        feature_indices = [idx for idx in range(importances.shape[0]) for _ in range(importances.shape[1])]

        importance_df = pd.DataFrame(data=zip(feature_indices, importance_data), columns=("feature", "score",))

        if self.c.feature_importance_export is not None:
            filepath = f"{self.c.feature_importance_export}/importance_data"
            importance_df.to_csv(f"{filepath}_{importance_type}.csv", sep=";")
            filepath = f"{self.c.feature_importance_export}/importance_data_names"
            pd.DataFrame(data=dataset_names).to_csv(f"{filepath}_{importance_type}.csv", sep=";")

        fig, ax = plt.subplots(figsize=(16, 32))
        g = sns.barplot(ax=ax, x="score", y="feature", data=importance_df, orient="h")
        g.set_title(f"All Feature Importance ({importance_type})")
        g.set_yticklabels(labels=dataset_names[importance_df.feature.unique()])
        plt.show()

        def split_name(name: str) -> pd.Series:
            splits = name.split(":")
            if len(splits) == 1:
                return pd.Series([ splits[ 0 ], "" ],
                                 index=[ "base_name", "spec_name" ])
            if self.c.feature_importance_aggregate_category:
                return pd.Series([ splits[1], ":".join(splits[2:]) ],
                                 index=[ "base_name", "spec_name" ])
            else:
                return pd.Series([ ":".join(splits[:2]), ":".join(splits[2:]) ],
                                 index=[ "base_name", "spec_name"])

        names = dataset_names
        names_split = [ split_name(n) for n in names ]
        base_names, base_name_indices, base_name_mapping = np.unique(
            [ n["base_name"] for n in names_split ],
            return_index=True, return_inverse=True
        )

        if self.c.feature_importance_aggregate_before:
            importance_df["feature"] = base_name_mapping[importance_df["feature"]]
            names = base_names

        def display_importance_information(features: pd.DataFrame, count: int, most: bool):
            self.__l.info(f"Feature ({importance_type}) Ranking up to {self.c.feature_importance_top} {'most' if most else 'least'} important features: ")
            importance_features = features.iloc[:count]
            for order, (idx, data) in enumerate(importance_features.iterrows()):
                self.__l.info(f"\t{order + 1}. Feature {idx} \"{data[ 'name' ]}\" ({data[ 'mean' ]}+-{data[ 'var' ]})")

            top_importance_df = importance_df.loc[ importance_df[ "feature" ].isin(importance_features.index) ]

            fig, ax = plt.subplots(figsize=(16, 32))
            g = sns.barplot(ax=ax, x="score", y="feature", data=top_importance_df, orient="h",
                            order=importance_features.index if self.c.feature_importance_top_sort else None)
            g.set_title(f"{'Top' if most else 'Bottom'} Feature Importance ({importance_type})")
            g.set_yticklabels(labels=dataset_names[
                importance_features.index
                if self.c.feature_importance_top_sort else
                top_importance_df.feature.unique()
            ])

            if self.c.feature_importance_export is not None:
                filename = f"{'top' if most else 'bottom'}_aggregate_feature_importance" \
                    if self.c.feature_importance_aggregate_before or \
                       self.c.feature_importance_aggregate_after \
                    else f"{'top' if most else 'bottom'}_feature_importance"
                filepath = f"{self.c.feature_importance_export}/{filename}"
                top_importance_df.to_csv(f"{filepath}_{importance_type}.csv", sep=";")
                GraphSaver.export_graph(f"{filepath}_{importance_type}.png")

            plt.show()

        aggregate_importance_df = importance_df.groupby("feature").agg(("mean", "var"))["score"]
        top_features = aggregate_importance_df.sort_values(by="mean", ascending=False)
        top_features["name"] = names[top_features.index]
        top_features[[ "base_name", "spec_name" ]] = top_features["name"].apply(split_name)

        if self.c.feature_importance_aggregate_after:
            top_features_agg = top_features.reset_index(drop=False)
            top_features_agg = top_features_agg.loc[top_features_agg.groupby("base_name")["mean"].idxmax()]
            top_features = top_features_agg.set_index("feature", drop=True).sort_values(by="mean", ascending=False)

        display_importance_information(top_features, self.c.feature_importance_top, most=True)

        bottom_features = aggregate_importance_df.sort_values(by="mean", ascending=True)
        bottom_features["name"] = names[bottom_features.index]
        bottom_features[[ "base_name", "spec_name" ]] = bottom_features["name"].apply(split_name)

        if self.c.feature_importance_aggregate_after:
            bottom_features_agg = bottom_features.reset_index(drop=False)
            bottom_features_agg = bottom_features_agg.loc[bottom_features_agg.groupby("base_name")["mean"].idxmax()]
            bottom_features = bottom_features_agg.set_index("feature", drop=True).sort_values(by="mean", ascending=True)

        display_importance_information(bottom_features, self.c.feature_importance_bottom, most=False)

    def train(self):
        """ Train this model on pre-configured data. """

        model = self._prepare_model(type=self.c.model_type)

        if self.c.do_cross_validate:
            self._cross_validate_model(model=model)

        if self.c.do_fit:
            self._fit_model(model=model)

        if self.c.do_fit and self.c.do_test:
            self._test_model(model=model)

        if self.c.do_feature_importance:
            self._analyze_feature_importance(model_type=self.c.model_type)

        self._model = model

    def predict(self, prediction: "Prediction", data: BaseDataLoader):
        """ Predict score for given prediction. """

        if self._model is None:
            raise RuntimeError("Cannot perform prediction, no model is trained!")

        if prediction.view_ids and not prediction.complete_tree:
            tree_filter = { prediction.tree_id: list(prediction.view_ids).copy() }
            tree_filter[prediction.tree_id].append(( -1, 0 ))
        else:
            tree_filter = [ ( prediction.tree_id, ( -1, 0 ) ) ]

        if isinstance(self._model, DeepFeatureNetwork):
            score_config = self._model.data_configuration["score_config"]
        else:
            score_config = self._model.data_configuration
        dataset = self._prepare_data(
            feature_types=self.c.feature_types,
            data_loader=data,
            data_configuration=score_config,
            tree_filter=tree_filter,
            prediction_mode=True,
            use_tree_variants=self.c.use_tree_variants,
            use_view_variants=self.c.use_view_variants,
        )

        dataset2 = self._prepare_data(
            feature_types=self.c.feature_types,
            data_loader=data,
            data_configuration=score_config,
            tree_filter=[ ((130, 0), (-1, 0)) ],
            prediction_mode=True,
            use_tree_variants=self.c.use_tree_variants,
            use_view_variants=self.c.use_view_variants,
        )

        if not prediction.complete_tree and prediction.score_expected:
            dataset = np.repeat(dataset.inputs(), len(prediction.score_expected), axis=0)

        start = time.process_time()
        predictions = self._model.predict(dataset)
        self.__l.info(f"Feature prediction took: {time.process_time() - start}")

        if len(predictions) == 0:
            self.__l.info(f"Failed to predict score, no data!")
            predictions = np.array([ [ 0.0 ] ], dtype=np.float32)

        if prediction.view_ids is not None and len(prediction.view_ids) > 1 and \
                not prediction.complete_tree and prediction.score_expected is None:
            predictions = np.repeat(predictions, len(prediction.view_ids))

        if prediction.score_expected is not None and \
                len(predictions) != len(prediction.score_expected) and \
                not prediction.complete_tree:
            predictions = np.repeat(predictions, len(prediction.score_expected))

        prediction.score_prediction = predictions

        self.__l.info("\tPrediction finished!")

