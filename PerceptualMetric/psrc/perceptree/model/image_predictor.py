# -*- coding: utf-8 -*-

"""
Model used for image-only predictions.
"""

from perceptree.model.base import *

from perceptree.common.logger import FittingBar
from perceptree.common.graph_saver import GraphSaver
from perceptree.model.util import model_by_name
from perceptree.model.util import imagenet_transform
from perceptree.model.util import imagenet_classes
from perceptree.model.util import res2net_transform
from perceptree.model.util import AddTransform


class SiameseNetwork(BaseNetwork):
    """
    Siamese network using tree views for prediction.

    Internal sub-models are "feature", "cmp" and "score".
    Optimizer, loss and lr_scheduler may be provided as a tuple containing class
    and a dictionary of keyword arguments passed to the constructor.

    :param batch_size: Size of batches or dictionary specifying for each model.
    :param optimizer: Optimizer factory taking list of parameters or dictionary specifying for each model.
    :param loss: Loss factory taking optimizer or dictionary specifying for each model.
    :param lr_scheduler: Optional lr scheduler factory taking optimizer or dictionary specifying for each model.
    :param model_parameters: Parameters for each model.

    :usage:
        >>> network = SiameseNetwork(
        >>>     batch_size={ "cmp": 32, "score": 64 },
        >>>     optimizer=(torch.optim.SGD, { "lr": 0.0001, "momentum": 0.9 }),
        >>>     loss={ "cmp": tnn.MSELoss, "score": tnn.MSELoss },
        >>>     lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        >>>     model_parameters={
        >>>         "feature": { "pretrained": True },
        >>>         "cmp": { "feature_classes": 1000, "differential": True },
        >>>         "score": { "feature_classes": 1000 },
        >>>     }
        >>>)
    """

    def __init__(self, batch_size: Union[int, Dict[str, int]],
                 optimizer: Union[any, Dict[str, any]], loss: Union[any, Dict[str, any]],
                 lr_scheduler: Optional[Union[Any, Dict[str, any]]] = None,
                 model_parameters: Optional[Dict[str, Dict[str, any]]] = None):
        super().__init__(cfg={
            "batch_size": batch_size,
            "optimizer": optimizer,
            "loss": loss,
            "lr_scheduler": lr_scheduler,
            "model_parameters": model_parameters
        }, model_names=[ "feature", "cmpb", "cmpd", "score" ])

        self._initialize()

    @classmethod
    def from_serialized(cls, serialized: dict) -> "SiameseNetwork":
        """ Load model from serialized representation. """

        model = SiameseNetwork(**serialized["cfg"])
        model.deserialize(serialized)

        return model

    def deserialize(self, serialized: dict):
        """ Deserialize model from given dictionary. """

        if "cmp" in serialized["model"]:
            # Using old model hierarchy, change it over to the new one.
            for key, value in recurse_dict(serialized):
                if isinstance(value, dict) and "cmp" in value:
                    value["cmpb"] = value["cmp"]
                    del value["cmp"]

        return super().deserialize(serialized=serialized)

    class TreeFeatureExtractor(tnn.Module):
        def __init__(self, pretrained: bool = True, network_style: str = "resnet18"):
            super(SiameseNetwork.TreeFeatureExtractor, self).__init__()

            # Prepare standard resnet architecture, optionally get pretrained weights.
            self.resnet = model_by_name(name=network_style)(pretrained=pretrained)
            if pretrained:
                with torch.no_grad():
                    for param in self.resnet.parameters():
                        param.add_(torch.randn(param.size()) * 0.01)

            self.using_resnet = hasattr(self.resnet, "conv1")
            self.network_style = network_style

        def forward(self, x):
            """ Calculate forward pass for current network configuration. """

            if self.using_resnet:
                # Calculate forward pass in resnet:
                x = self.resnet.conv1(x)
                x = self.resnet.bn1(x)
                x = self.resnet.relu(x)
                x = self.resnet.maxpool(x)

                x = self.resnet.layer1(x)
                x = self.resnet.layer2(x)
                x = self.resnet.layer3(x)
                x = self.resnet.layer4(x)
            else:
                # Calculate forward pass in DLA:
                x = x.permute(0, 3, 1, 2)
                x = self.resnet.base_layer(x)
                for idx in range(6):
                    x = getattr(self.resnet, f"level{idx}")(x)

            return x

    class TreeComparisonClassifier(tnn.Module):
        """ Classifier utilizing pre-generated features to generate comparison estimate. """

        def __init__(self, input_feature_count: int, feature_classes: int = 1000, differential: bool = True):
            super(SiameseNetwork.TreeComparisonClassifier, self).__init__()

            # Prepare output layer for comparison output.
            self.avgpool = tnn.AdaptiveAvgPool2d(( 1, 1 ))
            self.fc1 = tnn.Linear(input_feature_count, feature_classes)
            self.fc2 = tnn.Linear(feature_classes, 1)
            self.act = tnn.ReLU() if differential else tnn.Sigmoid()

        def forward(self, x):
            """ Calculate forward pass for current network configuration. """

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)
            x = self.act(x)

            return x

    class TreeScoreRegressor(tnn.Module):
        """ Regressor utilizing pre-generated features to generate score estimate. """

        def __init__(self, input_feature_count: int, feature_classes: int = 1000):
            super(SiameseNetwork.TreeScoreRegressor, self).__init__()

            # Prepare output layer for score output.
            self.avgpool = tnn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = tnn.Linear(input_feature_count, feature_classes)
            self.fc2 = tnn.Linear(feature_classes, 1)
            self.relu = tnn.ReLU()

        def forward(self, x):
            """ Calculate forward pass for current network configuration. """

            # Generate class based on the choice:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)

            return x

    def _initialize(self):
        """ Initialize the networks to default state. """
        super()._initialize()

        device, cpu_device = self._get_devices()

        self._model["feature"] = SiameseNetwork.TreeFeatureExtractor(
            **self._cfg["model_parameters"].get("feature", { })
        ).train().to(device=device)

        feature_count = self._model["feature"].resnet.fc.in_features
        double_feature_count = 2 * feature_count

        self._model["cmpb"] = SiameseNetwork.TreeComparisonClassifier(
            # Getting features from both branches -> 2x.
            input_feature_count=double_feature_count,
            **self._cfg["model_parameters"].get("cmpb", { })
        ).train().to(device=device)
        self._model["cmpd"] = SiameseNetwork.TreeComparisonClassifier(
            # Getting features from both branches -> 2x.
            input_feature_count=double_feature_count,
            **self._cfg["model_parameters"].get("cmpd", { })
        ).train().to(device=device)
        self._model["score"] = SiameseNetwork.TreeScoreRegressor(
            # Getting features from single branch only -> no multiplication.
            input_feature_count=feature_count,
            **self._cfg["model_parameters"].get("score", { })
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

        if self._data_configuration is None:
            self._data_configuration = { }
        self._data_configuration["cmp_config"] = dataset.data_config

        self._fit(
            dataset=dataset, verbose=verbose,
            model_type=estimator, epochs=epochs,
            feature=self._model["feature"],
            estimate=self._model[estimator],
            fit_feature=fit_feature,
            epoch_callback=lambda e, te, l: self._epoch_callback(
                model_type=estimator, model_name="ImagePredictor",
                epoch=e, total_epochs=te, losses=l, do_loss_graph=verbose,
                snapshot_path=snapshot_path, epoch_stride=save_epoch_stride
            )
        )
        torch.cuda.empty_cache()

        return dataset.data_config

    def fit_score(self, dataset: TreeDataset, epochs: int, fit_feature: bool = True, verbose: bool = True,
                  snapshot_path: Optional[str] = None, save_epoch_stride: Optional[int] = None) -> dict:
        """ Fit score estimation part of the model. """

        if self._data_configuration is None:
            self._data_configuration = { }
        self._data_configuration["score_config"] = dataset.data_config

        self._fit(
            dataset=dataset, verbose=verbose,
            model_type="score", epochs=epochs,
            feature=self._model["feature"],
            estimate=self._model["score"],
            fit_feature=fit_feature,
            epoch_callback=lambda e, te, l: self._epoch_callback(
                model_type="score", model_name="ImagePredictor",
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
        self._training_state["losses"] = { }

    def _fit(self, dataset: TreeDataset, verbose: bool,
             model_type: str, epochs: int,
             feature: tnn.Module, estimate: tnn.Module,
             fit_feature: bool,
             epoch_callback: Optional[callable]):
        """ Perform fitting on provided feature and estimation models. """

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
            x = torch.cat(x, 1).reshape((-1,) + view_shapes[ 0 ])
            features = feature(x.float().to(device=dev))
            features = features.reshape((batch_size, -1,) + features.shape[ 2: ])

            prediction = estimate(features)

            loss_grad = loss_fun(prediction, y.float().to(device=dev))

            return loss_grad

        # Prepare for training:
        view_shapes = dataset.view_shapes()
        view_sizes = np.prod(view_shapes, axis=-1)
        view_count = len(view_sizes)
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
                                      save_path=f"ImagePredictor_{model_type}_loss_final")
            GraphSaver.save_graph(f"ImagePredictor_{model_type}_loss_final")

    def predict(self, inputs: Union[np.array, TreeDataset]) -> np.array:
        assert(self._training_state["finalized"])
        if isinstance(inputs, TreeDataset):
            model_inputs = torch.from_numpy(inputs.inputs().reshape((-1,) + inputs.view_shapes()[0])).float()
        else:
            model_inputs = torch.from_numpy(inputs).float()
        if len(model_inputs) > 0:
            with torch.no_grad():
                return self._model["score"](self._model["feature"](model_inputs)).numpy()
        else:
            return np.array([ ])

    def cross_validate(self, dataset: TreeDataset, verbose: bool) -> np.array:
        raise RuntimeError("Cross validation for DNN not implemented!")

    @property
    def data_configuration(self) -> dict:
        """ Get data configuration used to train this model. """
        return self._data_configuration


class ImagePredictor(BaseModel):
    """
    Model used for image-only predictions.

    :param config: Application configuration.
    """

    MODEL_NAME = "iop"
    """ Name of this prediction model. """
    CATEGORY_NAME = "ImagePredictor"
    """ Category as visible to the user. """

    def __init__(self, config: Config):
        super().__init__(config, self.MODEL_NAME)

        self._model = None

    @classmethod
    def register_options(cls, parser: Config.Parser):
        """ Register configuration options for this class. """

        super().register_common_options(parser)

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

        option_name = cls._add_config_parameter("batch_size")
        parser.add_argument("--batch-size",
                            action="store",
                            default=32, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Mini-batch size used for training.")

        option_name = cls._add_config_parameter("score_epochs")
        parser.add_argument("--score-epochs",
                            action="store",
                            default=500, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Number of epochs to train the score network for.")

        option_name = cls._add_config_parameter("view_resolution")
        parser.add_argument("--view-resolution",
                            action="store",
                            default=256, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Resolution of the views.")

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

        option_name = cls._add_config_parameter("snapshot_path")
        parser.add_argument("--snapshot-path",
                            action="store",
                            default=None, type=str,
                            metavar=("PATH_TO_SNAPSHOTS"),
                            dest=option_name,
                            help="Save model snapshots to given directory.")

        option_name = cls._add_config_parameter("binary_pretraining")
        parser.add_argument("--binary-pretraining",
                            action="store",
                            default=True, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Use binary data in the pretraining phase?")

        option_name = cls._add_config_parameter("binary_pretraining_epochs")
        parser.add_argument("--binary-pretraining-epochs",
                            action="store",
                            default=50, type=int,
                            metavar=("COUNT"),
                            dest=option_name,
                            help="Number of epochs in the binary pre-training phase")

        option_name = cls._add_config_parameter("binary_pretraining_ptg")
        parser.add_argument("--binary-pretraining-ptg",
                            action="store",
                            default=None, type=float,
                            metavar=("<0.0, 1.0>"),
                            dest=option_name,
                            help="Percentage of records to use in binary pre-training.")

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

        option_name = cls._add_config_parameter("load_pretrained")
        parser.add_argument("--load-pretrained",
                            action="store",
                            default=None, type=str,
                            metavar=("PATH_TO_PTH"),
                            dest=option_name,
                            help="Load pre-trained snapshot from given path.")

        option_name = cls._add_config_parameter("pretrained_continue")
        parser.add_argument("--pretrained-continue",
                            action="store",
                            default=None, type=str,
                            metavar=("PATH_TO_BASE_DIR"),
                            dest=option_name,
                            help="Find latest compatible snapshot in given "
                                 "path and continue training.")

        option_name = cls._add_config_parameter("network_style")
        parser.add_argument("--network-style",
                            action="store",
                            default="resnet18", type=str,
                            metavar=("NETWORK_STYLE"),
                            dest=option_name,
                            help="Name of the network architecture to use.")

        option_name = cls._add_config_parameter("network_pretrained")
        parser.add_argument("--network-pretrained",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Use ImageNet pre-trained variant of the network architecture?")

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

    def _get_current_feature_config(self) -> CacheDict:
        """ Get current configuration of the featurizer. """

        return CacheDict({
            "view_resolution": self.c.view_resolution,
            "view_interpolation": None,
            "generate_names": False,
        })

    def _prepare_data(self,
                      split_name: Optional[List[str]] = None,
                      view_types: Optional[List[str]] = None,
                      view_transform: Optional[callable] = None,
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
                      prediction_mode: bool = False
                      ) -> TreeDataset:
        """ Prepare data for given split, return generated dataset. """

        self.__l.info("Preparing dataset...")

        config = self._prepare_dataset_config(
            split_name=split_name,
            view_types=view_types,
            data_loader=data_loader,
            data_configuration=data_configuration,
            tree_filter=tree_filter,
            prediction_mode=prediction_mode,
            feature_config=self._get_current_feature_config()
        )

        config.set_featurizer_option("comparison_mode", comparison_mode)
        config.set_featurizer_option("comparison_score_difference", comparison_score_difference)
        config.set_featurizer_option("comparison_sample_ptg", comparison_sample_ptg)
        #config.set_featurizer_option("view_flatten", False)
        config.set_featurizer_option("comparison_view_scores", comparison_view_scores)
        config.set_featurizer_option("use_view_scores", use_view_scores)
        config.set_featurizer_option("use_tree_variants", use_tree_variants)
        config.set_featurizer_option("use_view_variants", use_view_variants)
        config.set_featurizer_option("pre_generate_variants", pre_generate_variants)
        config.set_featurizer_option("view_transform", view_transform)

        return TreeDataset(config=config)

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

    def train(self):
        """ Train this model. """

        binary = self.c.binary_pretraining
        differential = self.c.differential_pretraining

        if self.c.network_style.startswith("resnet"):
            view_transform = imagenet_transform(max_value=255.0)
        elif self.c.network_style.startswith("res2net"):
            view_transform = res2net_transform(max_value=255.0)
        else:
            view_transform = None

        model = SiameseNetwork(
            batch_size={ "cmpb": self.c.batch_size, "cmpd": self.c.batch_size, "score": self.c.batch_size },
            optimizer=(torch.optim.Adam, { "lr": 0.0001, "amsgrad": True }),
            loss={ "cmpb": tnn.BCELoss, "cmpd": tnn.MSELoss, "score": tnn.MSELoss },
            lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            model_parameters={
                "feature": {
                    "pretrained": self.c.network_pretrained,
                    "network_style": self.c.network_style
                },
                "cmpb": { "feature_classes": 1000, "differential": False },
                "cmpd": { "feature_classes": 1000, "differential": True },
                "score": { "feature_classes": 1000 }
            }
        )

        binary_epochs = self.c.binary_pretraining_epochs
        differential_epochs = self.c.differential_pretraining_epochs
        score_epochs = self.c.score_epochs

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
                view_types=[ "base" ],
                view_transform=view_transform,
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
                snapshot_path=self.c.snapshot_path, save_epoch_stride=1
            )
            self.__l.info("\tFitting completed!")
            cmp_ds = None
        else:
            cmp_config = model.data_configuration["cmp_config"] if binary else None

        if differential and model.is_fitting_necessary(estimator="cmpd", epochs=differential_epochs):
            cmp_ds = self._prepare_data(
                split_name=self.c.fit_on,
                view_types=[ "base" ],
                view_transform=view_transform,
                comparison_mode=True,
                comparison_score_difference=True,
                comparison_sample_ptg=self.c.differential_pretraining_ptg,
                comparison_view_scores=self.c.use_view_scores,
                use_tree_variants=self.c.use_tree_variants,
                use_view_variants=self.c.use_view_variants,
                pre_generate_variants=self.c.pre_generate_variants,
            )
            self.__l.info(f"Fitting differential comparison model on {cmp_ds.inputs_shape()}, {cmp_ds.outputs_shape()}...")
            cmp_config = model.fit_cmp(
                dataset=cmp_ds, estimator="cmpd", epochs=differential_epochs, fit_feature=True, verbose=True,
                snapshot_path=self.c.snapshot_path, save_epoch_stride=1
            )
            self.__l.info("\tFitting completed!")
            cmp_ds = None
        else:
            cmp_config = model.data_configuration["cmp_config"] if differential else None

        if model.is_fitting_necessary(estimator="score", epochs=score_epochs):
            score_ds = self._prepare_data(
                split_name=self.c.fit_on,
                view_types=[ "base" ],
                view_transform=view_transform,
                comparison_mode=False,
                use_view_scores=self.c.use_view_scores,
                use_tree_variants=self.c.use_tree_variants,
                use_view_variants=self.c.use_view_variants,
                pre_generate_variants=self.c.pre_generate_variants,
            )
            self.__l.info(f"Fitting score model on {score_ds.inputs_shape()}, {score_ds.outputs_shape()}...")
            score_config = model.fit_score(
                dataset=score_ds, epochs=score_epochs, fit_feature=True, verbose=True,
                snapshot_path=self.c.snapshot_path, save_epoch_stride=50
            )
            self.__l.info("\tFitting completed!")
            score_ds = None
        else:
            score_config = model.data_configuration["score_config"]

        model.finalize(cmp_config=cmp_config, score_config=score_config)

        self._model = model

    def predict(self, prediction: "Prediction", data: BaseDataLoader):
        """ Predict results for provided trees. """

        if self._model is None:
            raise RuntimeError("Cannot perform prediction, no model is trained!")

        if prediction.view_ids:
            tree_filter = { prediction.tree_id: prediction.view_ids }
        else:
            tree_filter = [ prediction.tree_id ]

        score_config = self._model.data_configuration["score_config"]
        view_types = score_config.get("data", { }).get("views", { }).get("view_types", [ "base" ])
        view_transform = imagenet_transform(max_value=255.0)
        dataset = self._prepare_data(
            view_types=view_types,
            view_transform=view_transform,
            comparison_mode=False,

            data_loader=data,
            data_configuration=score_config,
            tree_filter=tree_filter,
            prediction_mode=True,
            use_tree_variants=self.c.use_tree_variants,
            use_view_variants=self.c.use_view_variants,
        )

        start = time.process_time()
        predictions = self._model.predict(dataset)
        print("Image prediction took: ", time.process_time() - start)

        if len(predictions) == 0:
            self.__l.info(f"Failed to predict score, no data!")
            predictions = np.array([ [ 0.0 ] ], dtype=np.float32)

        prediction.score_prediction = [ np.mean(predictions) ] if prediction.complete_tree else predictions

        self.__l.info("\tPrediction finished!")

