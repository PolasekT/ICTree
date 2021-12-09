# -*- coding: utf-8 -*-

"""
Training logging using Tensorboard.
"""

import datetime

from perceptree.common.configuration import Config
from perceptree.common.configuration import Configurable
from perceptree.common.configuration import ConfigTemplate
from perceptree.common.logging import Logger

from perceptree.common.pytorch_safe import *


class TensorBoardLogger(Logger, Configurable):
    """ Module allowing logging of network training using Tensorboard. """

    COMMAND_NAME = "TensorBoard"
    """ Name of this command, used for configuration. """

    @classmethod
    def instance(cls) -> "TensorBoardLogger":
        return cls._instance

    @classmethod
    def initialize(cls, config: Config):
        cls._instance = TensorBoardLogger(config)

    def __init__(self, config: Config):
        super().__init__(config=config)

        self.__l.info("Initializing TensorBoard logger...")

        self.enabled = self.c.enabled
        self.trace_graphs = self.c.trace_graphs
        self.trace_profiler = self.c.trace_profiler
        self.output_dir = self.c.output_dir

    @classmethod
    def register_options(cls, parser: Config.Parser):
        """ Register configuration options for this class. """

        option_name = cls._add_config_parameter("enabled")
        parser.add_argument("--enable",
                            action="store_true",
                            default=False,
                            dest=option_name,
                            help="Provide this flag in order to enable TensorBoard logging.")

        option_name = cls._add_config_parameter("trace_graphs")
        parser.add_argument("--trace-graphs",
                            action="store_true",
                            default=False,
                            dest=option_name,
                            help="Provide this flag in order to enable graph tracing.")

        option_name = cls._add_config_parameter("trace_profiler")
        parser.add_argument("--trace-profiler",
                            action="store_true",
                            default=False,
                            dest=option_name,
                            help="Provide this flag in order to enable profiler tracing.")

        option_name = cls._add_config_parameter("output_dir")
        parser.add_argument("--output-dir",
                            action="store",
                            default="./", type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Set base output directory for the TensorBoard outputs.")

    def callbacks(self, model_name: str = "model", timestamp: bool = True) -> list:
        """
        Get list of callbacks which should be provided when fitting model.

        :param model_name: Name of model used for naming the output directory.
        :param timestamp: Add timestamp to the name of the output directory?

        :return: Returns list of callbacks used for logging.
        """

        # TODO - Implement...
        return [ ]

        """
        if not self.enabled:
            return []

        if self.trace_graphs or self.trace_profiler:
            tf.summary.trace_on(graph=self.trace_graphs, profiler=self.trace_profiler)

        output_path = self.output_dir + "/" + model_name
        if timestamp:
            output_path += datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # TODO - Add these options to the command line?
        callback = ks.callbacks.TensorBoard(
            log_dir=output_path,
            histogram_freq=1, batch_size=32,
            write_graph=True, write_images=True,
            profile_batch="0,10",
            update_freq="epoch"
        )

        def fixed_log_weights(self, epoch):
            with self._train_writer.as_default():
                with tf.python.ops.summary_ops_v2.always_record_summaries():
                    for layer in self.model.layers:
                        for weight in layer.weights:
                            weight_name = weight.name.replace(':', '_')

                            # Bool cannot be dumped -> cast it to int.
                            if weight.dtype == tf.bool:
                                weight = tf.cast(weight, dtype=tf.int32)

                            tf.python.ops.summary_ops_v2.histogram(weight_name, weight, step=epoch)
                            if self.write_images:
                                self._log_weight_as_image(weight, weight_name, epoch)
                    self._train_writer.flush()

        callback.__class__._log_weights = fixed_log_weights

        return [callback]
        """
