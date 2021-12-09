# -*- coding: utf-8 -*-

"""
Logging setup and configuration
"""

import datetime
import logging
import os
import pathlib
import sys
import traceback
from typing import List, Optional

from perceptree.common.configuration import Config
from perceptree.common.configuration import Configurable
from perceptree.common.file_saver import FileSaver
from perceptree.common.graph_saver import GraphSaver
from perceptree.common.logger import Logger
from perceptree.common.logger import LoggingBar
from perceptree.common.logger import LogMeta
from perceptree.common.logger import DumpingBar
from perceptree.common.logger import PredictionBar
from perceptree.common.mailer import Mailer
from perceptree.common.mailer import parse_source_e_mail
from perceptree.common.mailer import parse_e_mail
from perceptree.common.profiler import Profiler
from perceptree.common.util import parse_bool_string


class LoggingConfigurator(Logger, Configurable):
    """
    Configuration of the logging system.

    :param config: Application configuration.
    """

    COMMAND_NAME = "Logging"
    """ Name of this command, used for configuration. """

    SMTP_SERVER_PORT = 587
    """ Port used to connect to SMTP mail servers. """
    SUBJECT_PREFIX = "[SP] "
    """ Prefix for each e-mail subject. """
    BODY_FOOTER = "\n\nSolarPowerPrediction automated logging system"
    """ Footer appended to each e-mail body. """

    def __init__(self, config: Config):
        super().__init__(config=config)

        self._logging_directory = self._generate_logging_directory(
            base_dir=self.c.logging_directory,
            logging_name=self.c.logging_name,
            use_timestamp=self.c.logging_directory_use_timestamp,
            use_model=self.c.logging_directory_use_model
        )
        self.setup_profiling(
            enabled=self.c.prof_enabled,
            display=self.c.prof_display,
            save_results_dir=self.c.prof_save_dir,
        )
        self.setup_logging(
            logging_level=self.c.verbosity,
            save_log_to_file=self.c.save_log_to_file,
            file_prefix=self._logging_directory,
        )
        self.__l.info("Initializing logging configurator...")

        self.setup_graphs(
            graphs_as_file=self.c.graphs_to_file or self.c.graphs_to_file_and_show,
            show_graphs=not self.c.graphs_to_file or self.c.graphs_to_file_and_show,
            graph_file_prefix=self._logging_directory
        )

        self.setup_files(
            log_files=self.c.log_files,
            file_prefix=self._logging_directory
        )

        source_mail_password_server = self.c.report_source_mail
        if source_mail_password_server is not None:
            Mailer.configure(
                subject_prefix=self.SUBJECT_PREFIX,
                body_footer=self.BODY_FOOTER,
                source_account=source_mail_password_server[0],
                source_password=source_mail_password_server[1],
                source_server=source_mail_password_server[2],
                source_server_port=self.SMTP_SERVER_PORT,
                report_list=self.c.report_to_mail,
                runtime_info_fun=lambda: self.generate_runtime_info(),
                exception_info_fun=lambda e: self.generate_exception_info(e)
            )
        else:
            Mailer.disable()

        self._report_messages = self.c.report_messages
        self._report_message_subject = self.c.report_message_subject
        if self._report_messages:
            self.send_report_messages(self._report_messages, self._report_message_subject)

    @classmethod
    def register_options(cls, parser: Config.Parser):
        """ Register configuration options for this class. """

        option_name = cls._add_config_parameter("verbosity")
        parser.add_argument("-v", "--verbose",
                            action="store_const",
                            const=logging.INFO,
                            default=logging.INFO,
                            dest=option_name,
                            help="Set to enable informative messages, this is the default state.")

        parser.add_argument("-vv", "--very-verbose",
                            action="store_const",
                            const=logging.DEBUG,
                            dest=option_name,
                            help="Set to enable debug messages.")

        option_name = cls._add_config_parameter("quiet")
        parser.add_argument("-q", "--quiet",
                            action="store_const",
                            const=logging.ERROR,
                            dest=option_name,
                            help="Set to disable all non-error messages.")

        option_name = cls._add_config_parameter("graphs_to_file")
        parser.add_argument("--graphs-to-file",
                            action="store_true",
                            default=False,
                            dest=option_name,
                            help="Save all requested graphs to files instead of "
                                 "displaying them.")

        option_name = cls._add_config_parameter("graphs_to_file_and_show")
        parser.add_argument("--graphs-to-file-and-show",
                            action="store",
                            default=False, type=parse_bool_string,
                            dest=option_name,
                            help="Save all requested graphs to files displaying "
                                 "them as well.")

        option_name = cls._add_config_parameter("logging_directory")
        parser.add_argument("--logging-directory",
                            action="store",
                            default="", type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Base directory used for model logging.")

        option_name = cls._add_config_parameter("logging_directory_use_timestamp")
        parser.add_argument("--logging-directory-use-timestamp",
                            action="store",
                            default=True, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Use timestamp for the output logging directory?")

        option_name = cls._add_config_parameter("logging_directory_use_model")
        parser.add_argument("--logging-directory-use-model",
                            action="store",
                            default=True, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Use model name for the output logging directory?")

        option_name = cls._add_config_parameter("logging_name")
        parser.add_argument("--logging-name",
                            action="store",
                            default="", type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Use given logging name. Specify 'ask' for "
                                 "interactive specification.")

        option_name = cls._add_config_parameter("save_log_to_file")
        parser.add_argument("--save-log-to-file",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Save logs to file in the logging directory?")

        option_name = cls._add_config_parameter("log_files")
        parser.add_argument("--log-files",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Enable file logging into the logging directory?")

        option_name = cls._add_config_parameter("report_source_mail")
        parser.add_argument("--report-source-mail",
                            action="store",
                            default=None, type=parse_source_e_mail,
                            metavar=("USER@MAIL_SERVER:PASSWORD"),
                            dest=option_name,
                            help="Specify source e-mail address from which to send the reports.")

        option_name = cls._add_config_parameter("report_to_mail")
        parser.add_argument("--report-to-mail",
                            action="append",
                            default=[], type=parse_e_mail,
                            metavar=("MAIL_ADDRESS"),
                            dest=option_name,
                            help="Specify target e-mail address to report the results to.")

        option_name = cls._add_config_parameter("report_message_subject")
        parser.add_argument("--report-message-subject",
                            action="store",
                            default="Report Message", type=str,
                            metavar=("SUBJECT"),
                            dest=option_name,
                            help="Specified string will be used as subject to the command line "
                                 "specified report messages.")

        option_name = cls._add_config_parameter("report_messages")
        parser.add_argument("--report-message",
                            action="append",
                            default=[], type=str,
                            metavar=("MESSAGE"),
                            dest=option_name,
                            help="Specify string which should be sent using the e-mail reporting. "
                                 "Multiple messages may be specified, in which case they will be "
                                 "aggregated and sent in one e-mail.")

        option_name = cls._add_config_parameter("prof_enabled")
        parser.add_argument("--prof-enabled",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Enable collection of profiling information?")

        option_name = cls._add_config_parameter("prof_display")
        parser.add_argument("--prof-display",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Display profiling results at the end of the program?")

        option_name = cls._add_config_parameter("prof_save_dir")
        parser.add_argument("--prof-save-dir",
                            action="store",
                            default=None, type=str,
                            metavar=("DIR"),
                            dest=option_name,
                            help="Directory to save the profiling results to.")

    @staticmethod
    def generate_timestamp_str() -> str:
        """
        Generate timestamp string to identify files.

        :return: Returns string representation of the timestamp.
        """

        now = datetime.datetime.now()

        return "{D}_{M}_{Y}-{h}-{m}-{s}-{ms}".format(
            h=now.hour,
            m=now.minute,
            s=now.second,
            ms=int(now.microsecond / 1000.0),
            D=now.day,
            M=now.month,
            Y=now.month
        )

    def _generate_logging_directory(self, base_dir: str, logging_name: str,
                                    use_timestamp: bool, use_model: bool):
        """ Generate unique logging directory for this run. """

        if not base_dir and not logging_name:
            return ""

        if logging_name == "ask":
            logging_name = input("Enter logging name: ")

        model_names = ("_" + "_".join(self.config.get_requested_models().keys())) if use_model else ""
        timestamp = f"_{LoggingConfigurator.generate_timestamp_str()}" if use_timestamp else ""
        spec_dir = f"{logging_name}{model_names}{timestamp}"

        full_path = pathlib.Path(f"{base_dir}/{spec_dir}").absolute()
        os.makedirs(full_path, exist_ok=True)

        return str(full_path)

    def setup_file_logging_to_file(self, file_path: str) -> logging.FileHandler:
        """ Setup logging of all handlers to given log file. """
        return LogMeta.setup_file_logging_to_file(file_path=file_path)

    def remove_file_logging_to_file(self, file_handler: logging.FileHandler):
        """ Remove logging to given file handler. """
        return LogMeta.remove_file_logging_to_file(file_handler=file_handler)

    def setup_profiling(self, enabled: bool, display: bool, save_results_dir: Optional[str]):
        """
        Setup profiling system.

        :param enabled: Enable collection of profiling information?
        :param display: Display profiling results at the end of the application?
        :param save_results_dir: Base directory to save the results to.
        """

        Profiler.configure(
            collect_data=enabled,
            display_data=display,
            save_results_dir=save_results_dir,
        )

        Profiler.reset_all_data()

    def setup_logging(self, logging_level: int = logging.INFO,
                      save_log_to_file: bool = False,
                      file_prefix: str = ""):
        """
        Setup logging streams and configure them.

        :param logging_level: Which level of messages should
            be displayed?
        :param save_log_to_file: Save logs to file in the
            file_prefix?
        :param file_prefix: Prefix used for saving files.
        """

        """
        absl_logging_level = absl.logging.converter.STANDARD_TO_ABSL[logging_level]
        absl.logging.set_verbosity(absl_logging_level)

        # TODO - Re-enable abseil logging?
        logging.root.removeHandler(absl.logging._absl_handler)
        absl.logging._warn_preinit_stderr = False
        """

        logging.basicConfig(level=logging_level)

        # Enable progress bar only when logging INFO or lower.
        LoggingBar.enabled = (logging_level <= logging.INFO)
        LoggingBar.check_tty = False

        # Enable log redirection to file if requested.
        if save_log_to_file:
            LogMeta.setup_file_logging(base_path=file_prefix)

    def setup_graphs(self, graphs_as_file: bool = False,
                     show_graphs: bool = False,
                     graph_file_prefix: str = ""):
        """
        Setup graph processing.

        :param graphs_as_file: Save graphs as files instead
            of displaying them?
        :param show_graphs: Display graphs?
        :param graph_file_prefix: Filepath prefix for each
            saved graph.
        """

        # Setup graph saving.
        GraphSaver.save_as_file = graphs_as_file
        GraphSaver.show_graph = show_graphs
        if graph_file_prefix and graph_file_prefix[-1] != "/":
            graph_file_prefix += "/"
        GraphSaver.directory_prefix = graph_file_prefix

    def setup_files(self, log_files: bool = False,
                    file_prefix: str = ""):
        """
        Setup file logging.

        :param log_files: Enable file logging?
        :param file_prefix: Filepath prefix for each file.
        """

        FileSaver.enabled = log_files
        FileSaver.file_prefix = file_prefix

    def send_report_messages(self, message_list: List[str], subject: Optional[str]):
        """
        Send aggregated report message consisting of messages in
        given list.

        :param message_list: List of messages to be sent.
        :param subject: Subject of the message.
        """

        aggregated_message = "\n".join(message_list)
        subject = subject or "Report Message"
        body = f"Reporting following messages from command line: \n{aggregated_message}"
        self.__m.report(subject, body)

    def generate_runtime_info(self) -> str:
        """
        Generate information about current runtime configuration.

        :return: Returns formatted string containing all information.
        """

        return "Runtime information: \n" \
               f"\tPython runtime: {sys.version}\n" \
               f"\tStart time: {self.config.start_time}\n" \
               f"\tCommand line arguments: {self.config.runtime_arguments}\n" \
               f"\tConfiguration: {self.config.save_config()}"

    def generate_exception_info(self, exception: Optional[Exception]) -> str:
        """
        Generate exception information including a stack trace.

        :param exception: Exception to describe.

        :return: Returns formatted string containing all information.
        """

        return "Exception information: \n" \
               f"\tException type: {type(exception)}\n" \
               f"\tException text: {str(exception)}\n" \
               f"\tTrace: {traceback.format_exc()}"

    @property
    def logging_level(self) -> "logging_level":
        """ Get current logging level, such as logging.WARNING . """
        return self.c.verbosity
