#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-validation helper script for the PerceptualMetric.
"""

from perceptree.perceptree_main import *

import datetime
from dateutil.relativedelta import relativedelta
import itertools
import logging
import os
import pathlib
import re
from typing import Dict, Optional, Tuple

import pickle as pk

from perceptree.common.file_saver import FileSaver
from perceptree.common.util import parse_bool_string
from perceptree.common.profiler import ProfTimer


class PerceptreeCrossMain(Logger, Configurable):
    """ Wrapper around the main function. """

    COMMAND_NAME = "Cross"
    """ Name of this command, used for configuration. """

    def __init__(self):
        pass

    @classmethod
    def register_options(cls, parser: Config.Parser):
        """ Register configuration options for this class. """

        option_name = cls._add_config_parameter("cross_search")
        parser.add_argument("-s", "--cross-search",
                            action="store",
                            default="20/42", type=str,
                            metavar=("count/seed"),
                            dest=option_name,
                            help="Specify cross validation intervals to search through.")

        option_name = cls._add_config_parameter("grid_search")
        parser.add_argument("-g", "--grid-search",
                            action="append",
                            default=[ ],
                            metavar=("param-name:start:end:step|param-name:v1/v2/v3,..."),
                            dest=option_name,
                            help="Perform grid search on given parameters. Specify multiple "
                                 "times to get all combinations. Parameter name may be a "
                                 "comma separated list of parameters, in which case all "
                                 "of the values should be tuples.")

        option_name = cls._add_config_parameter("dry_run")
        parser.add_argument("-d", "--dry-run",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Only print commands without running them. "
                                 "Skips modification of continuation file as well.")

        option_name = cls._add_config_parameter("base_logging_path")
        parser.add_argument("-p", "--base-logging-path",
                            action="store",
                            default=None, type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Base path used for logging outputs.")

        option_name = cls._add_config_parameter("base_logging_name")
        parser.add_argument("-n", "--base-logging-name",
                            action="store",
                            default=None, type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="Base name used for logging outputs.")

        option_name = cls._add_config_parameter("use_timestamps")
        parser.add_argument("-t", "--use-timestamps",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Use timestamps in the file and directory names?")

        option_name = cls._add_config_parameter("continuation_file")
        parser.add_argument("-c", "--continuation-file",
                            action="store",
                            default=None, type=str,
                            metavar=("PATH"),
                            dest=option_name,
                            help="File used for continuation after crash.")

        option_name = cls._add_config_parameter("wait")
        parser.add_argument("-w", "--wait",
                            action="store",
                            default=True, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Wait after each validation step until enter is pressed?")

        option_name = cls._add_config_parameter("recalculate_only")
        parser.add_argument("-r", "--recalculate-only",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Only re-calculate the predictions")

        option_name = cls._add_config_parameter("no_wait_during")
        parser.add_argument("--no-wait-during",
                            action="store",
                            default=None, type=PerceptreeCrossMain.parse_time_interval,
                            metavar=("HH:MM:SS,HH:MM:SS"),
                            dest=option_name,
                            help="Disable waiting when current time is in given interval, inclusively."
                                 "Waiting must be enabled by --wait!")

        option_name = cls._add_config_parameter("prof_mode")
        parser.add_argument("--prof-mode",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=option_name,
                            help="Enable profiling mode?")

    def cross_arguments(self, argv: List[str]) -> (List[str], List[str]):
        """ Get cross-validation arguments and model arguments. """

        split_idx = argv.index("CrossValidationArguments")
        return argv[:split_idx], [ argv[0] ] + argv[split_idx + 1:]

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
            Y=now.year
        )

    @staticmethod
    def format_date(dt: datetime.datetime) -> str:
        """ Format given date-time into compatible format. """

        return f"{dt.year}-{dt.month}-{dt.day}"

    @staticmethod
    def parse_time_interval(spec: str) -> Optional[Tuple[datetime.time, datetime.time]]:
        """ Parse time interval specification and return the result. """

        parts = spec.split(",")
        if len(parts) != 2:
            raise RuntimeError(f"Invalid time interval: \"{spec}\"")

        return (
            datetime.datetime.strptime(parts[0], "%H:%M:%S").time(),
            datetime.datetime.strptime(parts[1], "%H:%M:%S").time(),
        )

    def _prepare_get_continuation(self, config: Config, path: Optional[str], use_timestamps: bool) -> (str, dict):
        """ Get continuation if available, return base logging path and completed dict. """

        cross_timestamp = f"_{PerceptreeCrossMain.generate_timestamp_str()}" if use_timestamps else ""
        base_logging_path = f"{config['cross.base_logging_path']}/{config['cross.base_logging_name']}" \
                            f"{cross_timestamp}"
        os.makedirs(base_logging_path, exist_ok=True)

        completed_splits = { }

        try:
            with open(path, "rb") as f:
                data = pk.load(f)
                base_logging_path = data.get("base_logging_path", base_logging_path)
                completed_splits = data.get("completed_splits", completed_splits)
        except Exception as e:
            self.__l.info(f"Failed to get continuation \"{e}\"!")

        return base_logging_path, completed_splits

    def _save_continuation(self, base_logging_path: str, completed_splits: dict,
                           path: Optional[str]):
        """ Save current state for continuation purposes. """

        if path is not None:
            try:
                with open(path, "wb") as f:
                    data = {
                        "base_logging_path": base_logging_path,
                        "completed_splits": completed_splits
                    }
                    pk.dump(data, f)
            except Exception as e:
                self.__l.info(f"Failed to save continuation \"{e}\"!")

    def _prepare_cross_validations(self, cross_search: str) -> List[str]:
        """ Prepare cross validation strings for the search. """

        split = cross_search.split("/")
        if len(split) != 2:
            raise RuntimeError(f"Invalid cross_search specified \"{cross_search}\"!")

        count, seed = int(split[0]), int(split[1])
        cross_validations = [
            f"{idx + 1}/{count}/{seed}"
            for idx in range(count)
        ]

        return cross_validations

    def _prepare_grid_search(self, grid_specs: List[str]) -> List[Dict[str, any]]:
        """ Prepare grid search from given specification. """

        self.__l.info(f"Preparing {len(grid_specs)} grid search parameters...")

        def parse_value(v):
            if v.find(".") >= 0:
                return float(v)
            else:
                return int(v)

        specs = [ ]
        names = [ ]
        for grid_spec in grid_specs:
            splits = grid_spec.split(":")
            if splits[0].find(",") >= 0:
                parameter_name = [ "--" + str(n) for n in splits[0].split(",") ]

                if len(splits) == 4:
                    parameter_start = tuple(parse_value(v) for v in splits[1].split(","))
                    parameter_end = tuple(parse_value(v) for v in splits[2].split(","))
                    parameter_step = tuple(parse_value(v) for v in splits[3].split(","))

                    self.__l.info(f"\t Parameters \"{parameter_name}\" "
                                       f"{parameter_start}:{parameter_end}:{parameter_step}")

                    parameter_space = list(zip([
                        list(
                            range(
                                start,
                                end + 1,
                                step
                            )
                        )
                        for start, end, step in zip(parameter_start, parameter_end, parameter_step)
                    ]))
                else:
                    parameter_space = [
                        [ parse_value(v) for v in vs.split(",") ]
                        for vs in splits[1].split("/")
                    ]
                    self.__l.info(f"\t Parameters \"{parameter_name}\" {parameter_space}")
            else:
                parameter_name = "--" + str(splits[ 0 ])

                if len(splits) == 4:
                    parameter_start = parse_value(splits[1])
                    parameter_end = parse_value(splits[2])
                    parameter_step = parse_value(splits[3])

                    self.__l.info(f"\t Parameter \"{parameter_name}\" "
                                       f"{parameter_start}:{parameter_end}:{parameter_step}")

                    parameter_space = list(range(
                        parameter_start, parameter_end + 1, parameter_step
                    ))
                else:
                    parameter_space = [ parse_value(v) for v in splits[1].split("/") ]
                    self.__l.info(f"\t Parameter \"{parameter_name}\" {parameter_space}")

            names.append(parameter_name)
            specs.append(parameter_space)

        search_grid = [ ]
        for vals in itertools.product(*specs):
            grid_point = { }
            for name, val in zip(names, vals):
                if isinstance(name, list):
                    for n, v in zip(name, val):
                        grid_point[n] = v
                else:
                    grid_point[name] = val
            search_grid.append(grid_point)

        self.__l.info(f"Search grid finished with {len(search_grid)} total points!")

        return search_grid

    @staticmethod
    def _list_regex_first_index(values: List[str], matcher: "re") -> int:
        """ Find index of the first matching string in the list. Returns -1 if no matches are found."""

        if isinstance(matcher, str):
            matcher = re.compile(matcher)

        for idx, val in enumerate(values):
            if matcher.match(val):
                return idx

        return -1

    @staticmethod
    def _list_index(values: List[str], key: str) -> int:
        """ Find index of the first matching key in the list. Returns -1 if no matches are found."""
        try:
            return values.index(key)
        except ValueError:
            return -1

    def _set_argv(self, argv: List[str], values: Dict[str, dict]) -> List[str]:
        """ Set given values in the argument vector. Missing values may be added after given mark. """

        for key, d in values.items():
            value = d["value"]
            replace_key = d.get("replace", None)
            skip_missing = d.get("kip", False)
            missing_mark = d.get("mark", None) or ""

            position = PerceptreeCrossMain._list_index(argv, key)

            if not skip_missing and position < 0:
                assert(len(missing_mark) > 0)
                mark_position = PerceptreeCrossMain._list_regex_first_index(argv, missing_mark)
                assert(mark_position >= 0)
                argv.insert(mark_position + 1, value)
                argv.insert(mark_position + 1, key)
                position = mark_position + 1
            elif position < 0:
                raise RuntimeError(f"Argument \"{key}\" not found!")

            argv[position + 1] = value
            if replace_key is not None:
                argv[position] = replace_key

        return argv

    def check_is_during(self, interval: Optional[Tuple[datetime.time, datetime.time]]) -> bool:
        """ Check if current time is in given interval. """

        now = datetime.datetime.now()
        return interval is None or \
               (now.time() >= interval[0] and now.time() <= interval[1])

    def main(self, argv: List[str]):
        """
        Main function which contains:
            * Parameter processing
            * Calling inner functions according to the parameters
            * Error reporting

        :param argv: Argument vector including the app name.
        """

        # Initialize configuration.
        config = Config()

        # Register systems.
        PerceptreeCrossMain.register_config(config)
        LoggingConfigurator.register_config(config)

        # Initialize all configuration options.
        config.init_options()

        # Parse arguments passed from the command line.
        cross_argv, model_argv = self.cross_arguments(argv)
        model_argv[0] = model_argv[0].replace("_cross", "")
        cross_argv = cross_argv[1:]
        config.parse_args(cross_argv)

        # Setup configuration for this class and get command line arguments.
        super().__init__(config=config)

        # Prepare common logging directory
        dry_run = self.c.dry_run
        use_timestamps = self.c.use_timestamps
        base_logging_path, completed_splits = self._prepare_get_continuation(
            config=config, path=self.c.continuation_file,
            use_timestamps=use_timestamps
        )

        # Prepare runtime arguments.
        no_wait_during = self.c.no_wait_during
        recalculate_only = self.c.recalculate_only
        prof_mode = self.c.prof_mode

        # Setup logging.
        config["logging.verbosity"] = logging.INFO
        config["logging.logging_directory"] = base_logging_path
        config["logging.logging_name"] = "CrossValidation"
        config["logging.logging_directory_use_timestamp"] = use_timestamps
        config["logging.logging_directory_use_model"] = False
        config["logging.save_log_to_file"] = True
        config["logging.log_files"] = True
        config["logging.graphs_to_file_and_show"] = True
        logging_config = LoggingConfigurator(config)

        # Save current configuration:
        FileSaver.save_string("arguments", argv)

        # Prepare cross validations:
        cross_validations = self._prepare_cross_validations(
            cross_search=self.c.cross_search
        )

        # In profiling mode use only the cross validation.
        if prof_mode:
            cross_validations = cross_validations[:1]

        # Prepare the grid search, if requested.
        grid_search = self._prepare_grid_search(self.c.grid_search)
        if len(grid_search) == 0:
            # Create dummy to perform at least one run.
            grid_search.append({ })

        # Run for each point on the parameter grid.
        for grid_params in grid_search:
            self.__l.info(f"Processing grid point {grid_params} ...")
            # Run cross-validation.
            for idx, split in enumerate(cross_validations):
                if split in completed_splits:
                    if (("grid_params" not in completed_splits[split]) or \
                        ("grid_params" in completed_splits[split] and \
                         grid_params in completed_splits[split]["grid_params"])) and \
                            not recalculate_only:
                        self.__l.info(f"Split already completed ({split}), skipping!")
                        continue
                elif recalculate_only:
                    self.__l.info(f"Performing pure recalculation, {split} is not available, skipping!")
                    continue

                # Prepare argument vector.
                split_model_argv = model_argv.copy()

                # Set parameters from the search grid.
                grid_path_spec = ""
                for parameter_name, parameter_value in grid_params.items():
                    split_model_argv[ split_model_argv.index(parameter_name) + 1 ] = str(parameter_value)
                    grid_path_spec += f"{parameter_name.replace('-', '_')}_{parameter_value}"

                if len(grid_path_spec) > 0:
                    grid_path_spec += "/"

                if recalculate_only:
                    # Load information from continuation.
                    logging_name = completed_splits[split]["logging_name"]
                    logging_path = completed_splits[split]["logging_path"]
                    logging_name_aug = next(pathlib.Path(logging_path).glob("**/evaluation")).parts[-2]
                    self.__l.info("Performing recalculation, loaded name and path!")
                else:
                    # Prepare names and paths.
                    logging_name = f"Cross_{idx}_{config['cross.base_logging_name']}"
                    timestamp_str = f"_{PerceptreeCrossMain.generate_timestamp_str()}" if use_timestamps else ""
                    logging_path = f"{base_logging_path}/{grid_path_spec}{idx}{timestamp_str}/"
                    logging_name_aug = logging_name
                    os.makedirs(logging_path, exist_ok=True)

                model_save_path = f"{logging_path}/{logging_name}.pcm"
                snapshot_path = f"{logging_path}/snapshots/"
                log_save_path = f"{logging_path}/{logging_name}.log"
                evaluation_save_path = f"{logging_path}/{logging_name}_eval.csv"
                logging_handler = logging_config.setup_file_logging_to_file(file_path=log_save_path)

                # Prepare parameter overrides:
                parameter_override = { }

                # Set common parameters
                parameter_override["--logging-directory"] = {
                    "value": logging_path, "skip": False, "mark": "Logging" }
                parameter_override["--logging-name"] = {
                    "value": logging_name_aug, "skip": False, "mark": "Logging" }
                parameter_override["--save"] = {
                    "value": model_save_path, "skip": False, "mark": ".*Predictor" }
                parameter_override["--snapshot-path"] = {
                    "value": snapshot_path, "skip": False, "mark": ".*Predictor" }
                parameter_override["--pretrained-continue"] = {
                    "value": snapshot_path, "skip": False, "mark": ".*Predictor" }
                parameter_override["--cross-validation-split"] = {
                    "value": split, "skip": False, "mark": "Featurize" }
                parameter_override["--export-evaluations"] = {
                    "value": evaluation_save_path, "skip": False, "mark": "Evaluate" }

                if recalculate_only:
                    # Disable timestamp appending.
                    parameter_override["--use-timestamp"] = {
                        "value": "False", "skip": False, "mark": "Logging" }
                    parameter_override["--use-model-names"] = {
                        "value": "False", "skip": False, "mark": "Logging" }
                    parameter_override["--use-model-names"] = {
                        "value": "False", "skip": False, "mark": "Logging" }
                    split_model_argv.remove("--train")
                    parameter_override["--save"] = {
                        "value": model_save_path, "skip": False, "mark": ".*Predictor", "replace": "--load" }

                if prof_mode:
                    # Set profiling flags if profiling is enabled.
                    parameter_override["--prof-save-dir"] = {
                        "value": logging_path, "skip": False, "mark": "Logging" }
                    parameter_override["--prof-display"] = {
                        "value": True, "skip": False, "mark": "Logging" }
                    parameter_override["--prof-enabled"] = {
                        "value": True, "skip": False, "mark": "Logging" }

                # Perform parameter override.
                split_model_argv = self._set_argv(argv=split_model_argv, values=parameter_override)

                self.__l.info(f"Running cross-validation #{idx + 1}: \n{' '.join(split_model_argv)}")

                # Run train-predict-evaluate routine.
                with ProfTimer() as t:
                    if not dry_run:
                        perceptree_main = PerceptreeMain()
                        ret_val = perceptree_main.main(split_model_argv)
                    else:
                        self.__l.info("Run skipped, performing dry run!")
                        ret_val = 0

                # Remove the logging handler before going further.
                logging_config.remove_file_logging_to_file(file_handler=logging_handler)

                if ret_val < 0:
                    self.__l.warn("Run failed with error!")
                    return
                elif not recalculate_only:
                    orig_grid_params = completed_splits[split]["grid_params"] \
                        if split in completed_splits else [ ]
                    completed_splits[split] = {
                        "cmd": " ".join(split_model_argv),
                        "logging_path": logging_path,
                        "logging_name": logging_name,
                        "model_save_path": model_save_path,
                        "split": split,
                        "grid_params": orig_grid_params + [ grid_params ]
                    }
                    if not dry_run:
                        self._save_continuation(
                            base_logging_path=base_logging_path,
                            completed_splits=completed_splits,
                            path=self.c.continuation_file
                        )

                if self.c.wait and not self.check_is_during(interval=no_wait_during):
                    self.__l.info("Press <ENTER> to continue...")
                    input()


def main(argv: Optional[List[str]] = None):
    perceptree_cross_main = PerceptreeCrossMain()
    perceptree_cross_main.main(sys.argv if argv is None else argv)


if __name__ == "__main__":
    main()
