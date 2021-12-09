# -*- coding: utf-8 -*-

"""
Simple automated profiler for collecting quantitative data.
"""

import datetime
import pathlib
import re
import timeit
from typing import Callable, List, Optional

from perceptree.common.cache import Cache


class ProfTimer(object):
    """
    Timer used for profiling.
    """

    def __init__(self, enter_cb: Optional[Callable] = None, exit_cb: Optional[Callable] = None):
        self._start_time = None
        self._enter_cb = enter_cb
        self._exit_cb = exit_cb

        self.reset()

    @staticmethod
    def _get_current_time() -> float:
        return timeit.default_timer()

    @staticmethod
    def _get_elapsed_time(start_seconds: float, end_seconds: float) -> datetime.timedelta:
        return datetime.timedelta(seconds=end_seconds - start_seconds)

    @staticmethod
    def _format_timestamp_difference(start_seconds: float, end_seconds: float) -> str:
        return str(ProfTimer._get_elapsed_time(start_seconds, end_seconds))

    def reset(self):
        """ Reset the timer. """
        self._start_time = ProfTimer._get_current_time()

    def elapsed(self) -> datetime.timedelta:
        return ProfTimer._get_elapsed_time(
            start_seconds=self._start_time,
            end_seconds=self._get_current_time()
        )

    def __str__(self) -> str:
        """ Get timer information as a string. """
        current_time = ProfTimer._get_current_time()
        difference_str = ProfTimer._format_timestamp_difference(
            start_seconds=self._start_time,
            end_seconds=current_time
        )
        return f"Timer: < {self._start_time} - {current_time} > -> {difference_str}"

    def __enter__(self):
        """ Start the timer. """
        self.reset()

        if self._enter_cb is not None:
            self._enter_cb(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Print timer info on scope exit. """

        if self._exit_cb is not None:
            self._exit_cb(self)


def check_profiler_enabled(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        if Profiler.enabled:
            return func(*args, **kwargs)
        else:
            return None

    return wrapper


class DummyScope(object):
    """ Dummy object used when profiling is disabled. """

    def __enter__(self): pass

    def __exit__(self, exc_type, exc_val, exc_tb): pass


def check_profiler_enabled_scope(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        if Profiler.enabled:
            return func(*args, **kwargs)
        else:
            return DummyScope()

    return wrapper


class Profiler(object):
    """
    Simple automated profiler for collecting quantitative data.

    :param class_name: Name of the class profiling for.
    """

    enabled = True
    """
    Enable graph saving.
    """

    @classmethod
    def configure(cls, collect_data: bool, display_data: bool,
                  save_results_dir: Optional[str]):
        """
        Configure the profiler for operation.

        :param collect_data: Enable data collection?
        :param display_data: Enable data displaying?
        :param save_results_dir: Save results under given path.
        """

        cls.enabled = collect_data
        cls.display = display_data
        cls.save_results_dir = save_results_dir
        cls.prof = None

        cls.reset_all_data()

    @classmethod
    def reset_all_data(cls):
        """ Reset all profiling information. """

        cls.prof = Cache()

    @classmethod
    def _record_prof_data(cls, class_name: str, prof_path: str, data: any):
        """ Record given data under a class and profiling path. """

        full_path = f"{class_name}{Cache.CACHE_PATH_SEP}{prof_path}{Cache.CACHE_PATH_SEP}values"
        prof_dict = cls.prof.get_path_dict(
            path=full_path, create=True,
            none_when_missing=False
        )

        prof_dict["values"] = prof_dict.get("values", [ ].copy()) + [ data ]

    @classmethod
    def display_results(cls, logger):
        """ Display the results if enabled. """

        if not cls.display:
            return

        logger.info(f"Profiling results: \n{cls.prof.cache}")

    @classmethod
    def save_results(cls):
        """ Save the results if saving is enabled. """

        if cls.save_results_dir is not None:
            save_dir = pathlib.Path(cls.save_results_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "profiling_data.pk"

            cls.prof.save_cache_pickle_to_file(path=save_path)

    def __init__(self, class_name: str):
        self._class_name = class_name

    @check_profiler_enabled
    def record_data(self, prof_path: str, data: any):
        """ Save given profiling data under a path. """

        Profiler._record_prof_data(
            class_name=self._class_name,
            prof_path=prof_path,
            data=data
        )

    @check_profiler_enabled
    def record_timer(self, prof_path: str, timer: ProfTimer):
        """ Record current elapsed time under given name. """

        self.record_data(prof_path=prof_path, data=timer.elapsed())

    @check_profiler_enabled_scope
    def profile_scope(self, prof_path: str) -> ProfTimer:
        """ Profile scope or until __exit__ is called. """

        return ProfTimer(
            enter_cb=None,
            exit_cb=lambda x : self.record_timer(timer=x, prof_path=prof_path)
        )

