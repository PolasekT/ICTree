# -*- coding: utf-8 -*-

"""
Helper used for logging messages.
"""

import logging

from progress.bar import Bar
from progress import monotonic
from typing import Callable, Optional

from perceptree.common.mailer import Mailer
from perceptree.common.profiler import Profiler


class LoggingBar(Bar):
    enabled = True
    """
    Enable for progress bar printing.
    """

    def clearln(self):
        """
        Overwriting so we can disable bar printing
        if enabled == False.
        """

        if LoggingBar.enabled:
            super().clearln()

    def write(self, s):
        """
        Overwriting so we can disable bar printing
        if enabled == False.
        """

        if LoggingBar.enabled:
            super().write(s)

    def writeln(self, line):
        """
        Overwriting so we can disable bar printing
        if enabled == False.
        """

        if LoggingBar.enabled:
            super().write(line)


class DumpingBar(LoggingBar):
    message = "Dumping"
    suffix = "%(index)d/%(max)d (%(percent).1f%%) : %(eta)s seconds remaining"


class CopyBar(LoggingBar):
    message = "Copying"
    suffix = "%(index)d/%(max)d (%(percent).1f%%) : %(eta)s seconds remaining"


class FittingBar(LoggingBar):
    message = "Fitting"
    loss = ""
    suffix = "%(index)d/%(max)d (%(percent).1f%%) : %(loss)s : %(eta)s s remaining (-> %(elapsed)s s)"


class ParsingBar(LoggingBar):
    message = "Parsing"
    suffix = "%(index)d/%(max)d (%(percent).1f%%) : %(eta)s seconds remaining"


class ProcessingBar(LoggingBar):
    message = "Processing"
    suffix = "%(index)d/%(max)d (%(percent).1f%%) : %(eta)s seconds remaining"


class LoadingBar(LoggingBar):
    message = "Loading"
    suffix = "%(index)d/%(max)d (%(percent).1f%%) : %(eta)s seconds remaining"


class PredictionBar(LoggingBar):
    message = "Predicting"
    suffix = "%(index)d/%(max)d (%(percent).1f%%) : %(eta)s seconds remaining"


class LogMeta(type):
    """
    Meta-class which generates self.__l self.__m for
    each class which uses it.

    Inspired by: https://stackoverflow.com/a/50731615 .
    """

    directory_prefix = ""
    """
    Prefix used for all of the saved files.
    """

    log_path = ""
    """
    Path to the currently used log
    """

    loggers = [ ]
    """
    List of currently registered loggers. 
    """

    def __init__(cls, *args):
        super().__init__(*args)

        # Explicit name mangling
        logger_attribute_name = f"_{cls.__name__}__l"
        mailer_attribute_name = f"_{cls.__name__}__m"
        profiler_attribute_name = '_' + cls.__name__ + '__prof'

        # Logger name derived accounting for inheritance
        logger_name = '.'.join([c.__name__ for c in cls.mro()[-2::-1]])

        logger = logging.getLogger(logger_name)
        LogMeta._add_handlers(logger, LogMeta.log_path)
        LogMeta.loggers.append(logger)

        setattr(cls, logger_attribute_name, logging.getLogger(logger_name))
        setattr(cls, mailer_attribute_name, Mailer())
        setattr(cls, profiler_attribute_name, Profiler(cls.__name__))

    @classmethod
    def _remove_all_handlers(mcs):
        """ Remove all logging handlers. """

        for logger in mcs.loggers:
            handlers = logger.handlers.copy()
            for handler in handlers:
                logger.removeHandler(handler)

        mcs.loggers = [ ]

    @classmethod
    def _add_handlers(mcs, logger: logging.Logger, log_path: str):
        """ Add handlers to given logger if they don't exist yet. """

        if len(log_path) == 0 or len(logger.handlers) > 10:
            return

        file_handler = logging.FileHandler(filename=log_path)
        file_handler.setFormatter(logging.root.handlers[0].formatter)
        logger.addHandler(file_handler)

    @classmethod
    def _remove_handlers(mcs, logger: logging.Logger, file_handler: logging.FileHandler):
        """ Remove handlers from given logger. """

        logger.removeHandler(file_handler)

    @classmethod
    def setup_file_logging(mcs, base_path: str):
        """ Setup logging to file in given base path. """

        mcs.directory_prefix = base_path
        mcs.log_path = f"{base_path}/runtime.log"

        for logger in mcs.loggers:
            mcs._add_handlers(logger, mcs.log_path)

    @classmethod
    def setup_file_logging_to_file(mcs, file_path: str):
        """ Setup logging to given file path. """

        for logger in mcs.loggers:
            mcs._add_handlers(logger, file_path)

    @classmethod
    def remove_file_logging_to_file(mcs, file_handler: logging.FileHandler):
        """ Remove logging to given file handler. """

        for logger in mcs.loggers:
            mcs._remove_handlers(logger, file_handler)


class Logger(metaclass=LogMeta):
    """
    Inheritable helper, which allows any class to
    access its own self.__l and self.__m.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def last_logger(self) -> logging.Logger:
        """ Access the real logger for current self. """
        return getattr(self.__class__, f"_{self.__class__.__name__}__logger")

    def last_mailer(self) -> Mailer:
        """ Access the real mailer for current self. """
        return getattr(self.__class__, f"_{self.__class__.__name__}__mailer")

    def last_prof(self) -> Profiler:
        """ Access the real profiler for current self. """
        return getattr(self.__class__, f"_{self.__class__.__name__}__prof")


def profiled(name: Optional[str] = None, use_last: bool = True) -> Callable:
    def decorator(func: Callable) -> Callable:
        nonlocal name
        name = name or func.__name__

        def wrapper(*args, **kwargs):
            if len(args) > 0 and isinstance(args[0], Logger):
                prof = args[0].last_prof() if use_last else args[0].__prof
            else:
                prof = Profiler(class_name="Base")

            with prof.profile_scope(prof_path=name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

