# -*- coding: utf-8 -*-

"""
Utility functions and classes.
"""

import sys
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, Union

import argparse as ap
import numpy as np
import pandas as pd
import swifter


def is_debugging():
    """
    Check if the application is running under a debugger.

    :return: Returns True if debugger is attached.
    """

    return sys.gettrace() is not None


def calculate_dask_threshold():
    """
    Calculate Dask threshold of when to use parallel computation.

    :return: Returns number of seconds a computation must take before using Dask.
    """

    if is_debugging():
        # Debugger slows down thread creation, make sure it is really worth it.
        return 600
    else:
        return 1

def pd_optimal_apply(df: pd.DataFrame, l: Callable, *args, **kwargs):
    """
    Perform optimal apply on given dataframe. Uses either normal apply or
    parallel apply based on current state and operation complexity.

    :param df: DataFrame containing the input data.
    :param l: Lambda applied to the input data.
    :param args: Arguments passed to the apply function.
    :param kwargs: Arguments passed to the apply function.

    :return: Returns DataFrame which contains result of the operation application.
    """

    if is_debugging() and False:
        return df.apply(l, *args, **kwargs)
    else:
        return df.swifter\
            .progress_bar(False)\
            .set_dask_threshold(calculate_dask_threshold())\
            .set_dask_scheduler("threads")\
            .apply(l, *args, **kwargs)


def parse_bool_string(val: str) -> bool:
    """
    Parse string representation of a boolean and
    return its presumed truth value.

    Following inputs are valid "True" values:
        - "True", "true", "T", "t",
        - "Yes", "yes", "Y", "y"
        - "1"

    Following inputs are valid "False" values:
        - "False", "false", "F", "f",
        - "No", "no", "N", "n"
        - "0"

    :param val: String representation of the boolean.

    :raises argparse.ArgumentTypeError: Raised when
        given string is not any of the valid options.

    :return: Returns boolean value of given string.
    """

    if val.lower() in ("true", "t", "yes", "y", "1"):
        return True
    elif val.lower() in ("false", "f", "no", "n", "0"):
        return False
    else:
        raise ap.ArgumentTypeError(f"Unknown boolean value \"{val}\"!")


def parse_list(val: str, typ: callable, sep: str = ",") -> list:
    """
    Parse separated list of objects into python list.

    :param val: List of object, separated using separator.
    :param typ: Parser for the type used as typ(str).
    :param sep: Separator of list elements.

    :raises argparse.ArgumentTypeError: Raised when
        given string does not represent valid list.

    :return: Returns list of types produced by typ(str).
    """

    try:
        if val == sep:
            return [ ]
        items = val.split(sep)
        result = [ typ(item) for item in items ]
    except:
        raise ap.ArgumentTypeError(f"Invalid list value \"{val}\"!")

    return result


def parse_list_string(typ: callable, sep: str = ",") -> callable:
    """ Construct a list parser for given type and separator. """

    def parser(val: str) -> list:
        return parse_list(val=val, typ=typ, sep=sep)

    return parser


def reshape_scalar(val: any) -> np.array:
    """ Create array from given value, keeping dimensions for array types and creating [ val ] for scalars. """
    arr = np.array(val)
    return arr if arr.shape else arr.reshape((-1, ))


def remove_nan_inf(values: Union[List, np.array]) -> np.array:
    """ Remove NaN and INF values from given list of values and return the result. """
    arr = np.array(values)
    return arr[np.isfinite(arr)]


def dict_of_lists(values: Union[List[Tuple[any, any]], List[List[any]]]) -> Dict[any, List[any]]:
    """ Convert list of pairs into dictionary of lists. """

    result = { }

    for val in values:
        result[val[0]] = result.get(val[0], [ ])
        if len(val) > 2:
            result[val[0]].append(val[1:])
        else:
            result[val[0]].append(val[1])

    return result


def tuple_array_to_numpy(values: list, axis: int = -1) -> np.array:
    """ Convert input list of tuples into numpy array of tuples. """

    if len(values) == 0:
        return np.array([ ], dtype=object)

    result = np.empty(np.shape(values)[:axis], dtype=object)
    result[:] = values

    return result


def numpy_array_to_tuple_numpy(values: np.array) -> np.array:
    """ Convert input array into numpy array of tuples. """

    if len(values) == 0:
        return np.array([ ], dtype=object)

    result = np.empty(np.shape(values)[:-1], dtype=object)
    result[:] = [ tuple(value) for value in values ]

    return result


def recurse_dictionary_endpoint(input_dict: any, separator: str = ".") -> Iterable:
    """ Iterate over input dictionary yielding containing dict, name and full path. """

    dict_type = type(input_dict)

    def recurse_helper(path_dict, path, name):
        if name not in path_dict:
            raise RuntimeError(f"Found invalid name in recurse_dictionary_endpoint \"{name}\"!")
        elif isinstance(path_dict[name], dict_type):
            for n in path_dict[name].keys():
                yield from recurse_helper(
                    path_dict=path_dict[name],
                    path=separator.join(filter(None, [path, name])),
                    name=n
                )
        else:
            yield path_dict, name, path

    for start_name in input_dict.keys():
        yield from recurse_helper(
            path_dict=input_dict, name=start_name, path=""
        )


def recurse_dict(data: Union[List[dict], dict], raise_unaligned: bool = True,
                 only_endpoints: bool = False, key_dict: bool = False) -> Iterator:
    """
    Recursively iterate over input dictionary or a list of dictionaries.

    :param data: A single dictionary or list of dictionaries to iterate
        over. In case of multiple dictionaries, the values will be zipped.
        The first dictionary is considered as the primary and only keys
        from it are used - other dictionaries may contain other keys as
        well, but will be unused!
    :param raise_unaligned: Set to True in order to throw raise RuntimeError
        in case when dictionaries are not aligned with their content.
    :param only_endpoints: Set to True to return only end-points. In case
        of multiple dictionaries, all of the values must be end-points!
    :param key_dict: Set to True to return holder dictionary and key instead
        of the end values themselves. Keep at False to return only end-point
        values.

    :return: Returns recursive dictionary iterator. Values are returned in
        depth-first order. Only keys shared by all dictionaries will be
        iterated over!
    """

    if len(data) == 0:
        return

    data = tuple([ data ]) if isinstance(data, dict) else tuple(data)
    # Start processing at the first key in all of the dictionaries.
    processing_stack = [ tuple( [ iter(data[0].keys()) ] ) + data ]

    while len(processing_stack) != 0:
        current_rec = processing_stack.pop(-1)

        src_key_iter = current_rec[0]
        src_dicts = current_rec[1:]

        src_key_valid = False
        while not src_key_valid:
            try:
                src_key = next(src_key_iter)
            except StopIteration as e:
                # No more keys available -> End.
                break

            # Check whether the key is in all of the dictionaries.
            key_presence = [ src_key in d for d in src_dicts ]
            src_key_valid = np.any(key_presence)
            if raise_unaligned and not np.all(key_presence):
                # Throwing is requested and not all dictionaries contain the key.
                raise RuntimeError(f"Unaligned dictionaries in recurse_dict, key \"{src_key}\"")

        if not src_key_valid:
            # No valid key found -> Continue processing.
            continue

        # Recover dictionaries in the target:
        src_dst_dicts = tuple( d for d in src_dicts if src_key in d )
        dst_dicts = tuple( d[src_key] for d in src_dst_dicts )
        dst_endpoints = [ not isinstance(d, dict) for d in dst_dicts ]
        dst_any_endpoint = np.any(dst_endpoints)
        dst_all_endpoint = np.all(dst_endpoints)

        # Add the original with moved iterator.
        processing_stack.append(tuple([ src_key_iter ]) + src_dicts)
        # Add the new ones if no endpoints are present.
        if not dst_any_endpoint:
            dst_key_iter = iter(dst_dicts[ 0 ].keys())
            processing_stack.append(tuple([ dst_key_iter ]) + dst_dicts)

        if raise_unaligned and dst_any_endpoint and not dst_all_endpoint:
            # Only some of the dictionaries contain endpoint.
            raise RuntimeError(f"Only some of the dictionaries contain endpoints ({dst_dicts})!")

        if only_endpoints and dst_any_endpoint:
            # Yielding only endpoints:
            if key_dict:
                yield tuple([ src_key ]) + tuple(
                    parent
                    for parent, is_endpoint in zip(src_dst_dicts, dst_endpoints)
                    if is_endpoint
                )
            else:
                yield tuple([ src_key ]) + tuple(
                    child
                    for child, is_endpoint in zip(dst_dicts, dst_endpoints)
                    if is_endpoint
                )
        elif not only_endpoints:
            # Yielding all:
            if key_dict:
                yield tuple([ src_key ]) + tuple(src_dst_dicts)
            else:
                yield tuple([ src_key ]) + tuple(dst_dicts)

    return


def numpy_zero(dtype: np.dtype) -> any:
    """ Return zero of given dtype. """
    return np.zeros(1, dtype=dtype)[0]


def numpy_op_or_zero(arr: np.array, op: Callable) -> any:
    """ Perform operation on given list of not empty, else return zero. """
    return op(arr) if len(arr) else numpy_zero(dtype=arr.dtype)

