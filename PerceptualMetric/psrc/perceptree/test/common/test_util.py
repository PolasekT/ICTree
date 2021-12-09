# -*- coding: utf-8 -*-

"""
Testing of the util module.
"""

import pytest

from perceptree.common.util import *


class TestUtil:

    def test_reshape_scalar(self):
        assert reshape_scalar(val=1) == np.array([ 1 ])
        assert reshape_scalar(val=[ 1 ]) == np.array([ 1 ])
        assert reshape_scalar(val=[ [ 1 ] ]) == np.array([ [ 1 ] ])

    def test_dict_of_lists_simple1(self):
        assert dict_of_lists(values=[
            ("a", 1), ("a", 2), ("a", 2),
            ("b", 4)
        ]) == {
            "a": [ 1, 2, 2 ],
            "b": [ 4 ]
        }

    def test_dict_of_lists_simple2(self):
        assert dict_of_lists(values=[
            ["a", 1], ["a", 2], ["a", 2],
            ["b", 4]
        ]) == {
            "a": [ 1, 2, 2 ],
            "b": [ 4 ]
        }

    def test_dict_of_lists_multi1(self):
        assert dict_of_lists(values=[
            ("a", 1, 2), ("a", 2, 2), ("a", 2, 2),
            ("b", 4, 4)
        ]) == {
           "a": [ ( 1, 2 ), ( 2, 2 ), ( 2, 2 ) ],
           "b": [ ( 4, 4 ) ]
        }

    def test_dict_of_lists_multi2(self):
        assert dict_of_lists(values=[
            ["a", 1, 2], ["a", 2, 2], ["a", 2, 2],
            ["b", 4, 4]
        ]) == {
           "a": [ [ 1, 2 ], [ 2, 2 ], [ 2, 2 ] ],
           "b": [ [ 4, 4 ] ]
       }

    def test_tuple_array_to_numpy(self):
        a = [ (1, 2), (3, 4) ]
        assert np.array(a).tolist() != a
        assert tuple_array_to_numpy(a).tolist() == a

    def test_recurse_dict_single(self):
        a = { "a": 1, "b": { "b1": 20, "b2": 21 }, "c": 3 }

        it = recurse_dict(data=a, raise_unaligned=False, only_endpoints=False, key_dict=False)
        assert next(it) == ( "a", 1 )
        assert next(it) == ( "b", a["b"] )
        assert next(it) == ( "b1", 20 )
        assert next(it) == ( "b2", 21 )
        assert next(it) == ( "c", 3 )
        with pytest.raises(StopIteration):
            next(it)

        it = recurse_dict(data=a, raise_unaligned=False, only_endpoints=True, key_dict=False)
        assert next(it) == ( "a", 1 )
        assert next(it) == ( "b1", 20 )
        assert next(it) == ( "b2", 21 )
        assert next(it) == ( "c", 3 )
        with pytest.raises(StopIteration):
            next(it)

        it = recurse_dict(data=a, raise_unaligned=False, only_endpoints=False, key_dict=True)
        assert next(it) == ( "a", a )
        assert next(it) == ( "b", a )
        assert next(it) == ( "b1", a["b"] )
        assert next(it) == ( "b2", a["b"] )
        assert next(it) == ( "c", a )
        with pytest.raises(StopIteration):
            next(it)

    def test_recurse_dict_multi(self):
        a = { "a": 1, "b": { "b1": 20, "b2": 21 }, "c": 3 }
        b = { "a": 11, "b": { "b1": 22, "b2": 23 }, "c": 33 }

        it = recurse_dict(data=[ a, b ], raise_unaligned=False, only_endpoints=False, key_dict=False)
        assert next(it) == ( "a", 1, 11 )
        assert next(it) == ( "b", a["b"], b["b"] )
        assert next(it) == ( "b1", 20, 22 )
        assert next(it) == ( "b2", 21, 23 )
        assert next(it) == ( "c", 3, 33 )
        with pytest.raises(StopIteration):
            next(it)

        it = recurse_dict(data=[ a, b ], raise_unaligned=False, only_endpoints=False, key_dict=True)
        assert next(it) == ( "a", a, b )
        assert next(it) == ( "b", a, b )
        assert next(it) == ( "b1", a["b"], b["b"] )
        assert next(it) == ( "b2", a["b"], b["b"] )
        assert next(it) == ( "c", a, b )
        with pytest.raises(StopIteration):
            next(it)
