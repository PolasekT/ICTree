# -*- coding: utf-8 -*-

"""
Testing of the caching system.
"""

import pytest

from perceptree.common.cache import *


class TestCache:

    @pytest.fixture()
    def cache(self) -> Cache:
        return Cache()

    def test_init(self):
        cache = Cache()

    def test_simple_items(self, cache: Cache):
        assert cache["abc"] is None
        cache["abc"] = 42
        assert cache["abc"] == 42

    def test_nested_items(self, cache: Cache):
        assert cache["abc.def"] is None
        cache["abc.def"] = 42
        assert cache["abc.def"] == 42

    def test_serialization(self):
        cache1 = Cache()
        cache1["abc.def"] = 42
        assert cache1["abc.def"] == 42

        cache2 = Cache()
        assert cache2["abc.def"] is None

        cache2.load_cache_yaml(config=cache1.save_cache_yaml())
        assert cache2["abc.def"] == 42

