"""Tests for Modules/diffusion/utils.py"""

import torch

from Modules.diffusion.utils import (
    closest_power_2,
    default,
    exists,
    group_dict_by_prefix,
    groupby,
    iff,
    is_sequence,
    prefix_dict,
    prod,
    rand_bool,
    to_list,
)


class TestExists:
    def test_with_none(self):
        assert exists(None) is False

    def test_with_value(self):
        assert exists(42) is True
        assert exists("hello") is True
        assert exists(0) is True
        assert exists([]) is True


class TestIff:
    def test_true_condition(self):
        assert iff(True, 42) == 42

    def test_false_condition(self):
        assert iff(False, 42) is None


class TestIsSequence:
    def test_list(self):
        assert is_sequence([1, 2, 3]) is True

    def test_tuple(self):
        assert is_sequence((1, 2, 3)) is True

    def test_non_sequence(self):
        assert is_sequence("string") is False
        assert is_sequence(42) is False
        assert is_sequence({"a": 1}) is False


class TestDefault:
    def test_with_value(self):
        assert default(42, 0) == 42

    def test_with_none_and_default_value(self):
        assert default(None, 0) == 0

    def test_with_none_and_callable(self):
        assert default(None, lambda: 100) == 100


class TestToList:
    def test_single_value(self):
        assert to_list(42) == [42]

    def test_list(self):
        assert to_list([1, 2, 3]) == [1, 2, 3]

    def test_tuple(self):
        assert to_list((1, 2, 3)) == [1, 2, 3]


class TestProd:
    def test_simple(self):
        assert prod([1, 2, 3, 4]) == 24

    def test_single_element(self):
        assert prod([5]) == 5

    def test_with_one(self):
        assert prod([1, 1, 1]) == 1


class TestClosestPower2:
    def test_exact_power_of_2(self):
        assert closest_power_2(8) == 8
        assert closest_power_2(16) == 16
        assert closest_power_2(64) == 64

    def test_round_down(self):
        assert closest_power_2(9) == 8  # closer to 8 than 16

    def test_round_up(self):
        assert closest_power_2(15) == 16  # closer to 16 than 8


class TestRandBool:
    def test_proba_one(self):
        result = rand_bool((3, 3), 1.0)
        assert result.all()
        assert result.dtype == torch.bool

    def test_proba_zero(self):
        result = rand_bool((3, 3), 0.0)
        assert not result.any()
        assert result.dtype == torch.bool

    def test_shape(self):
        result = rand_bool((2, 4), 0.5)
        assert result.shape == (2, 4)


class TestGroupDictByPrefix:
    def test_basic(self):
        d = {"model_a": 1, "model_b": 2, "other": 3}
        with_prefix, without_prefix = group_dict_by_prefix("model_", d)
        assert with_prefix == {"model_a": 1, "model_b": 2}
        assert without_prefix == {"other": 3}

    def test_no_matches(self):
        d = {"a": 1, "b": 2}
        with_prefix, without_prefix = group_dict_by_prefix("model_", d)
        assert with_prefix == {}
        assert without_prefix == {"a": 1, "b": 2}


class TestGroupby:
    def test_basic(self):
        d = {"model_a": 1, "model_b": 2, "other": 3}
        no_prefix, kwargs = groupby("model_", d)
        assert no_prefix == {"a": 1, "b": 2}
        assert kwargs == {"other": 3}

    def test_keep_prefix(self):
        d = {"model_a": 1, "model_b": 2, "other": 3}
        with_prefix, kwargs = groupby("model_", d, keep_prefix=True)
        assert with_prefix == {"model_a": 1, "model_b": 2}
        assert kwargs == {"other": 3}


class TestPrefixDict:
    def test_basic(self):
        d = {"a": 1, "b": 2}
        result = prefix_dict("pre_", d)
        assert result == {"pre_a": 1, "pre_b": 2}

    def test_empty(self):
        assert prefix_dict("pre_", {}) == {}
