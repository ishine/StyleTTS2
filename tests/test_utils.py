"""Tests for utils.py and Modules/utils.py"""

import torch
from munch import Munch

from Modules.utils import get_padding
from utils import length_to_mask, log_norm, recursive_munch


class TestLengthToMask:
    def test_basic(self):
        lengths = torch.tensor([2, 3, 1])
        mask = length_to_mask(lengths)
        expected = torch.tensor(
            [
                [False, False, True],  # length 2: first 2 valid
                [False, False, False],  # length 3: all valid
                [False, True, True],  # length 1: only first valid
            ]
        )
        assert torch.equal(mask, expected)

    def test_single_length(self):
        lengths = torch.tensor([3])
        mask = length_to_mask(lengths)
        assert mask.shape == (1, 3)
        assert not mask.any()  # all valid (False = not masked)

    def test_different_lengths(self):
        lengths = torch.tensor([1, 4])
        mask = length_to_mask(lengths)
        assert mask.shape == (2, 4)


class TestLogNorm:
    def test_output_shape(self):
        x = torch.randn(2, 80, 100)  # (batch, mel_bins, time)
        result = log_norm(x)  # default dim=2 reduces along time dimension
        assert result.shape == (2, 80)  # reduced along time dimension

    def test_different_dim(self):
        x = torch.randn(2, 80, 100)
        result = log_norm(x, dim=1)  # reduces along mel dimension
        assert result.shape == (2, 100)


class TestRecursiveMunch:
    def test_simple_dict(self):
        d = {"a": 1, "b": 2}
        result = recursive_munch(d)
        assert isinstance(result, Munch)
        assert result.a == 1
        assert result.b == 2

    def test_nested_dict(self):
        d = {"a": {"b": {"c": 3}}}
        result = recursive_munch(d)
        assert isinstance(result, Munch)
        assert isinstance(result.a, Munch)
        assert isinstance(result.a.b, Munch)
        assert result.a.b.c == 3

    def test_list_in_dict(self):
        d = {"data": [{"x": 1}, {"x": 2}]}
        result = recursive_munch(d)
        assert isinstance(result, Munch)
        assert isinstance(result.data, list)
        assert isinstance(result.data[0], Munch)
        assert result.data[0].x == 1

    def test_non_dict(self):
        assert recursive_munch(42) == 42
        assert recursive_munch("string") == "string"


class TestGetPadding:
    def test_kernel_3(self):
        # kernel_size=3, dilation=1: (3*1 - 1) / 2 = 1
        assert get_padding(3) == 1

    def test_kernel_5(self):
        # kernel_size=5, dilation=1: (5*1 - 1) / 2 = 2
        assert get_padding(5) == 2

    def test_with_dilation(self):
        # kernel_size=3, dilation=2: (3*2 - 2) / 2 = 2
        assert get_padding(3, dilation=2) == 2

    def test_kernel_7_dilation_3(self):
        # kernel_size=7, dilation=3: (7*3 - 3) / 2 = 9
        assert get_padding(7, dilation=3) == 9
