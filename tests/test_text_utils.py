"""Tests for text_utils.py"""

import pytest

from text_utils import TextCleaner, _pad, dicts, symbols


class TestSymbols:
    def test_symbols_not_empty(self):
        assert len(symbols) > 0

    def test_symbols_starts_with_pad(self):
        assert symbols[0] == _pad

    def test_symbols_contains_letters(self):
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
            assert char in symbols

    def test_symbols_contains_punctuation(self):
        for char in ";:,.!? ":
            assert char in symbols

    def test_symbols_count(self):
        # symbols list has 180 items (with 3 duplicates due to overlapping chars: " x3, ' x2)
        assert len(symbols) == 180


class TestDicts:
    def test_dicts_has_unique_symbols(self):
        # dicts is a dictionary, so it only contains unique keys
        # symbols has 180 items with 3 duplicates, so dicts has 177
        assert len(dicts) == 177

    def test_pad_is_index_zero(self):
        assert dicts[_pad] == 0

    def test_all_symbols_in_dicts(self):
        for symbol in symbols:
            assert symbol in dicts

    def test_indices_cover_range(self):
        # Due to duplicate in symbols, dicts values skip one index
        indices = sorted(dicts.values())
        # All indices should be valid (0 to 177)
        assert min(indices) == 0
        assert max(indices) == len(symbols) - 1


class TestTextCleaner:
    @pytest.fixture
    def cleaner(self, capsys):
        cleaner = TextCleaner()
        capsys.readouterr()  # Clear the print output from __init__
        return cleaner

    def test_simple_text(self, cleaner):
        result = cleaner("hello")
        assert len(result) == 5
        assert all(isinstance(i, int) for i in result)

    def test_with_punctuation(self, cleaner):
        result = cleaner("Hello, World!")
        assert len(result) == 13

    def test_with_space(self, cleaner):
        result = cleaner("a b")
        assert len(result) == 3
        # Space should be in the punctuation
        assert result[1] == dicts[" "]

    def test_consistent_mapping(self, cleaner):
        # Same character should always map to same index
        result1 = cleaner("aaa")
        assert result1[0] == result1[1] == result1[2]

    def test_pad_character(self, cleaner):
        result = cleaner("$")
        assert result == [0]  # _pad is index 0

    def test_ipa_characters(self, cleaner):
        # Test some IPA characters
        if "ə" in dicts:
            result = cleaner("ə")
            assert len(result) == 1
            assert result[0] == dicts["ə"]
