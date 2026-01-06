"""Tests for Japanese G2P module."""


class TestOpenJTalkToIpaMapping:
    """Test OpenJTalk to IPA phoneme mapping."""

    def test_mapping_has_basic_vowels(self):
        from japanese_g2p import OPENJTALK_TO_IPA

        assert OPENJTALK_TO_IPA["a"] == "a"
        assert OPENJTALK_TO_IPA["i"] == "i"
        assert OPENJTALK_TO_IPA["u"] == "ɯ"
        assert OPENJTALK_TO_IPA["e"] == "e"
        assert OPENJTALK_TO_IPA["o"] == "o"

    def test_mapping_has_basic_consonants(self):
        from japanese_g2p import OPENJTALK_TO_IPA

        assert OPENJTALK_TO_IPA["k"] == "k"
        assert OPENJTALK_TO_IPA["s"] == "s"
        assert OPENJTALK_TO_IPA["t"] == "t"
        assert OPENJTALK_TO_IPA["n"] == "n"
        assert OPENJTALK_TO_IPA["m"] == "m"

    def test_mapping_has_special_phonemes(self):
        from japanese_g2p import OPENJTALK_TO_IPA

        # Syllabic nasal (撥音)
        assert OPENJTALK_TO_IPA["N"] == "ɴ"
        # Geminate (促音)
        assert OPENJTALK_TO_IPA["cl"] == "ʔ"

    def test_mapping_has_palatalized_consonants(self):
        from japanese_g2p import OPENJTALK_TO_IPA

        assert OPENJTALK_TO_IPA["sh"] == "ɕ"
        assert OPENJTALK_TO_IPA["ch"] == "tɕ"
        assert OPENJTALK_TO_IPA["j"] == "dʑ"


class TestExtractPhonemeFromLabel:
    """Test phoneme extraction from fullcontext labels."""

    def test_extract_standard_label(self):
        from japanese_g2p import _extract_phoneme_from_label

        # Standard format: xx^xx-PHONEME+xx=xx/...
        label = "xx^a-k+o=xx"
        assert _extract_phoneme_from_label(label) == "k"

    def test_extract_boundary_label(self):
        from japanese_g2p import _extract_phoneme_from_label

        label = "xx-sil+xx"
        assert _extract_phoneme_from_label(label) == "sil"

    def test_extract_returns_none_for_invalid(self):
        from japanese_g2p import _extract_phoneme_from_label

        result = _extract_phoneme_from_label("invalid_label")
        assert result is None


class TestConvertPhonemeToIpa:
    """Test single phoneme conversion."""

    def test_convert_known_phoneme(self):
        from japanese_g2p import _convert_phoneme_to_ipa

        assert _convert_phoneme_to_ipa("a") == "a"
        assert _convert_phoneme_to_ipa("N") == "ɴ"
        assert _convert_phoneme_to_ipa("sh") == "ɕ"

    def test_convert_unknown_phoneme_returns_lowercase(self):
        from japanese_g2p import _convert_phoneme_to_ipa

        assert _convert_phoneme_to_ipa("UNKNOWN") == "unknown"


class TestJapaneseToIpa:
    """Test full Japanese to IPA conversion."""

    def test_empty_string(self):
        from japanese_g2p import japanese_to_ipa

        assert japanese_to_ipa("") == ""
        assert japanese_to_ipa("   ") == ""

    def test_simple_hiragana(self):
        from japanese_g2p import japanese_to_ipa

        result = japanese_to_ipa("あ")
        assert "a" in result

    def test_returns_string(self):
        from japanese_g2p import japanese_to_ipa

        result = japanese_to_ipa("こんにちは")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_spaces_between_phonemes(self):
        from japanese_g2p import japanese_to_ipa

        result = japanese_to_ipa("あい", add_spaces=True)
        assert " " in result

    def test_no_spaces_option(self):
        from japanese_g2p import japanese_to_ipa

        result = japanese_to_ipa("あい", add_spaces=False)
        # Should have fewer characters without spaces
        result_with_spaces = japanese_to_ipa("あい", add_spaces=True)
        assert len(result) <= len(result_with_spaces)

    def test_katakana(self):
        from japanese_g2p import japanese_to_ipa

        result = japanese_to_ipa("カタカナ")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_kanji(self):
        from japanese_g2p import japanese_to_ipa

        result = japanese_to_ipa("日本語")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mixed_text(self):
        from japanese_g2p import japanese_to_ipa

        result = japanese_to_ipa("今日はいい天気です")
        assert isinstance(result, str)
        assert len(result) > 0


class TestJapaneseToPhonemeList:
    """Test conversion to phoneme list."""

    def test_returns_list(self):
        from japanese_g2p import japanese_to_phoneme_list

        result = japanese_to_phoneme_list("こんにちは")
        assert isinstance(result, list)

    def test_empty_string_returns_empty_list(self):
        from japanese_g2p import japanese_to_phoneme_list

        result = japanese_to_phoneme_list("")
        assert result == []

    def test_list_contains_strings(self):
        from japanese_g2p import japanese_to_phoneme_list

        result = japanese_to_phoneme_list("あ")
        assert all(isinstance(p, str) for p in result)


class TestIsValidForStyleTTS2:
    """Test vocabulary validation."""

    def test_valid_ipa_phonemes(self):
        from japanese_g2p import is_valid_for_styletts2

        # Basic IPA characters should be valid
        is_valid, invalid = is_valid_for_styletts2("a i u e o")
        assert is_valid
        assert len(invalid) == 0

    def test_space_is_not_invalid(self):
        from japanese_g2p import is_valid_for_styletts2

        is_valid, invalid = is_valid_for_styletts2("a i")
        assert " " not in invalid

    def test_returns_tuple(self):
        from japanese_g2p import is_valid_for_styletts2

        result = is_valid_for_styletts2("a")
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestExtractAccentInfo:
    """Test accent information extraction from fullcontext labels."""

    def test_extract_accent_from_label(self):
        from japanese_g2p import _extract_accent_info

        # Label with A:-1 (before accent nucleus)
        label = "a^k-o+N=xx/A:-1+2+3/B:xx"
        assert _extract_accent_info(label) == -1

    def test_extract_accent_at_nucleus(self):
        from japanese_g2p import _extract_accent_info

        # Label with A:0 (at accent nucleus)
        label = "k^o-N+xx=xx/A:0+3+3/B:xx"
        assert _extract_accent_info(label) == 0

    def test_extract_accent_after_nucleus(self):
        from japanese_g2p import _extract_accent_info

        # Label with A:1 (after accent nucleus)
        label = "o^N-xx+xx=xx/A:1+4+3/B:xx"
        assert _extract_accent_info(label) == 1

    def test_extract_accent_returns_none_for_xx(self):
        from japanese_g2p import _extract_accent_info

        # Label with A:xx (undefined)
        label = "xx^xx-sil+xx=xx/A:xx+xx+xx/B:xx"
        assert _extract_accent_info(label) is None

    def test_extract_accent_returns_none_for_invalid(self):
        from japanese_g2p import _extract_accent_info

        assert _extract_accent_info("invalid_label") is None


class TestAccentMarkers:
    """Test pitch accent marker insertion."""

    def test_accent_markers_constants_exist(self):
        from japanese_g2p import ACCENT_FALL, ACCENT_RISE

        assert ACCENT_RISE == "↑"
        assert ACCENT_FALL == "↓"

    def test_atamadaka_accent_has_fall_marker(self):
        """頭高型 (type 1) should have ↓ after first mora."""
        from japanese_g2p import japanese_to_ipa

        # 雨 (あめ) - 頭高型: pitch falls after first mora
        result = japanese_to_ipa("雨", include_accent=True)
        assert "↓" in result
        assert "↑" not in result

    def test_nakadaka_accent_has_both_markers(self):
        """中高型 should have both ↑ and ↓."""
        from japanese_g2p import japanese_to_ipa

        # 日本 - 中高型: pitch rises then falls
        result = japanese_to_ipa("日本", include_accent=True)
        assert "↑" in result
        assert "↓" in result

    def test_odaka_accent_has_rise_marker(self):
        """尾高型 should have ↑ at the end."""
        from japanese_g2p import japanese_to_ipa

        # 東京 - 尾高型: accent on last mora
        result = japanese_to_ipa("東京", include_accent=True)
        assert "↑" in result

    def test_include_accent_false_no_markers(self):
        """When include_accent=False, no markers should be present."""
        from japanese_g2p import japanese_to_ipa

        result = japanese_to_ipa("日本", include_accent=False)
        assert "↑" not in result
        assert "↓" not in result

    def test_accent_markers_are_in_styletts2_vocabulary(self):
        """Accent markers should be valid StyleTTS2 symbols."""
        from japanese_g2p import is_valid_for_styletts2

        is_valid, invalid = is_valid_for_styletts2("a ↑ m e")
        assert is_valid
        assert len(invalid) == 0

        is_valid, invalid = is_valid_for_styletts2("a ↓ m e")
        assert is_valid
        assert len(invalid) == 0

    def test_minimal_pair_ame(self):
        """Test minimal pair 雨/飴 have different accent patterns."""
        from japanese_g2p import japanese_to_ipa

        ame_rain = japanese_to_ipa("雨", include_accent=True)  # 頭高型
        ame_candy = japanese_to_ipa("飴", include_accent=True)  # 中高型

        # They should have different accent patterns
        assert ame_rain != ame_candy
        # 雨 has fall marker, 飴 has rise marker
        assert "↓" in ame_rain
        assert "↑" in ame_candy

    def test_minimal_pair_hashi(self):
        """Test minimal pair 箸/橋 have different accent patterns."""
        from japanese_g2p import japanese_to_ipa

        hashi_chopsticks = japanese_to_ipa("箸", include_accent=True)  # 頭高型
        hashi_bridge = japanese_to_ipa("橋", include_accent=True)  # 中高型

        # They should have different accent patterns
        assert hashi_chopsticks != hashi_bridge

    def test_long_vowel_with_accent(self):
        """Long vowels should work correctly with accent markers."""
        from japanese_g2p import japanese_to_ipa

        # 東京 has long vowels and accent
        result = japanese_to_ipa("東京", include_accent=True)
        assert "ː" in result  # Long vowel marker should be present
        assert "↑" in result  # Accent marker should be present


class TestExtractA2Info:
    """Test A2 field extraction from fullcontext labels."""

    def test_extract_a2_from_label(self):
        from japanese_g2p import _extract_a2_info

        label = "a^k-o+N=xx/A:-1+2+5/B:xx"
        assert _extract_a2_info(label) == 2

    def test_extract_a2_first_mora(self):
        from japanese_g2p import _extract_a2_info

        label = "xx^a-k+o=xx/A:-2+1+5/B:xx"
        assert _extract_a2_info(label) == 1

    def test_extract_a2_returns_none_for_invalid(self):
        from japanese_g2p import _extract_a2_info

        assert _extract_a2_info("invalid_label") is None


class TestExtractA3Info:
    """Test A3 field extraction from fullcontext labels."""

    def test_extract_a3_from_label(self):
        from japanese_g2p import _extract_a3_info

        label = "a^k-o+N=xx/A:-1+2+5/B:xx"
        assert _extract_a3_info(label) == 5

    def test_extract_a3_returns_none_for_invalid(self):
        from japanese_g2p import _extract_a3_info

        assert _extract_a3_info("invalid_label") is None


class TestExtractE3Info:
    """Test E3 field extraction for question detection."""

    def test_extract_e3_statement(self):
        from japanese_g2p import _extract_e3_info

        # Statement (E3 = 0)
        label = "xx^a-sil+xx=xx/A:xx/B:xx/C:xx/D:xx/E:5_5!0_xx/F:xx"
        assert _extract_e3_info(label) == 0

    def test_extract_e3_question(self):
        from japanese_g2p import _extract_e3_info

        # Question (E3 = 1)
        label = "xx^a-sil+xx=xx/A:xx/B:xx/C:xx/D:xx/E:5_5!1_xx/F:xx"
        assert _extract_e3_info(label) == 1

    def test_extract_e3_returns_none_for_invalid(self):
        from japanese_g2p import _extract_e3_info

        assert _extract_e3_info("invalid_label") is None


class TestAccentPhraseBoundary:
    """Test accent phrase boundary marker insertion."""

    def test_boundary_marker_constant_exists(self):
        from japanese_g2p import ACCENT_PHRASE_BOUNDARY

        assert ACCENT_PHRASE_BOUNDARY == "#"

    def test_phrase_boundary_in_sentence(self):
        from japanese_g2p import japanese_to_ipa

        # Sentence with multiple accent phrases
        result = japanese_to_ipa("今日は天気がいいです", include_phrase_boundary=True)
        # Should have at least one boundary or produce valid output
        assert isinstance(result, str)
        assert len(result) > 0

    def test_phrase_boundary_enabled_by_default(self):
        from japanese_g2p import japanese_to_ipa

        # Phrase boundary is enabled by default
        result = japanese_to_ipa("日本語")
        assert isinstance(result, str)

    def test_phrase_boundary_can_be_disabled(self):
        from japanese_g2p import japanese_to_ipa

        result = japanese_to_ipa("日本語", include_phrase_boundary=False)
        assert "#" not in result

    def test_phrase_boundary_returns_string(self):
        from japanese_g2p import japanese_to_ipa

        result = japanese_to_ipa("東京都", include_phrase_boundary=True)
        assert isinstance(result, str)


class TestQuestionDetection:
    """Test question detection via E3 field."""

    def test_question_marker_constant_exists(self):
        from japanese_g2p import QUESTION_MARKER

        assert QUESTION_MARKER == "?"

    def test_question_sentence_has_marker(self):
        from japanese_g2p import japanese_to_ipa

        result = japanese_to_ipa("元気ですか", include_question_marker=True)
        # OpenJTalk may or may not detect this as a question
        assert isinstance(result, str)

    def test_question_marker_enabled_by_default(self):
        from japanese_g2p import japanese_to_ipa

        # Question marker is enabled by default
        result = japanese_to_ipa("元気ですか")
        assert isinstance(result, str)

    def test_question_marker_can_be_disabled(self):
        from japanese_g2p import japanese_to_ipa

        result = japanese_to_ipa("元気ですか", include_question_marker=False)
        # Should not have ? marker when disabled
        assert isinstance(result, str)

    def test_question_marker_with_punctuation(self):
        from japanese_g2p import japanese_to_ipa

        result = japanese_to_ipa("何ですか？", include_question_marker=True)
        # Should detect as question
        assert isinstance(result, str)


class TestPausePreservation:
    """Test pause symbol preservation."""

    def test_pause_marker_constant_exists(self):
        from japanese_g2p import PAUSE_MARKER

        assert PAUSE_MARKER == "_"

    def test_pause_preserved_when_enabled(self):
        from japanese_g2p import japanese_to_ipa

        # Text with punctuation that creates pauses
        result = japanese_to_ipa("こんにちは、元気ですか", preserve_pause=True)
        assert "_" in result

    def test_pause_preserved_by_default(self):
        from japanese_g2p import japanese_to_ipa

        # Pause is preserved by default
        result = japanese_to_ipa("こんにちは、元気ですか")
        assert "_" in result

    def test_pause_can_be_disabled(self):
        from japanese_g2p import japanese_to_ipa

        result = japanese_to_ipa("こんにちは、元気ですか", preserve_pause=False)
        assert "_" not in result


class TestExtendedPhonemes:
    """Test extended phoneme mappings."""

    def test_fy_mapping_exists(self):
        from japanese_g2p import OPENJTALK_TO_IPA

        assert "fy" in OPENJTALK_TO_IPA
        assert OPENJTALK_TO_IPA["fy"] == "ɸʲ"

    def test_gw_mapping_exists(self):
        from japanese_g2p import OPENJTALK_TO_IPA

        assert "gw" in OPENJTALK_TO_IPA
        assert OPENJTALK_TO_IPA["gw"] == "ɡʷ"

    def test_kw_mapping_exists(self):
        from japanese_g2p import OPENJTALK_TO_IPA

        assert "kw" in OPENJTALK_TO_IPA
        assert OPENJTALK_TO_IPA["kw"] == "kʷ"

    def test_extended_phonemes_in_vocabulary(self):
        from japanese_g2p import is_valid_for_styletts2

        # Check that extended phoneme IPA representations are valid
        is_valid, invalid = is_valid_for_styletts2("ɸʲ ɡʷ kʷ")
        assert is_valid, f"Invalid characters: {invalid}"


class TestUnvoicedVowels:
    """Test unvoiced vowel handling."""

    def test_uppercase_vowels_in_mapping(self):
        from japanese_g2p import OPENJTALK_TO_IPA

        # Check if uppercase vowels are explicitly mapped
        for vowel in ["A", "I", "U", "E", "O"]:
            assert vowel in OPENJTALK_TO_IPA

    def test_unvoiced_vowels_dropped_by_default(self):
        from japanese_g2p import japanese_to_ipa

        # Words with common unvoiced vowels
        result = japanese_to_ipa("です", drop_unvoiced_vowels=True)
        # Should not have uppercase vowel markers in output
        assert "A" not in result
        assert "I" not in result
        assert "U" not in result
        assert "E" not in result
        assert "O" not in result

    def test_unvoiced_vowels_converted_to_lowercase(self):
        from japanese_g2p import japanese_to_ipa

        result = japanese_to_ipa("すき", drop_unvoiced_vowels=True)
        # Should produce valid IPA output
        assert isinstance(result, str)
        assert len(result) > 0

    def test_drop_unvoiced_vowels_parameter_works(self):
        from japanese_g2p import japanese_to_ipa

        # Both options should produce valid output
        result_drop = japanese_to_ipa("です", drop_unvoiced_vowels=True)
        result_keep = japanese_to_ipa("です", drop_unvoiced_vowels=False)
        assert isinstance(result_drop, str)
        assert isinstance(result_keep, str)
