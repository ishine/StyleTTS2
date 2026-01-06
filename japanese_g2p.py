"""Japanese G2P (Grapheme-to-Phoneme) module using pyopenjtalk-plus.

This module converts Japanese text to IPA phonemes compatible with StyleTTS2.
"""

import re

import pyopenjtalk

from text_utils import dicts

# OpenJTalk phoneme to IPA mapping
# Based on Japanese phonology and IPA standards
OPENJTALK_TO_IPA = {
    # Vowels
    "a": "a",
    "i": "i",
    "u": "ɯ",
    "e": "e",
    "o": "o",
    # Consonants
    "k": "k",
    "g": "ɡ",
    "s": "s",
    "z": "z",
    "t": "t",
    "d": "d",
    "n": "n",
    "h": "h",
    "f": "ɸ",
    "b": "b",
    "p": "p",
    "m": "m",
    "y": "j",
    "r": "ɾ",
    "w": "w",
    "v": "v",
    # Palatalized consonants
    "ky": "kʲ",
    "gy": "ɡʲ",
    "sy": "ɕ",
    "sh": "ɕ",
    "zy": "ʑ",
    "jy": "ʑ",
    "j": "dʑ",
    "ty": "tɕ",
    "ch": "tɕ",
    "dy": "dʑ",
    "ny": "ɲ",
    "hy": "ç",
    "by": "bʲ",
    "py": "pʲ",
    "my": "mʲ",
    "ry": "ɾʲ",
    # Special phonemes
    "N": "ɴ",  # Syllabic nasal (撥音)
    "cl": "ʔ",  # Geminate consonant (促音) - represented as glottal stop
    "q": "ʔ",  # Alternative geminate notation
    "ts": "ts",
    "dz": "dz",
    # Extended consonants (pyopenjtalk-plus)
    "fy": "ɸʲ",  # フュ (fu + palatalization)
    "gw": "ɡʷ",  # グァ, グォ, etc. (g + labialization)
    "kw": "kʷ",  # クァ, クォ, etc. (k + labialization)
    # Unvoiced vowels (uppercase from OpenJTalk)
    "A": "a",  # Unvoiced /a/ -> voiced
    "I": "i",  # Unvoiced /i/ -> voiced
    "U": "ɯ",  # Unvoiced /u/ -> voiced
    "E": "e",  # Unvoiced /e/ -> voiced
    "O": "o",  # Unvoiced /o/ -> voiced
    # Silence/pause
    "pau": " ",
    "sil": "",
    "br": " ",
}

# Long vowel marker
LONG_VOWEL_MARKER = "ː"

# Pitch accent markers (already in StyleTTS2 vocabulary)
ACCENT_RISE = "↑"  # Pitch rise marker
ACCENT_FALL = "↓"  # Pitch fall marker

# Prosodic markers (Style-Bert-VITS2 inspired)
ACCENT_PHRASE_BOUNDARY = "#"  # Accent phrase boundary marker
PAUSE_MARKER = "_"  # Pause/silence marker
QUESTION_MARKER = "?"  # Question ending marker


def _extract_phoneme_from_label(label: str) -> str | None:
    """Extract phoneme from OpenJTalk fullcontext label.

    Args:
        label: A single fullcontext label string from OpenJTalk.

    Returns:
        The extracted phoneme string, or None if not found.
    """
    # OpenJTalk fullcontext format: p1^p2-p3+p4=p5/...
    # p3 is the current phoneme
    match = re.search(r"\^([a-zA-Z]+)-([a-zA-Z]+)\+", label)
    if match:
        return match.group(2)

    # Alternative pattern for boundary phonemes
    match = re.search(r"-([a-zA-Z]+)\+", label)
    if match:
        return match.group(1)

    return None


def _extract_accent_info(label: str) -> int | None:
    """Extract A1 (accent position) from OpenJTalk fullcontext label.

    The A field format is: /A:a1+a2+a3
    - a1: Position relative to accent nucleus (negative=before, 0=at, positive=after)
    - a2: Mora position within accent phrase
    - a3: Number of moras in accent phrase

    Args:
        label: A single fullcontext label string from OpenJTalk.

    Returns:
        The A1 value (accent position), or None if not found.
    """
    match = re.search(r"/A:([^+]+)\+", label)
    if match:
        a1_str = match.group(1)
        if a1_str != "xx":
            try:
                return int(a1_str)
            except ValueError:
                pass
    return None


def _extract_a2_info(label: str) -> int | None:
    """Extract A2 (mora position within accent phrase) from OpenJTalk fullcontext label.

    The A field format is: /A:a1+a2+a3
    - a2: Mora position within the current accent phrase (1-indexed)

    Args:
        label: A single fullcontext label string from OpenJTalk.

    Returns:
        The A2 value (mora position), or None if not found.
    """
    match = re.search(r"/A:[^+]+\+(\d+)\+", label)
    if match:
        return int(match.group(1))
    return None


def _extract_a3_info(label: str) -> int | None:
    """Extract A3 (number of morae in accent phrase) from OpenJTalk fullcontext label.

    The A field format is: /A:a1+a2+a3
    - a3: Number of morae in the current accent phrase

    Args:
        label: A single fullcontext label string from OpenJTalk.

    Returns:
        The A3 value (total morae count), or None if not found.
    """
    match = re.search(r"/A:[^+]+\+\d+\+(\d+)/", label)
    if match:
        return int(match.group(1))
    return None


def _extract_e3_info(label: str) -> int | None:
    """Extract E3 (question/statement indicator) from OpenJTalk fullcontext label.

    The E field contains utterance-level information.
    E3 indicates: 0 = statement, 1 = question

    Args:
        label: A single fullcontext label string from OpenJTalk.

    Returns:
        The E3 value (0 for statement, 1 for question), or None if not found.
    """
    match = re.search(r"!(\d+)_", label)
    if match:
        return int(match.group(1))
    return None


def _convert_phoneme_to_ipa(phoneme: str) -> str:
    """Convert a single OpenJTalk phoneme to IPA.

    Args:
        phoneme: OpenJTalk phoneme string.

    Returns:
        IPA representation of the phoneme.
    """
    # Check direct mapping first
    if phoneme in OPENJTALK_TO_IPA:
        return OPENJTALK_TO_IPA[phoneme]

    # Handle unknown phonemes by returning as-is (lowercase)
    return phoneme.lower()


def japanese_to_ipa(
    text: str,
    add_spaces: bool = True,
    include_accent: bool = True,
    include_phrase_boundary: bool = True,
    include_question_marker: bool = True,
    preserve_pause: bool = True,
    drop_unvoiced_vowels: bool = True,
) -> str:
    """Convert Japanese text to IPA phoneme string.

    Args:
        text: Japanese text to convert.
        add_spaces: If True, add spaces between phonemes for StyleTTS2 compatibility.
        include_accent: If True, include pitch accent markers (↑↓).
        include_phrase_boundary: If True, include accent phrase boundary markers (#).
        include_question_marker: If True, use ? for question endings.
        preserve_pause: If True, preserve pause markers (_) instead of converting to space.
        drop_unvoiced_vowels: If True, convert unvoiced vowels to their voiced equivalents.
            If False, preserve unvoiced vowel markers (uppercase).

    Returns:
        IPA phoneme string.

    Examples:
        >>> japanese_to_ipa("こんにちは")
        'k o ɴ n i tɕ i w a'
        >>> japanese_to_ipa("雨", include_accent=True)
        'a ↓ m e'
        >>> japanese_to_ipa("飴", include_accent=True)
        'a ↑ m e'
        >>> japanese_to_ipa("元気ですか？", include_question_marker=True)
        'ɡ e ɴ k i d e s ɯ k a ?'
    """
    if not text or not text.strip():
        return ""

    # Get fullcontext labels from OpenJTalk
    labels = pyopenjtalk.extract_fullcontext(text)

    phonemes = []
    prev_phoneme = None
    prev_a1 = None
    last_e3 = None
    add_phrase_boundary = False

    for i, label in enumerate(labels):
        phoneme = _extract_phoneme_from_label(label)
        if phoneme is None:
            continue

        # Handle unvoiced vowels
        if not drop_unvoiced_vowels and phoneme in ("A", "I", "U", "E", "O"):
            # Keep uppercase for unvoiced vowels
            pass
        elif phoneme in ("A", "I", "U", "E", "O"):
            # Convert to lowercase (will be mapped to voiced vowels)
            phoneme = phoneme.lower()

        # Handle pause markers
        if phoneme == "pau":
            if preserve_pause:
                phonemes.append(PAUSE_MARKER)
            prev_phoneme = phoneme
            continue

        # Skip silence markers at boundaries
        if phoneme == "sil":
            prev_phoneme = phoneme
            continue

        # Extract prosodic info
        a1 = _extract_accent_info(label) if include_accent else None
        a3 = _extract_a3_info(label) if include_phrase_boundary else None
        e3 = _extract_e3_info(label)
        if e3 is not None:
            last_e3 = e3

        # Get next label's A2 for phrase boundary detection
        a2_next = None
        if include_phrase_boundary and i + 1 < len(labels):
            a2_next = _extract_a2_info(labels[i + 1])

        ipa = _convert_phoneme_to_ipa(phoneme)

        # Handle long vowels (consecutive same vowels) - check before accent markers
        prev_ipa = _convert_phoneme_to_ipa(prev_phoneme) if prev_phoneme else None
        is_long_vowel = ipa and prev_ipa and ipa == prev_ipa and ipa in ("a", "i", "ɯ", "e", "o")

        # Insert accent markers based on A1 transitions
        if include_accent and a1 is not None and prev_a1 is not None:
            # ↑: pitch rise (A1 transitions from negative to 0)
            if prev_a1 < 0 and a1 == 0:
                phonemes.append(ACCENT_RISE)
            # ↓: pitch fall (A1 transitions from 0 to positive)
            elif prev_a1 == 0 and a1 > 0:
                phonemes.append(ACCENT_FALL)

        # Insert pending phrase boundary marker (from previous iteration)
        if add_phrase_boundary:
            phonemes.append(ACCENT_PHRASE_BOUNDARY)
            add_phrase_boundary = False

        # Handle long vowels
        if is_long_vowel:
            # Find the last vowel in phonemes and extend it
            for j in range(len(phonemes) - 1, -1, -1):
                if phonemes[j] == ipa:
                    phonemes[j] = ipa + LONG_VOWEL_MARKER
                    break
            prev_phoneme = phoneme
            prev_a1 = a1
            continue

        if ipa and ipa.strip():
            phonemes.append(ipa)

        # Check for accent phrase boundary (insert after current phoneme in next iteration)
        # Condition: a3 == 1 means this is the last mora of accent phrase
        # and next mora starts new phrase (a2_next == 1)
        # and current phoneme is a vowel, N, or cl
        if (
            include_phrase_boundary
            and a3 is not None
            and a2_next is not None
            and a3 == 1
            and a2_next == 1
            and phoneme.lower() in ("a", "i", "u", "e", "o", "n", "cl")
        ):
            add_phrase_boundary = True

        prev_phoneme = phoneme
        prev_a1 = a1

    # Add question marker at the end if detected
    if include_question_marker and last_e3 == 1:
        phonemes.append(QUESTION_MARKER)

    # Join phonemes
    separator = " " if add_spaces else ""
    result = separator.join(phonemes)

    # Clean up multiple spaces
    result = re.sub(r"\s+", " ", result).strip()

    return result


def japanese_to_phoneme_list(text: str) -> list[str]:
    """Convert Japanese text to a list of IPA phonemes.

    Args:
        text: Japanese text to convert.

    Returns:
        List of IPA phoneme strings.

    Examples:
        >>> japanese_to_phoneme_list("こんにちは")
        ['k', 'o', 'ɴ', 'n', 'i', 'tɕ', 'i', 'w', 'a']
    """
    ipa_string = japanese_to_ipa(text, add_spaces=True)
    if not ipa_string:
        return []
    return ipa_string.split()


def is_valid_for_styletts2(phonemes: str) -> tuple[bool, list[str]]:
    """Check if phonemes are valid for StyleTTS2 vocabulary.

    Args:
        phonemes: Space-separated IPA phoneme string.

    Returns:
        Tuple of (is_valid, list of invalid phonemes).
    """
    invalid = []
    for char in phonemes:
        if char not in dicts and char != " ":
            invalid.append(char)

    return len(invalid) == 0, list(set(invalid))
