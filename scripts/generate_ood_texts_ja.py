#!/usr/bin/env python3
"""Generate Japanese OOD (Out-of-Distribution) texts for StyleTTS2 training.

This script extracts text from JVS corpus speakers that are not used in training
to create OOD text data required by StyleTTS2.

Usage:
    python scripts/generate_ood_texts_ja.py \
        --jvs_root /path/to/jvs_ver1 \
        --exclude_speakers jvs001 jvs002 jvs003 \
        --output Data/OOD_texts_ja.txt \
        --count 500
"""

import argparse
import random
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from japanese_g2p import is_valid_for_styletts2, japanese_to_ipa


def load_jvs_transcripts(transcript_path: Path) -> list[str]:
    """Load JVS transcript file and return list of texts.

    Args:
        transcript_path: Path to transcripts_utf8.txt.

    Returns:
        List of text strings.
    """
    texts = []
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            parts = line.split(":", 1)
            if len(parts) == 2:
                text = parts[1].strip()
                if text:
                    texts.append(text)
    return texts


def collect_texts_from_jvs(
    jvs_root: Path,
    exclude_speakers: list[str],
    subsets: list[str] = None,
) -> list[str]:
    """Collect texts from JVS corpus, excluding specified speakers.

    Args:
        jvs_root: Root directory of JVS corpus.
        exclude_speakers: List of speaker IDs to exclude.
        subsets: List of subsets to include (default: all).

    Returns:
        List of unique text strings.
    """
    if subsets is None:
        subsets = ["parallel100", "nonpara30"]

    texts = set()  # Use set to avoid duplicates

    # Find all speaker directories
    for speaker_dir in sorted(jvs_root.iterdir()):
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name
        if speaker_id in exclude_speakers:
            continue

        if not speaker_id.startswith("jvs"):
            continue

        for subset in subsets:
            transcript_file = speaker_dir / subset / "transcripts_utf8.txt"
            if transcript_file.exists():
                subset_texts = load_jvs_transcripts(transcript_file)
                texts.update(subset_texts)

    return list(texts)


def convert_to_ipa(texts: list[str], validate: bool = True) -> list[str]:
    """Convert texts to IPA format for OOD data.

    Args:
        texts: List of Japanese text strings.
        validate: If True, validate phonemes against StyleTTS2 vocabulary.

    Returns:
        List of IPA phoneme strings.
    """
    ipa_texts = []
    skipped = 0

    for text in texts:
        phonemes = japanese_to_ipa(text)

        if not phonemes:
            skipped += 1
            continue

        if validate:
            is_valid, _ = is_valid_for_styletts2(phonemes)
            if not is_valid:
                skipped += 1
                continue

        ipa_texts.append(phonemes)

    if skipped > 0:
        print(f"Skipped {skipped} texts due to conversion issues")

    return ipa_texts


def main():
    parser = argparse.ArgumentParser(description="Generate Japanese OOD texts for StyleTTS2")
    parser.add_argument(
        "--jvs_root",
        type=Path,
        required=True,
        help="Root directory of JVS corpus",
    )
    parser.add_argument(
        "--exclude_speakers",
        nargs="+",
        default=["jvs001", "jvs002", "jvs003"],
        help="Speaker IDs to exclude (default: jvs001 jvs002 jvs003)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/OOD_texts_ja.txt"),
        help="Output file path (default: Data/OOD_texts_ja.txt)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of OOD texts to generate (default: 500)",
    )
    parser.add_argument(
        "--no_validate",
        action="store_true",
        help="Skip phoneme validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )

    args = parser.parse_args()

    print(f"Collecting texts from JVS corpus: {args.jvs_root}")
    print(f"Excluding speakers: {args.exclude_speakers}")

    # Collect texts
    texts = collect_texts_from_jvs(args.jvs_root, args.exclude_speakers)
    print(f"Collected {len(texts)} unique texts")

    # Convert to IPA
    print("Converting to IPA phonemes...")
    ipa_texts = convert_to_ipa(texts, validate=not args.no_validate)
    print(f"Converted {len(ipa_texts)} texts")

    # Shuffle and select
    random.seed(args.seed)
    random.shuffle(ipa_texts)

    selected = ipa_texts[: args.count]
    print(f"Selected {len(selected)} texts")

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(selected))

    print(f"Wrote {len(selected)} texts to {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
