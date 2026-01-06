#!/usr/bin/env python3
"""Prepare Japanese dataset for StyleTTS2 training.

This script converts Japanese audio-text pairs to the format required by StyleTTS2.
It uses pyopenjtalk-plus for G2P conversion.

Usage:
    python scripts/prepare_japanese_data.py --input_dir /path/to/dataset --output_dir Data

Expected input formats:
    1. CSV file with columns: filename, text, speaker_id (optional)
    2. Directory with wav files and corresponding .txt files
    3. Directory with wav files and a single transcript.txt file

Output format:
    filename.wav|IPA_phonemes|speaker_id
"""

import argparse
import csv
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from japanese_g2p import japanese_to_ipa


def load_csv_dataset(csv_path: Path) -> list[dict]:
    """Load dataset from CSV file.

    Expected columns: filename, text, speaker_id (optional)

    Args:
        csv_path: Path to CSV file.

    Returns:
        List of dictionaries with keys: filename, text, speaker_id.
    """
    entries = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                "filename": row.get("filename", row.get("file", row.get("audio", ""))),
                "text": row.get("text", row.get("transcript", row.get("transcription", ""))),
                "speaker_id": row.get("speaker_id", row.get("speaker", "0")),
            }
            if entry["filename"] and entry["text"]:
                entries.append(entry)
    return entries


def load_txt_pairs(audio_dir: Path) -> list[dict]:
    """Load dataset from directory with audio files and corresponding .txt files.

    Args:
        audio_dir: Directory containing .wav and .txt file pairs.

    Returns:
        List of dictionaries with keys: filename, text, speaker_id.
    """
    entries = []
    for wav_file in sorted(audio_dir.glob("*.wav")):
        txt_file = wav_file.with_suffix(".txt")
        if txt_file.exists():
            text = txt_file.read_text(encoding="utf-8").strip()
            entries.append({"filename": wav_file.name, "text": text, "speaker_id": "0"})
    return entries


def load_transcript_file(audio_dir: Path, transcript_path: Path) -> list[dict]:
    """Load dataset from a single transcript file.

    Expected format per line: filename|text or filename|text|speaker_id

    Args:
        audio_dir: Directory containing audio files.
        transcript_path: Path to transcript file.

    Returns:
        List of dictionaries with keys: filename, text, speaker_id.
    """
    entries = []
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) >= 2:
                filename = parts[0].strip()
                text = parts[1].strip()
                speaker_id = parts[2].strip() if len(parts) > 2 else "0"

                # Check if audio file exists
                wav_path = audio_dir / filename
                if not wav_path.suffix:
                    wav_path = wav_path.with_suffix(".wav")

                if wav_path.exists():
                    entries.append({"filename": wav_path.name, "text": text, "speaker_id": speaker_id})
    return entries


def convert_to_styletts2_format(entries: list[dict], validate: bool = True) -> list[str]:
    """Convert entries to StyleTTS2 training format.

    Args:
        entries: List of dictionaries with filename, text, speaker_id.
        validate: If True, validate phonemes against StyleTTS2 vocabulary.

    Returns:
        List of formatted strings for train_list.txt.
    """
    from japanese_g2p import is_valid_for_styletts2

    formatted = []
    skipped = 0

    for entry in entries:
        text = entry["text"]

        # Convert to IPA phonemes
        phonemes = japanese_to_ipa(text)

        if not phonemes:
            print(f"Warning: Empty phonemes for '{text}', skipping")
            skipped += 1
            continue

        if validate:
            is_valid, invalid_chars = is_valid_for_styletts2(phonemes)
            if not is_valid:
                print(f"Warning: Invalid characters {invalid_chars} in '{text}', skipping")
                skipped += 1
                continue

        line = f"{entry['filename']}|{phonemes}|{entry['speaker_id']}"
        formatted.append(line)

    if skipped > 0:
        print(f"Skipped {skipped} entries due to conversion issues")

    return formatted


def split_dataset(entries: list[str], val_ratio: float = 0.1) -> tuple[list[str], list[str]]:
    """Split dataset into training and validation sets.

    Args:
        entries: List of formatted strings.
        val_ratio: Ratio of validation set (default 0.1 = 10%).

    Returns:
        Tuple of (train_entries, val_entries).
    """
    import random

    random.seed(42)
    shuffled = entries.copy()
    random.shuffle(shuffled)

    val_size = max(1, int(len(shuffled) * val_ratio))
    val_entries = shuffled[:val_size]
    train_entries = shuffled[val_size:]

    return train_entries, val_entries


def main():
    parser = argparse.ArgumentParser(description="Prepare Japanese dataset for StyleTTS2")
    parser.add_argument("--input_dir", type=Path, required=True, help="Input directory containing audio files")
    parser.add_argument("--output_dir", type=Path, default=Path("Data"), help="Output directory for train/val lists")
    parser.add_argument("--csv", type=Path, help="CSV file with filename, text, speaker_id columns")
    parser.add_argument("--transcript", type=Path, help="Transcript file (filename|text|speaker_id format)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio (default: 0.1)")
    parser.add_argument("--no_validate", action="store_true", help="Skip phoneme validation")
    parser.add_argument("--output_prefix", type=str, default="", help="Prefix for output files (e.g., 'train_list_ja')")

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.input_dir}")

    if args.csv:
        entries = load_csv_dataset(args.csv)
        print(f"Loaded {len(entries)} entries from CSV")
    elif args.transcript:
        entries = load_transcript_file(args.input_dir, args.transcript)
        print(f"Loaded {len(entries)} entries from transcript file")
    else:
        entries = load_txt_pairs(args.input_dir)
        print(f"Loaded {len(entries)} entries from txt pairs")

    if not entries:
        print("Error: No entries found")
        sys.exit(1)

    # Convert to StyleTTS2 format
    print("Converting to IPA phonemes...")
    formatted = convert_to_styletts2_format(entries, validate=not args.no_validate)
    print(f"Converted {len(formatted)} entries")

    # Split dataset
    train_entries, val_entries = split_dataset(formatted, args.val_ratio)
    print(f"Split: {len(train_entries)} train, {len(val_entries)} val")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filenames
    prefix = args.output_prefix or "train_list_ja"
    train_file = args.output_dir / f"{prefix}.txt"
    # Derive validation file name from prefix
    # Expected naming convention: train_list_* -> val_list_* (e.g., train_list_ja -> val_list_ja)
    # For custom prefixes without "train_list", prepends "val_" (e.g., my_data -> val_my_data)
    val_prefix = prefix.replace("train_list", "val_list") if "train_list" in prefix else f"val_{prefix}"
    val_file = args.output_dir / f"{val_prefix}.txt"

    # Write output files
    with open(train_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_entries))
    print(f"Wrote {len(train_entries)} entries to {train_file}")

    with open(val_file, "w", encoding="utf-8") as f:
        f.write("\n".join(val_entries))
    print(f"Wrote {len(val_entries)} entries to {val_file}")

    print("Done!")


if __name__ == "__main__":
    main()
