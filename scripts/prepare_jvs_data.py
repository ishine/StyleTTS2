#!/usr/bin/env python3
"""Prepare JVS corpus for StyleTTS2 training.

This script converts JVS corpus data to the format required by StyleTTS2.
It uses pyopenjtalk-plus for G2P conversion.

Usage:
    python scripts/prepare_jvs_data.py \
        --jvs_root /path/to/jvs_ver1 \
        --output_dir Data \
        --speakers jvs001 jvs002 jvs003 \
        --subsets parallel100

JVS corpus structure:
    jvs_ver1/
    ├── jvs001/
    │   ├── parallel100/
    │   │   ├── wav24kHz16bit/
    │   │   │   └── VOICEACTRESS100_001.wav ~ 100.wav
    │   │   └── transcripts_utf8.txt
    │   ├── nonpara30/
    │   ├── falset10/
    │   └── whisper10/
    ├── jvs002/
    ...

Output format:
    filename.wav|IPA_phonemes|speaker_id
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from japanese_g2p import is_valid_for_styletts2, japanese_to_ipa


def load_jvs_transcripts(transcript_path: Path) -> dict[str, str]:
    """Load JVS transcript file.

    JVS transcript format: VOICEACTRESS100_001:テキスト

    Args:
        transcript_path: Path to transcripts_utf8.txt.

    Returns:
        Dictionary mapping filename (without extension) to text.
    """
    transcripts = {}
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            parts = line.split(":", 1)
            if len(parts) == 2:
                filename = parts[0].strip()
                text = parts[1].strip()
                transcripts[filename] = text
    return transcripts


def process_speaker(
    jvs_root: Path,
    speaker_id: str,
    subsets: list[str],
    speaker_index: int,
) -> list[dict]:
    """Process a single JVS speaker.

    Args:
        jvs_root: Root directory of JVS corpus.
        speaker_id: Speaker ID (e.g., 'jvs001').
        subsets: List of subsets to include (e.g., ['parallel100']).
        speaker_index: Numeric index for the speaker.

    Returns:
        List of dictionaries with keys: wav_path, text, speaker_id, speaker_name, subset.
    """
    entries = []
    speaker_dir = jvs_root / speaker_id

    if not speaker_dir.exists():
        print(f"Warning: Speaker directory not found: {speaker_dir}")
        return entries

    for subset in subsets:
        subset_dir = speaker_dir / subset
        wav_dir = subset_dir / "wav24kHz16bit"
        transcript_file = subset_dir / "transcripts_utf8.txt"

        if not wav_dir.exists():
            print(f"Warning: WAV directory not found: {wav_dir}")
            continue

        if not transcript_file.exists():
            print(f"Warning: Transcript file not found: {transcript_file}")
            continue

        transcripts = load_jvs_transcripts(transcript_file)

        for wav_file in sorted(wav_dir.glob("*.wav")):
            filename_key = wav_file.stem
            if filename_key in transcripts:
                text = transcripts[filename_key]
                entries.append(
                    {
                        "wav_path": wav_file,
                        "text": text,
                        "speaker_id": speaker_index,
                        "speaker_name": speaker_id,
                        "subset": subset,
                    }
                )
            else:
                print(f"Warning: No transcript for {wav_file.name}")

    return entries


def convert_to_styletts2_format(
    entries: list[dict],
    validate: bool = True,
) -> list[tuple[str, Path]]:
    """Convert entries to StyleTTS2 training format.

    Args:
        entries: List of dictionaries with wav_path, text, speaker_id.
        validate: If True, validate phonemes against StyleTTS2 vocabulary.

    Returns:
        List of tuples (formatted_line, wav_path).
    """
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

        # Create unique filename: speaker_subset_originalname.wav
        original_name = entry["wav_path"].stem
        unique_name = f"{entry['speaker_name']}_{entry['subset']}_{original_name}.wav"

        line = f"{unique_name}|{phonemes}|{entry['speaker_id']}"
        formatted.append((line, entry["wav_path"]))

    if skipped > 0:
        print(f"Skipped {skipped} entries due to conversion issues")

    return formatted


def split_dataset(
    entries: list[tuple[str, Path]],
    val_ratio: float = 0.1,
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    """Split dataset into training and validation sets.

    Args:
        entries: List of tuples (formatted_line, wav_path).
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


def create_wav_links(
    entries: list[tuple[str, Path]],
    output_wav_dir: Path,
    use_symlinks: bool = True,
) -> None:
    """Create symlinks or copy wav files to output directory.

    Args:
        entries: List of tuples (formatted_line, wav_path).
        output_wav_dir: Directory to create links/copies in.
        use_symlinks: If True, create symlinks. Otherwise, copy files.
    """
    output_wav_dir.mkdir(parents=True, exist_ok=True)

    for line, wav_path in entries:
        # Extract filename from formatted line
        filename = line.split("|")[0]
        dest_path = output_wav_dir / filename

        if dest_path.exists():
            continue

        if use_symlinks:
            try:
                # Use absolute path for symlink target
                os.symlink(wav_path.resolve(), dest_path)
            except OSError as e:
                # Any symlink failure (e.g., Windows without admin rights) falls back to copying
                print(f"Warning: Symlink failed ({e}), falling back to copy")
                shutil.copy2(wav_path, dest_path)
        else:
            shutil.copy2(wav_path, dest_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare JVS corpus for StyleTTS2")
    parser.add_argument(
        "--jvs_root",
        type=Path,
        required=True,
        help="Root directory of JVS corpus (e.g., /path/to/jvs_ver1)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("Data"),
        help="Output directory for train/val lists (default: Data)",
    )
    parser.add_argument(
        "--speakers",
        nargs="+",
        default=["jvs001"],
        help="Speaker IDs to include (default: jvs001)",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["parallel100"],
        help="Subsets to include (default: parallel100)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--no_validate",
        action="store_true",
        help="Skip phoneme validation",
    )
    parser.add_argument(
        "--use_copy",
        action="store_true",
        help="Copy files instead of creating symlinks",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="jvs",
        help="Prefix for output files (default: jvs -> train_list_jvs.txt)",
    )

    args = parser.parse_args()

    # Process all speakers
    print(f"Processing JVS corpus from {args.jvs_root}")
    print(f"Speakers: {args.speakers}")
    print(f"Subsets: {args.subsets}")

    all_entries = []
    for idx, speaker in enumerate(args.speakers):
        entries = process_speaker(args.jvs_root, speaker, args.subsets, idx)
        all_entries.extend(entries)
        print(f"Loaded {len(entries)} entries from {speaker}")

    if not all_entries:
        print("Error: No entries found")
        sys.exit(1)

    print(f"Total entries: {len(all_entries)}")

    # Convert to StyleTTS2 format
    print("Converting to IPA phonemes...")
    formatted = convert_to_styletts2_format(all_entries, validate=not args.no_validate)
    print(f"Converted {len(formatted)} entries")

    # Split dataset
    train_entries, val_entries = split_dataset(formatted, args.val_ratio)
    print(f"Split: {len(train_entries)} train, {len(val_entries)} val")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create wav links/copies
    wav_dir = args.output_dir / f"wavs_{args.output_prefix}"
    print(f"Creating wav links in {wav_dir}...")
    create_wav_links(train_entries + val_entries, wav_dir, use_symlinks=not args.use_copy)

    # Write output files
    train_file = args.output_dir / f"train_list_{args.output_prefix}.txt"
    val_file = args.output_dir / f"val_list_{args.output_prefix}.txt"

    with open(train_file, "w", encoding="utf-8") as f:
        f.write("\n".join([line for line, _ in train_entries]))
    print(f"Wrote {len(train_entries)} entries to {train_file}")

    with open(val_file, "w", encoding="utf-8") as f:
        f.write("\n".join([line for line, _ in val_entries]))
    print(f"Wrote {len(val_entries)} entries to {val_file}")

    print("Done!")


if __name__ == "__main__":
    main()
