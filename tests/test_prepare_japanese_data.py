"""Tests for Japanese data preparation script."""

import csv
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prepare_japanese_data import (
    convert_to_styletts2_format,
    load_csv_dataset,
    load_transcript_file,
    load_txt_pairs,
    split_dataset,
)


class TestLoadCsvDataset:
    """Test CSV loading functionality."""

    def test_load_basic_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "text", "speaker_id"])
            writer.writerow(["audio1.wav", "こんにちは", "0"])
            writer.writerow(["audio2.wav", "ありがとう", "1"])

        entries = load_csv_dataset(csv_file)
        assert len(entries) == 2
        assert entries[0]["filename"] == "audio1.wav"
        assert entries[0]["text"] == "こんにちは"
        assert entries[0]["speaker_id"] == "0"

    def test_load_csv_with_alternative_columns(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "transcript", "speaker"])
            writer.writerow(["audio1.wav", "テスト", "speaker1"])

        entries = load_csv_dataset(csv_file)
        assert len(entries) == 1
        assert entries[0]["filename"] == "audio1.wav"
        assert entries[0]["text"] == "テスト"

    def test_load_csv_skips_empty_entries(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "text", "speaker_id"])
            writer.writerow(["audio1.wav", "こんにちは", "0"])
            writer.writerow(["", "missing filename", "0"])
            writer.writerow(["audio3.wav", "", "0"])

        entries = load_csv_dataset(csv_file)
        assert len(entries) == 1


class TestLoadTxtPairs:
    """Test txt pair loading functionality."""

    def test_load_txt_pairs(self, tmp_path):
        # Create wav and txt files
        (tmp_path / "audio1.wav").touch()
        (tmp_path / "audio1.txt").write_text("こんにちは", encoding="utf-8")
        (tmp_path / "audio2.wav").touch()
        (tmp_path / "audio2.txt").write_text("ありがとう", encoding="utf-8")

        entries = load_txt_pairs(tmp_path)
        assert len(entries) == 2

    def test_load_txt_pairs_ignores_unpaired(self, tmp_path):
        # Create only wav file without txt
        (tmp_path / "audio1.wav").touch()

        entries = load_txt_pairs(tmp_path)
        assert len(entries) == 0

    def test_load_txt_pairs_sorted(self, tmp_path):
        (tmp_path / "b_audio.wav").touch()
        (tmp_path / "b_audio.txt").write_text("B", encoding="utf-8")
        (tmp_path / "a_audio.wav").touch()
        (tmp_path / "a_audio.txt").write_text("A", encoding="utf-8")

        entries = load_txt_pairs(tmp_path)
        assert entries[0]["filename"] == "a_audio.wav"
        assert entries[1]["filename"] == "b_audio.wav"


class TestLoadTranscriptFile:
    """Test transcript file loading."""

    def test_load_basic_transcript(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "audio1.wav").touch()
        (audio_dir / "audio2.wav").touch()

        transcript = tmp_path / "transcript.txt"
        transcript.write_text("audio1.wav|こんにちは|0\naudio2.wav|ありがとう|1", encoding="utf-8")

        entries = load_transcript_file(audio_dir, transcript)
        assert len(entries) == 2
        assert entries[0]["text"] == "こんにちは"
        assert entries[1]["speaker_id"] == "1"

    def test_load_transcript_with_default_speaker(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "audio1.wav").touch()

        transcript = tmp_path / "transcript.txt"
        transcript.write_text("audio1.wav|テスト", encoding="utf-8")

        entries = load_transcript_file(audio_dir, transcript)
        assert len(entries) == 1
        assert entries[0]["speaker_id"] == "0"

    def test_load_transcript_skips_missing_audio(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        # Don't create audio file

        transcript = tmp_path / "transcript.txt"
        transcript.write_text("missing.wav|テスト|0", encoding="utf-8")

        entries = load_transcript_file(audio_dir, transcript)
        assert len(entries) == 0


class TestConvertToStyletts2Format:
    """Test format conversion."""

    def test_convert_basic_entries(self):
        entries = [{"filename": "audio1.wav", "text": "あ", "speaker_id": "0"}]

        formatted = convert_to_styletts2_format(entries, validate=False)
        assert len(formatted) == 1
        assert formatted[0].startswith("audio1.wav|")
        assert formatted[0].endswith("|0")

    def test_convert_preserves_speaker_id(self):
        entries = [{"filename": "audio1.wav", "text": "あ", "speaker_id": "5"}]

        formatted = convert_to_styletts2_format(entries, validate=False)
        assert formatted[0].endswith("|5")

    def test_convert_output_format(self):
        entries = [{"filename": "test.wav", "text": "テスト", "speaker_id": "0"}]

        formatted = convert_to_styletts2_format(entries, validate=False)
        parts = formatted[0].split("|")
        assert len(parts) == 3
        assert parts[0] == "test.wav"
        assert parts[2] == "0"


class TestSplitDataset:
    """Test dataset splitting."""

    def test_split_basic(self):
        entries = [f"entry{i}" for i in range(100)]

        train, val = split_dataset(entries, val_ratio=0.1)
        assert len(train) == 90
        assert len(val) == 10

    def test_split_deterministic(self):
        entries = [f"entry{i}" for i in range(100)]

        train1, val1 = split_dataset(entries, val_ratio=0.1)
        train2, val2 = split_dataset(entries, val_ratio=0.1)

        assert train1 == train2
        assert val1 == val2

    def test_split_minimum_val_size(self):
        entries = ["entry1", "entry2"]

        train, val = split_dataset(entries, val_ratio=0.1)
        assert len(val) >= 1

    def test_split_preserves_all_entries(self):
        entries = [f"entry{i}" for i in range(50)]

        train, val = split_dataset(entries, val_ratio=0.2)
        assert len(train) + len(val) == 50
        assert set(train + val) == set(entries)
