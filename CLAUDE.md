# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

StyleTTS2は、スタイル拡散とSLM（Speech Language Model）敵対的学習を組み合わせた最先端のText-to-Speech（TTS）モデルです。人間レベルの音声合成を実現します。

## 主要コマンド

### 環境セットアップ
```bash
uv sync
# 推論用（日本語対応含む）
uv sync --group inference
# 開発用
uv sync --group dev
```

### トレーニング
```bash
# 第1段階（音響モデルの事前学習）
accelerate launch train_first.py --config_path ./Configs/config.yml

# 第2段階（拡散モデル + SLM敵対的学習）
python train_second.py --config_path ./Configs/config.yml

# ファインチューニング（accelerate版）
accelerate launch --mixed_precision=fp16 --num_processes=1 train_finetune_accelerate.py --config_path ./Configs/config_ft.yml
```

### テスト
```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## アーキテクチャ

### 2段階トレーニングパイプライン
- **第1段階**: TextEncoder、ProsodyPredictor、Decoderの学習。テキストアライメント（TMA）はエポック50から開始
- **第2段階**: スタイル拡散（エポック20から）とSLM敵対的学習（エポック50から）を追加

### コアモジュール構成
```
models.py          - 全モデルアーキテクチャ（TextEncoder, ProsodyPredictor, StyleEncoder, Decoder, Diffusion）
losses.py          - 損失関数（スペクトル収束、STFT、特徴マッチング）
meldataset.py      - データローディングと前処理
optimizers.py      - MultiOptimizer（OneCycleLRスケジューラ付き）
```

### Modulesディレクトリ
- `diffusion/` - スタイル拡散モデル（Transformer、サンプラー）
- `hifigan.py`, `istftnet.py` - ボコーダー（HiFiGAN、iSTFTNet）
- `discriminators.py` - マルチスケール識別器
- `slmadv.py` - SLM敵対的学習（WavLMベース）

### Utilsディレクトリ（事前学習済みモジュール）
- `ASR/` - テキストアライナー（ASRCNN）
- `JDC/` - ピッチ抽出器（JDCNet）
- `PLBERT/` - 言語モデルエンコーダー

## 設定ファイル

- `Configs/config.yml` - 単一話者（LJSpeech）
- `Configs/config_libritts.yml` - 複数話者（LibriTTS）
- `Configs/config_ft.yml` - ファインチューニング用
- `Configs/config_japanese.yml` - 日本語（多言語PL-BERT使用）

### 重要なハイパーパラメータ
- `batch_size`: 安定性のため16以上を推奨
- `max_len`: フレーム数上限（OOM時は削減）
- 学習率: bert=0.00001、general=0.0001、acoustic=0.00001

## データ形式

学習データリスト（`Data/train_list.txt`）のフォーマット:
```
filename.wav|transcription|speaker
```

サンプリングレート: 24 kHz

## 推論

デモノートブック:
- `Demo/Inference_LJSpeech.ipynb` - 単一話者
- `Demo/Inference_LibriTTS.ipynb` - 複数話者（ゼロショット話者適応）

## 日本語対応

### 日本語G2P (`japanese_g2p.py`)
pyopenjtalk-plusを使用した日本語テキスト→IPA音素変換モジュール。

```python
from japanese_g2p import japanese_to_ipa

# 基本変換
japanese_to_ipa("こんにちは")
# → 'k o ɴ n i tɕ i w a'

# アクセント句境界とピッチアクセント付き
japanese_to_ipa("今日は天気がいいです")
# → 'kʲ oː ↓ w a # t e ↓ ɴ k i ɡ a # iː ↓ d e s ɯ'
```

#### 韻律機能
| 機能 | 記号 | パラメータ |
|------|------|-----------|
| ピッチアクセント | ↑↓ | `include_accent` |
| アクセント句境界 | # | `include_phrase_boundary` |
| 疑問形判定 | ? | `include_question_marker` |
| ポーズ保持 | _ | `preserve_pause` |

### データ準備スクリプト
```bash
python scripts/prepare_japanese_data.py --input_dir /path/to/dataset --output_dir Data
```

### 日本語学習
```bash
# 多言語PL-BERTをダウンロード
# https://huggingface.co/papercup-ai/multilingual-pl-bert から Utils/PLBERT_multilingual/ に配置

# 日本語設定で学習
accelerate launch train_first.py --config_path ./Configs/config_japanese.yml
```

## 既知の制限事項

- 第2段階のトレーニングはData Parallel（DP）のみ対応。DDPは非対応
- 混合精度（mixed precision）は第1段階では避ける（NaN発生の可能性）
- 古いGPUでは高周波ノイズが発生する場合あり（CPUで推論を推奨）
- 非英語データセットには多言語版PL-BERTを使用
