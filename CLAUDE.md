# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

StyleTTS2は、スタイル拡散とSLM（Speech Language Model）敵対的学習を組み合わせた最先端のText-to-Speech（TTS）モデルです。人間レベルの音声合成を実現します。

## 主要コマンド

### 環境セットアップ
```bash
pip install -r requirements.txt
# 音素変換用（デモ実行時に必要）
pip install phonemizer
sudo apt-get install espeak-ng
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

## 既知の制限事項

- 第2段階のトレーニングはData Parallel（DP）のみ対応。DDPは非対応
- 混合精度（mixed precision）は第1段階では避ける（NaN発生の可能性）
- 古いGPUでは高周波ノイズが発生する場合あり（CPUで推論を推奨）
- 非英語データセットには多言語版PL-BERTを使用
