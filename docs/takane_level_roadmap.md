# Takaneレベル精度達成のためのロードマップ

## 1. 現状分析

### 1.1 利用可能なリソース

| リソース | 仕様 |
|---------|------|
| **データセット** | 高品質日本語音声 1万時間 |
| **GPU** | A5000 (24GB) × 7台 |
| **期間** | 1ヶ月 |
| **総GPU時間** | 約5,040時間 (7台 × 24時間 × 30日) |

### 1.2 Takaneの技術仕様（参考）

| 項目 | Takane |
|------|--------|
| アーキテクチャ | 完全自動回帰型エンコーダ-デコーダTransformer |
| パラメータ数 | 500M |
| サンプリングレート | 44.1kHz |
| トークナイザー | Kanadec (25fps) |
| 特徴 | セマンティック・アコースティック蒸留、AHA |

### 1.3 比較対象プロジェクトの学習リソース

| プロジェクト | データ | GPU | 期間 |
|------------|-------|-----|------|
| Tsukasa_Speech | 800時間 | 8×A40 + 2×V100 | 3週間 |
| Kokoro-82M | 数百時間 | A100 | 1000時間 |
| CosyVoice 3 | 100万時間 | - | - |
| F5-TTS | 10万時間 | - | - |

**結論**: 1万時間のデータと7台のA5000は、Tsukasa_Speech（800時間）を大幅に上回る規模であり、高品質なモデルの訓練が十分可能。

---

## 2. アーキテクチャ選択肢

### 2.1 選択肢A: StyleTTS2改良型（Tsukasa方式）

**概要**: 現在のStyleTTS2をベースに、日本語に最適化した改良を加える

**改良点**:
- mLSTMレイヤー導入
- PL-BERT日本語再訓練
- Whisperエンコーダ採用
- 非言語音対応強化

**メリット**:
- 既存コードベースを活用可能
- 実績のある手法（Tsukasa_Speech）
- 比較的低リスク

**デメリット**:
- Diffusionモデルの制約が残る
- 44.1kHz対応には追加作業が必要

**推定学習時間**: 2-3週間

### 2.2 選択肢B: 自動回帰型（VALL-E 2/CosyVoice方式）

**概要**: Takaneと同様、完全自動回帰型Transformerへ移行

**主要技術**:
- 音声トークナイザー（WavTokenizer / EnCodec）
- AR（自動回帰）+ NAR（非自動回帰）の2段階モデル
- Repetition Aware Sampling (RAS)
- Grouped Code Modeling (GCM)

**メリット**:
- 最新のSOTA手法
- ゼロショット音声クローン能力
- 高い表現力

**デメリット**:
- 完全な再実装が必要
- 学習が不安定になりやすい
- デバッグが困難

**推定学習時間**: 3-4週間

### 2.3 選択肢C: Flow Matching型（F5-TTS / CosyVoice 3方式）

**概要**: Diffusion TransformerとFlow Matchingを組み合わせた非自動回帰型

**主要技術**:
- DiT (Diffusion Transformer)
- Flow Matching
- ConvNeXt V2
- FSQ (Finite Scalar Quantization)

**メリット**:
- 高速推論（RTF 0.15）
- テキスト-音声アライメントが不要
- 学習が比較的安定

**デメリット**:
- 新しい手法のため実装例が少ない
- 日本語特化の知見が限られる

**推定学習時間**: 2-3週間

---

## 3. 推奨ロードマップ

### 3.1 推奨アプローチ: ハイブリッド戦略

**Phase 1と2を並行開発し、Phase 3で統合**

```
Phase 1: StyleTTS2改良型（確実な成果）
    ↓
Phase 2: CosyVoice 3 / F5-TTSの実験（先進技術習得）
    ↓
Phase 3: 最良の手法を選択・統合
```

---

### Phase 1: StyleTTS2日本語最適化（Week 1-2）

#### 1.1 コアモジュールの再訓練

**PL-BERT日本語再訓練**
```
データ: 日本語Wikipedia + 書籍テキスト
GPU: A5000 × 2台
期間: 3-5日
目標: 日本語の文脈理解向上
```

**Pitch Extractor再訓練**
```
データ: 1万時間の音声データ
GPU: A5000 × 1台
期間: 2-3日
目標: 日本語のピッチパターン学習
```

**Text Aligner再訓練**
```
データ: 音声-テキストペア
GPU: A5000 × 1台
期間: 2-3日
目標: 日本語音素アライメント精度向上
```

#### 1.2 アーキテクチャ改良

**mLSTMレイヤー導入**
```python
# models.pyの変更
# 標準LSTMをmLSTMに置換
from xlstm import mLSTM

class ImprovedProsodyPredictor(nn.Module):
    def __init__(self, ...):
        # LSTMをmLSTMに置換
        self.lstm = mLSTM(input_size, hidden_size, ...)
```

**Whisperエンコーダ採用（SLM Discriminator）**
```python
# WavLMの代わりにWhisperを使用
from transformers import WhisperModel

class WhisperSLMDiscriminator(nn.Module):
    def __init__(self):
        self.whisper = WhisperModel.from_pretrained("openai/whisper-large-v3")
```

#### 1.3 サンプリングレート向上

**48kHz対応**
```yaml
# config_japanese_48k.yml
preprocess_params:
  sr: 48000
  spect_params:
    n_fft: 4096
    win_length: 2400
    hop_length: 600
```

---

### Phase 2: 次世代アーキテクチャ実験（Week 2-3）

#### 2.1 WavTokenizerの導入

**インストールと設定**
```bash
git clone https://github.com/jishengpeng/WavTokenizer
pip install -r requirements.txt
```

**日本語音声のトークン化**
```python
from wavtokenizer import WavTokenizer

# 24kHz、75トークン/秒モデルを使用
tokenizer = WavTokenizer.from_pretrained("wavtokenizer-medium-75token")

# 音声→トークン
tokens = tokenizer.encode(audio_24k)

# トークン→音声
reconstructed = tokenizer.decode(tokens)
```

#### 2.2 CosyVoice 3アーキテクチャの検討

**主要コンポーネント**:
1. **Speech Tokenizer**: FSQ (Finite Scalar Quantization)
2. **Text-Speech LM**: LLMバックボーン直接使用
3. **Flow Matching**: チャンク対応因果的Flow Matching

**日本語対応の確認**:
- CosyVoice 3は日本語を含む9言語に対応
- 既存の日本語モデルをファインチューニング可能

#### 2.3 F5-TTSの検討

**アーキテクチャ**:
- DiT (Diffusion Transformer)
- ConvNeXt V2
- Flow Matching
- テキスト長調整（Filler Token）

**利点**:
- Duration Modelが不要
- テキストエンコーダが不要
- 音素アライメントが不要

---

### Phase 3: 統合と最適化（Week 3-4）

#### 3.1 モデル選択基準

| 基準 | 重み | 評価方法 |
|------|-----|---------|
| 音声品質 (MOS) | 30% | 主観評価 |
| 発音精度 (CER) | 25% | 自動評価 |
| 話者類似度 | 20% | 埋め込み類似度 |
| 推論速度 (RTF) | 15% | ベンチマーク |
| 安定性 | 10% | 障害率 |

#### 3.2 ハイパーパラメータ最適化

```yaml
# 推奨設定
batch_size: 32  # A5000 24GBに最適化
gradient_accumulation: 4
learning_rate: 1e-4
warmup_steps: 10000
max_steps: 500000
```

#### 3.3 Anti-Hallucination対策（AHA）

```python
def anti_hallucination_sampling(model, text, num_candidates=5):
    """複数候補を生成し、最適な出力を選択"""
    candidates = []
    for _ in range(num_candidates):
        output = model.generate(text, temperature=0.8)
        score = evaluate_consistency(text, output)
        candidates.append((output, score))

    # 最高スコアの候補を選択
    return max(candidates, key=lambda x: x[1])[0]
```

---

## 4. 技術詳細

### 4.1 音声トークナイザー比較

| トークナイザー | トークン/秒 | サンプリングレート | 特徴 |
|--------------|-----------|-----------------|------|
| EnCodec | 75 | 24kHz | Meta製、広く使用 |
| WavTokenizer | 40-75 | 24kHz | ICLR 2025、SOTA圧縮 |
| DAC | 86 | 44.1kHz | 高音質、音楽対応 |
| FSQ (CosyVoice) | 25 | 22.05kHz | セマンティック分離 |

**推奨**: WavTokenizer-medium（75トークン/秒、24kHz）
- 理由: バランスの取れた圧縮率と品質

### 4.2 言語モデルバックボーン選択

| モデル | パラメータ | 日本語対応 | 推奨度 |
|--------|----------|----------|--------|
| Qwen2-0.5B | 500M | ○ | ★★★ |
| Llama-3.2-1B | 1B | △ | ★★ |
| Japanese-StableLM | 1.6B | ◎ | ★★★ |

**推奨**: Qwen2-0.5BまたはJapanese-StableLM
- 日本語性能と計算効率のバランス

### 4.3 学習スケジュール

```
Week 1:
├── Day 1-2: PL-BERT日本語訓練開始 (GPU 0-1)
├── Day 2-3: Pitch Extractor訓練 (GPU 2)
├── Day 3-4: Text Aligner訓練 (GPU 3)
└── Day 5-7: StyleTTS2本体訓練開始 (GPU 4-6)

Week 2:
├── Day 8-10: StyleTTS2訓練継続
├── Day 11-12: WavTokenizer実験 (GPU 0-1)
└── Day 13-14: CosyVoice 3実験 (GPU 2-3)

Week 3:
├── Day 15-17: 各モデル評価
├── Day 18-20: 最良モデル選択・追加訓練
└── Day 21: 中間評価

Week 4:
├── Day 22-25: 最終訓練・最適化
├── Day 26-28: 推論最適化・テスト
└── Day 29-30: デプロイ準備
```

---

## 5. 実装の優先順位

### 5.1 必須（Must Have）

1. **日本語PL-BERT**: 言語理解の基盤
2. **mLSTMレイヤー**: 表現力向上
3. **大規模データ訓練**: 1万時間を活用
4. **音声品質評価**: MOSスコア測定

### 5.2 推奨（Should Have）

1. **WavTokenizer統合**: 効率的な音声表現
2. **48kHz対応**: 高音質化
3. **Whisperエンコーダ**: SLM改善
4. **AHA実装**: 安定性向上

### 5.3 検討（Could Have）

1. **CosyVoice 3移行**: 最新アーキテクチャ
2. **強化学習最適化**: DiffRO
3. **ストリーミング対応**: 低遅延

---

## 6. リスクと対策

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| 学習不安定 | 中 | 高 | チェックポイント頻繁保存、学習率スケジュール調整 |
| GPU メモリ不足 | 低 | 中 | Gradient Checkpointing、Mixed Precision |
| 過学習 | 中 | 中 | データ拡張、Dropout調整 |
| 音声品質不足 | 中 | 高 | 複数アーキテクチャ並行実験 |

---

## 7. 成功指標

| 指標 | 目標値 | Takane参考値 |
|------|--------|-------------|
| MOS (音声品質) | > 4.0 | 4.2+ |
| CER (発音精度) | < 2% | - |
| 話者類似度 | > 75% | - |
| RTF (推論速度) | < 0.5 | 0.3 |

---

## 8. 参考リソース

### 論文
- [VALL-E 2](https://arxiv.org/abs/2406.05370) - 人間レベルゼロショットTTS
- [WavTokenizer](https://arxiv.org/abs/2408.16532) - ICLR 2025採択
- [F5-TTS](https://arxiv.org/abs/2410.06885) - Flow Matching TTS
- [CosyVoice 3](https://arxiv.org/abs/2505.17589) - 大規模ストリーミングTTS

### 実装
- [WavTokenizer GitHub](https://github.com/jishengpeng/WavTokenizer)
- [F5-TTS GitHub](https://github.com/SWivid/F5-TTS)
- [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [Tsukasa-Speech GitHub](https://github.com/Respaired/Tsukasa-Speech/)

### HuggingFace
- [Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- [WavTokenizer Models](https://huggingface.co/novateur/WavTokenizer-medium-music-audio-75token)

---

## 9. 結論

### 推奨戦略

**短期（1ヶ月）**: StyleTTS2改良型をメインに、CosyVoice 3/F5-TTSの実験を並行

1. **Week 1-2**: StyleTTS2 + mLSTM + 日本語PL-BERT
2. **Week 2-3**: WavTokenizer実験、CosyVoice 3ファインチューニング
3. **Week 3-4**: 最良モデル選択、最終訓練

**中期（3ヶ月）**: CosyVoice 3アーキテクチャへの完全移行を検討

**長期（6ヶ月以上）**: 独自の音声トークナイザー開発、強化学習最適化

### 期待される成果

- **MOS 4.0以上**の日本語音声品質
- **CER 2%未満**の発音精度
- **RTF 0.5未満**の推論速度
- ゼロショット音声クローン能力

---

*このロードマップは2025年1月時点の技術動向に基づいています。*
