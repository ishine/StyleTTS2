# StyleTTS2ベース日本語TTSプロジェクト調査レポート

## 1. 概要

### 1.1 StyleTTS2の基本アーキテクチャ

StyleTTS2は、スタイル拡散（Style Diffusion）と大規模音声言語モデル（SLM）による敵対的学習を組み合わせた最先端のText-to-Speech（TTS）モデルです。

**主要コンポーネント**:
- **Text Encoder**: テキストの言語特徴を抽出
- **Prosody Predictor**: 韻律（ピッチ、持続時間）を予測
- **Style Encoder**: 話者スタイルをエンコード
- **Diffusion Model**: スタイルベクトルを生成
- **Decoder (iSTFTNet/HiFiGAN)**: 波形生成
- **SLM Discriminator (WavLM)**: 敵対的学習による品質向上

### 1.2 日本語TTSの課題

1. **アクセント**: 日本語は高低アクセントが重要で、誤ったアクセントは大きな違和感を生む
2. **音素体系**: 英語IPAとは異なる音素セットが必要
3. **言語モデル**: PL-BERTが英語専用のため、日本語用の再訓練が必要
4. **データセット**: 高品質な日本語音声データの確保

---

## 2. Kokoro-82M

### 2.1 プロジェクト情報

| 項目 | 内容 |
|------|------|
| 開発者 | hexgrad |
| リポジトリ | https://github.com/hexgrad/kokoro |
| HuggingFace | https://huggingface.co/hexgrad/Kokoro-82M |
| パラメータ数 | 82M |
| ライセンス | Apache 2.0 |
| 対応言語 | 8言語（英語、日本語、中国語、フランス語、スペイン語、ヒンディー語、イタリア語、ポルトガル語） |

### 2.2 アーキテクチャの特徴

Kokoro-82Mは、StyleTTS2を大幅に軽量化したモデルです。

**設計方針**:
- デコーダのみのアーキテクチャ（エンコーダ・Diffusion非公開）
- StyleTTS2とiSTFTNetをベースに構築

### 2.3 StyleTTS2からの改善点

| 改善項目 | StyleTTS2 | Kokoro-82M |
|---------|-----------|------------|
| Diffusionモデル | あり（スタイル生成） | **削除**（軽量化） |
| 波形生成 | HiFiGAN/iSTFTNet | **iSTFTNetに統一** |
| パラメータ数 | 数百M以上 | **82M**（大幅削減） |
| 推論速度 | 標準 | **高速化** |
| コスト | 標準 | **$1/100万文字未満** |

**主な変更点**:
1. **Diffusion削除**: スタイル拡散モデルを削除し、複雑さを大幅に低減
2. **効率的な波形生成**: iSTFTNetによる高速な音声合成
3. **軽量化**: 82Mパラメータで大規模モデル相当の品質を実現

### 2.4 日本語対応状況

- 日本語は8言語中の1つとして対応
- 言語コード: `'j'`
- G2Pライブラリ: `misaki[ja]`を使用
- **注意**: 現時点では日本語音声は「英語を日本語アクセントで話す」レベルの可能性あり

### 2.5 学習コスト

| 項目 | 数値 |
|------|------|
| GPU | A100 |
| 総学習時間 | 1000時間 |
| 総コスト | 約$1000 |

### 2.6 利用可能性

- **オープンソース**: Apache 2.0ライセンスで完全公開
- **pip install**: `kokoro>=0.9.2`で簡単インストール
- **HuggingFace Spaces**: デモ利用可能

---

## 3. Tsukasa_Speech

### 3.1 プロジェクト情報

| 項目 | 内容 |
|------|------|
| 開発者 | Respair (Soshyant) |
| HuggingFace | https://huggingface.co/Respair/Tsukasa_Speech |
| GitHub | https://github.com/Respaired/Tsukasa-Speech/ |
| 学習データ | ~800時間（アニメ日本語） |
| ライセンス | CC-BY-NC-4.0 |
| 対応言語 | 日本語のみ |

### 3.2 アーキテクチャの特徴

Tsukasa_Speechは、StyleTTS2をベースに日本語に特化した大幅な改良を加えたモデルです。

**設計方針**:
- 表現力と制御性の最大化
- 日本語（特にアニメ日本語）に最適化

### 3.3 StyleTTS2からの改善点

| 改善項目 | StyleTTS2 | Tsukasa_Speech |
|---------|-----------|----------------|
| LSTMレイヤー | 標準PyTorch LSTM | **mLSTM**（容量増加） |
| Text/Prosody Encoder | 標準 | **パラメータ数増加** |
| PL-BERT | 英語Wikipedia訓練済み | **日本語で再訓練** |
| Pitch Extractor | 事前訓練済み | **再訓練** |
| Text Aligner | 事前訓練済み | **再訓練** |
| SLM | WavLM | **Whisper Encoder** |
| 非言語音 | 限定的 | **改善（ため息、間、笑い）** |

**主な変更点**:

1. **mLSTMレイヤー導入**
   - 標準LSTMをmLSTM（matrix LSTM）に置換
   - より大きなモデル容量と表現力

2. **コアモジュールの再訓練**
   - PL-BERT: 日本語データで一から訓練
   - Pitch Extractor: 日本語音声に最適化
   - Text Aligner: 日本語音素に対応

3. **Whisperエンコーダ採用**
   - WavLMの代わりにWhisperのエンコーダを使用
   - 多言語対応の恩恵を受ける

4. **スマート音素化**
   - ローマ字入力をサポート
   - 日本語とローマ字の混合テキストを処理
   - 独自のアルゴリズムで自然な音素変換

5. **非言語音の改善**
   - ため息、間（ポーズ）、笑いなどの表現が向上

### 3.4 利用可能なチェックポイント

| チェックポイント | サンプリングレート | 学習データ | 特徴 |
|----------------|------------------|----------|------|
| Tsukasa | 24kHz | ~800時間 | フル機能 |
| Tsumugi | 48kHz | ~300時間 | 高音質だが機能限定 |

### 3.5 学習環境

```
GPU: 8x A40 + 2x V100 (32GB each)
学習データ: 750-800時間
精度: BFloat16
学習期間: 約3週間
総開発期間: 約3ヶ月（データパイプライン含む）
CO2排出: 約66.6 kg CO2eq
```

### 3.6 利用可能性

- **ライセンス**: CC-BY-NC-4.0（非商用のみ）
- **HuggingFace Space**: デモ利用可能
- **学習コード**: 公開（一部修正が必要）

---

## 4. Takane

### 4.1 プロジェクト情報

| 項目 | 内容 |
|------|------|
| 開発者 | Respair (Soshyant) |
| HuggingFace Space | https://huggingface.co/spaces/Respair/Takane |
| パラメータ数 | 500M |
| 対応言語 | 日本語のみ |
| 状態 | **デモ終了、モデル非公開** |

### 4.2 アーキテクチャの特徴

Takaneは、StyleTTS2の拡散ベース手法から**完全自動回帰型**へと大きくアーキテクチャを転換したモデルです。

**設計方針**:
- 完全自動回帰型エンコーダ-デコーダTransformer
- Kanadecトークナイザーによる高品質音声トークン化

### 4.3 StyleTTS2からの改善点

| 改善項目 | StyleTTS2 | Takane |
|---------|-----------|--------|
| アーキテクチャ | 拡散ベース + 条件付け | **完全自動回帰型Transformer** |
| パラメータ数 | 可変 | **500M** |
| サンプリングレート | 24kHz | **44.1kHz** |
| トークナイザー | - | **Kanadec（25fps）** |

**主な変更点**:

1. **アーキテクチャ転換**
   - StyleTTS2の拡散モデルを完全に廃止
   - エンコーダ-デコーダ型完全自動回帰Transformerを採用

2. **Kanadecトークナイザー**
   - 世界初の44.1kHz、25フレーム/秒の音声トークナイザー
   - セマンティック・アコースティック蒸留を活用
   - より高品質な音声表現が可能

3. **Anti-Hallucination Algorithm (AHA)**
   - 複数の候補を並列生成
   - 最適な出力を自動選択
   - ハルシネーション（幻覚）を低減

4. **継続時間制御**
   - カスタムRoPE（Rotary Position Embedding）による継続時間操作
   - より細かい音声制御が可能

### 4.4 公開状況

**デモ終了の理由**:
- サーバー負荷の増大
- 声優への悪用懸念（話者IDマッピングが可能）
- 倫理的配慮を優先

**利用制限**:
- モデル非公開
- API販売予定なし
- 事前生成されたサンプルのみ閲覧可能

---

## 5. 比較表

| 項目 | Kokoro-82M | Tsukasa_Speech | Takane |
|------|------------|----------------|--------|
| **パラメータ数** | 82M | 非公開 | 500M |
| **ベースアーキテクチャ** | StyleTTS2 (Decoder) | StyleTTS2 (Full) | 自動回帰Transformer |
| **サンプリングレート** | 24kHz | 24/48kHz | 44.1kHz |
| **対応言語** | 8言語 | 日本語のみ | 日本語のみ |
| **日本語特化度** | 低〜中 | **高** | **高** |
| **ライセンス** | Apache 2.0 | CC-BY-NC-4.0 | 非公開 |
| **利用可能性** | **完全公開** | 公開（非商用） | **利用不可** |
| **学習データ** | 数百時間 | ~800時間 | 非公開 |
| **推論速度** | 高速 | 標準 | 高速（ターボモード） |

---

## 6. StyleTTS2日本語対応への示唆

### 6.1 各プロジェクトから学べる改善点

#### Kokoro-82Mから
1. **軽量化戦略**: Diffusion削除による効率化は、リソース制約がある場合に有効
2. **iSTFTNet統一**: 波形生成の効率化
3. **多言語対応**: 複数言語を1つのモデルで扱うアプローチ

#### Tsukasa_Speechから
1. **mLSTM導入**: モデル容量と表現力の向上
2. **コアモジュール再訓練**: PL-BERT、Pitch Extractor、Text Alignerの日本語特化
3. **Whisperエンコーダ**: WavLMの代替として検討価値あり
4. **非言語音対応**: ため息、間、笑いなどの表現改善
5. **スマート音素化**: ローマ字・日本語混合テキストの処理

#### Takaneから
1. **高サンプリングレート**: 44.1kHzによる高音質化
2. **自動回帰型への転換**: Diffusionに依存しない設計の可能性
3. **AHA**: ハルシネーション対策

### 6.2 推奨アプローチ

#### 短期的改善（現在の実装に追加可能）
1. **pyopenjtalk-plus**: 日本語G2P（既に実装済み）
2. **多言語PL-BERT**: papercup-ai/multilingual-pl-bertの導入
3. **日本語データセットでのファインチューニング**

#### 中期的改善
1. **PL-BERTの日本語再訓練**: Tsukasa方式に倣う
2. **mLSTMレイヤーの導入**: 表現力向上
3. **Pitch ExtractorとText Alignerの再訓練**

#### 長期的改善
1. **48kHz対応**: Tsumugiのような高音質バージョン
2. **Whisperエンコーダへの置換**: SLM Discriminatorの改善
3. **非言語音対応の強化**

---

## 参考リンク

### Kokoro-82M
- GitHub: https://github.com/hexgrad/kokoro
- HuggingFace: https://huggingface.co/hexgrad/Kokoro-82M
- Demo: https://huggingface.co/spaces/hexgrad/Kokoro-TTS

### Tsukasa_Speech
- HuggingFace: https://huggingface.co/Respair/Tsukasa_Speech
- GitHub: https://github.com/Respaired/Tsukasa-Speech/
- Demo: https://huggingface.co/spaces/Respair/Tsukasa_Speech
- 48kHz版: https://huggingface.co/spaces/Respair/Tsumugi_48khz

### Takane
- HuggingFace Space: https://huggingface.co/spaces/Respair/Takane

### StyleTTS2オリジナル
- GitHub: https://github.com/yl4579/StyleTTS2
- 論文: https://arxiv.org/abs/2306.07691

---

*このレポートは2025年1月時点の情報に基づいています。*
