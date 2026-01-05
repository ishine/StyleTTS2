# StyleTTS 2: スタイル拡散と大規模音声言語モデルによる敵対的学習を用いた人間レベルの音声合成

### Yinghao Aaron Li, Cong Han, Vinay S. Raghavan, Gavin Mischler, Nima Mesgarani

> 本論文では、スタイル拡散と大規模音声言語モデル（SLM）による敵対的学習を活用し、人間レベルの音声合成を実現するテキスト音声変換（TTS）モデル「StyleTTS 2」を提案します。StyleTTS 2は、拡散モデルを通じてスタイルを潜在確率変数としてモデル化することで、参照音声を必要とせずにテキストに最適なスタイルを生成し、効率的な潜在拡散を実現しながら拡散モデルが提供する多様な音声合成の恩恵を受けます。さらに、WavLMなどの大規模事前学習済みSLMを識別器として採用し、新しい微分可能な持続時間モデリングによるエンドツーエンド学習を行うことで、音声の自然性を向上させています。StyleTTS 2は、単一話者のLJSpeechデータセットでは人間の録音を上回り、複数話者のVCTKデータセットでは同等の性能を達成しました（英語ネイティブスピーカーによる評価）。また、LibriTTSデータセットで学習した場合、ゼロショット話者適応において既存の公開モデルを上回る性能を示しました。本研究は、単一話者・複数話者の両データセットで初めて人間レベルのTTS合成を達成し、スタイル拡散と大規模SLMによる敵対的学習の可能性を示しました。

論文: [https://arxiv.org/abs/2306.07691](https://arxiv.org/abs/2306.07691)

音声サンプル: [https://styletts2.github.io/](https://styletts2.github.io/)

オンラインデモ: [Hugging Face](https://huggingface.co/spaces/styletts2/styletts2)（[@fakerybakery](https://github.com/fakerybakery)氏に感謝）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yl4579/StyleTTS2/blob/main/) [![Discord](https://img.shields.io/discord/1197679063150637117?logo=discord&logoColor=white&label=Join%20our%20Community)](https://discord.gg/ha8sxdG2K4)

## 開発状況
- [x] 単一話者モデル（LJSpeech）のトレーニング・推論デモコード
- [x] 複数話者モデル（VCTK、LibriTTS）のトレーニングコードのテスト
- [x] 複数話者モデルのデモコード完成と事前学習済みモデルのアップロード
- [x] 事前学習済み複数話者モデルをベースとした新規話者向けファインチューニングスクリプトの追加
- [ ] `train_second.py`のDDP（accelerator）修正 **（解決に至っていません。協力いただける方は[#7](https://github.com/yl4579/StyleTTS2/issues/7)をご覧ください）**

## 前提条件

### uvでのインストール（推奨）

1. [uv](https://docs.astral.sh/uv/)をインストール:
```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. リポジトリをクローン:
```bash
git clone https://github.com/yl4579/StyleTTS2.git
cd StyleTTS2
```

3. 依存関係をインストール（デフォルトでPyTorch CUDA 12.1が設定されています）:
```bash
uv sync
```

> **注意 (Windows)**: `monotonic-align`のビルドにはC++ワークロード付きのVisual Studio Build Toolsが必要です。`vswhere.exe`がPATHに含まれていることを確認してください:
> ```bash
> set PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\Installer
> ```

4. （オプション）推論デモ用:
```bash
uv sync --group inference

# espeak-ngをインストール
# Windows: https://github.com/espeak-ng/espeak-ng/releases からダウンロード
# Linux: sudo apt-get install espeak-ng
```

### CUDAバージョンの変更

別のCUDAバージョンを使用するには、`pyproject.toml`を編集してください:

```toml
# CUDA 11.8の場合
[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu118" }
torchaudio = { index = "pytorch-cu118" }
```

その後、`uv sync`を再実行してください。

### データセットの準備

[LJSpeechデータセット](https://keithito.com/LJ-Speech-Dataset/)をダウンロードして展開し、dataフォルダに配置して24 kHzにアップサンプリングしてください。テキストアライナーとピッチ抽出器は24 kHzデータで事前学習されていますが、前処理を変更して独自の前処理で再学習することも可能です。
LibriTTSの場合は、train-clean-360とtrain-clean-100を結合し、フォルダ名をtrain-clean-460に変更する必要があります（例: [val_list_libritts.txt](https://github.com/yl4579/StyleTTS/blob/main/Data/val_list_libritts.txt)を参照）。

## トレーニング
第1段階のトレーニング:
```bash
accelerate launch train_first.py --config_path ./Configs/config.yml
```
第2段階のトレーニング **（DDP版は動作しないため、現在のバージョンはDPを使用しています。協力いただける方は[#7](https://github.com/yl4579/StyleTTS2/issues/7)をご覧ください）**:
```bash
python train_second.py --config_path ./Configs/config.yml
```
両方を連続して実行すると、第1段階と第2段階の両方がトレーニングされます。モデルは「epoch_1st_%05d.pth」および「epoch_2nd_%05d.pth」の形式で保存されます。チェックポイントとTensorboardログは`log_dir`に保存されます。

データリストの形式は`filename.wav|transcription|speaker`である必要があります。例として[val_list.txt](https://github.com/yl4579/StyleTTS2/blob/main/Data/val_list.txt)を参照してください。話者ラベルは複数話者モデルで必要です（スタイル拡散モデルのトレーニング用に参照音声をサンプリングするため）。

### 重要な設定
[config.yml](https://github.com/yl4579/StyleTTS2/blob/main/Configs/config.yml)には、いくつかの重要な設定があります:
- `OOD_data`: SLM敵対的学習用の分布外テキストのパス。形式は`text|anything`です。
- `min_length`: トレーニング用OODテキストの最小長。合成音声が最小長を持つことを保証します。
- `max_len`: トレーニング用音声の最大長。単位はフレームです。デフォルトのホップサイズは300なので、1フレームは約`300 / 24000`（0.0125）秒です。メモリ不足が発生した場合は値を下げてください。
- `multispeaker`: 複数話者モデルをトレーニングする場合はtrueに設定します。単一話者モデルと複数話者モデルではデノイザーのアーキテクチャが異なるため必要です。
- `batch_percentage`: SLM敵対的学習中のメモリ不足（OOM）を防ぐための設定です。OOM問題が発生した場合は、この値を下げてください。

### 事前学習済みモジュール
[Utils](https://github.com/yl4579/StyleTTS2/tree/main/Utils)フォルダには、3つの事前学習済みモデルがあります:
- **[ASR](https://github.com/yl4579/StyleTTS2/tree/main/Utils/ASR)フォルダ**: 英語（LibriTTS）、日本語（JVS）、中国語（AiShell）コーパスで事前学習されたテキストアライナーが含まれています。ほとんどの他の言語でもファインチューニングなしで良好に動作しますが、独自のテキストアライナーを[yl4579/AuxiliaryASR](https://github.com/yl4579/AuxiliaryASR)でトレーニングすることもできます。
- **[JDC](https://github.com/yl4579/StyleTTS2/tree/main/Utils/JDC)フォルダ**: 英語（LibriTTS）コーパスのみで事前学習されたピッチ抽出器が含まれています。ただし、F0は言語に依存しないため、他の言語でも良好に動作します。歌唱コーパスでトレーニングする場合は、[yl4579/PitchExtractor](https://github.com/yl4579/PitchExtractor)で新しいピッチ抽出器をトレーニングすることを推奨します。
- **[PLBERT](https://github.com/yl4579/StyleTTS2/tree/main/Utils/PLBERT)フォルダ**: 英語（Wikipedia）コーパスのみで事前学習された[PL-BERT](https://arxiv.org/abs/2301.08810)モデルが含まれています。他の言語ではうまく動作しない可能性があるため、[yl4579/PL-BERT](https://github.com/yl4579/PL-BERT)を使用して異なる言語用のPL-BERTをトレーニングする必要があります。14言語をサポートする[多言語PL-BERT](https://huggingface.co/papercup-ai/multilingual-pl-bert)も使用できます。

### よくある問題
- **損失がNaNになる**: 第1段階の場合、混合精度を使用しないでください。特定のデータセットでバッチサイズが適切に設定されていない場合（16以上が必要）、損失がNaNになることがあります。第2段階でも、異なるバッチサイズを試してください。バッチサイズが大きいほどNaN損失が発生しやすくなります。バッチサイズは16を推奨します。詳細はissue [#10](https://github.com/yl4579/StyleTTS2/issues/10)および[#11](https://github.com/yl4579/StyleTTS2/issues/11)を参照してください。
- **メモリ不足**: `batch_size`または`max_len`を下げてください。詳細はissue [#10](https://github.com/yl4579/StyleTTS2/issues/10)を参照してください。
- **英語以外のデータセット**: 任意の言語でトレーニングできますが、その言語用の事前学習済みPL-BERTモデルが必要です。14言語をサポートする[多言語PL-BERT](https://huggingface.co/papercup-ai/multilingual-pl-bert)があります。中国語データセットでのトレーニング例については、[yl4579/StyleTTS#10](https://github.com/yl4579/StyleTTS/issues/10)および[#70](https://github.com/yl4579/StyleTTS2/issues/70)を参照してください。 

## ファインチューニング
このスクリプトは`train_second.py`を修正したもので、DDPが`train_second.py`で動作しないためDPを使用しています。この問題の解決に協力いただける方は、上記の太字セクションをご覧ください。
```bash
python train_finetune.py --config_path ./Configs/config_ft.yml
```
LibriTTSチェックポイントをダウンロードしてフォルダ内に展開してください。デフォルト設定の`config_ft.yml`は、1時間の音声データ（約1000サンプル）を使用してLJSpeechで50エポックのファインチューニングを行います。これはNVidia A100 4台で約4時間かかります。品質は、24時間の音声データでゼロからトレーニングしたLJSpeechモデル（A100 4台で約2.5日）よりわずかに劣ります（LJSpeechでのNaturalSpeechと同程度）。サンプルは[#65 (comment)](https://github.com/yl4579/StyleTTS2/discussions/65#discussioncomment-7668393)で確認できます。

**単一GPU**を使用している場合（スクリプトがDDPで動作しないため）、トレーニング速度とVRAMを節約するには以下を実行できます（[@korakoe](https://github.com/korakoe)氏が[#100](https://github.com/yl4579/StyleTTS2/pull/100)でスクリプトを作成）:
```bash
accelerate launch --mixed_precision=fp16 --num_processes=1 train_finetune_accelerate.py --config_path ./Configs/config_ft.yml
```
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yl4579/StyleTTS2/blob/main/Colab/StyleTTS2_Finetune_Demo.ipynb)

### よくある問題
[@Kreevoz](https://github.com/Kreevoz)氏がファインチューニングでの一般的な問題と音質最大化の提案について詳細なメモを作成しています: [#81](https://github.com/yl4579/StyleTTS2/discussions/81)。これらの一部はゼロからのトレーニングにも適用されます。[@IIEleven11](https://github.com/IIEleven11)氏もファインチューニングのガイドラインを作成しています: [#128](https://github.com/yl4579/StyleTTS2/discussions/128)。

- **`joint_epoch`後のメモリ不足**: GPUのRAMがSLM敵対的学習の実行に十分でない可能性があります。スキップすることもできますが、品質が低下する可能性があります。`joint_epoch`を`epochs`より大きい値に設定すると、SLM敵対的学習をスキップできます。

## 推論
詳細は[Inference_LJSpeech.ipynb](https://github.com/yl4579/StyleTTS2/blob/main/Demo/Inference_LJSpeech.ipynb)（単一話者）および[Inference_LibriTTS.ipynb](https://github.com/yl4579/StyleTTS2/blob/main/Demo/Inference_LibriTTS.ipynb)（複数話者）を参照してください。LibriTTSの場合は、デモを実行する前に[reference_audio.zip](https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/reference_audio.zip)をダウンロードして`demo`フォルダに展開する必要があります。

- LJSpeechコーパス（24 kHz）で事前学習されたStyleTTS 2は[https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main](https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main)からダウンロードできます。

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yl4579/StyleTTS2/blob/main/Colab/StyleTTS2_Demo_LJSpeech.ipynb)

- LibriTTSで事前学習されたStyleTTS 2モデルは[https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main](https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main)からダウンロードできます。

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yl4579/StyleTTS2/blob/main/Colab/StyleTTS2_Demo_LibriTTS.ipynb)


StyleTTS 2をインポートして独自のコードで実行できます。ただし、推論はGPLライセンスのパッケージに依存するため、このリポジトリには直接含まれていません。[GPLライセンスのフォーク](https://github.com/NeuralVox/StyleTTS2)にはインポート可能なスクリプト、実験的なストリーミングAPIなどがあります。gruutを使用する[完全MITライセンスのパッケージ](https://pypi.org/project/styletts2/)も利用可能です（phonemizerとgruutの不一致により品質はやや低下）。

***これらの事前学習済みモデルを使用する前に、合成する音声の使用許可がない限り、音声サンプルが事前学習済みモデルによって合成されたものであることをリスナーに通知することに同意するものとします。つまり、合成音声を公開する前に、話者が直接またはライセンスにより音声クローンの許可を与えている音声のみを使用するか、これらの音声の使用許可がない場合は合成音声であることを公表する必要があります。***

### よくある問題
- **高周波のバックグラウンドノイズ**: 古いGPUでの数値浮動小数点の差異が原因です。詳細はissue [#13](https://github.com/yl4579/StyleTTS2/issues/13)を参照してください。基本的に、より新しいGPUを使用するか、CPUで推論を行う必要があります。
- **事前学習済みモデルのライセンス**: 上記のルールに従う必要があるのは、**事前学習済みモデル**を使用し、かつ音声がトレーニングセットに**含まれていない**場合のみです（参照話者がオープンアクセスデータセットからのものでない場合）。事前学習済みモデルの使用ルールの詳細は[#37](https://github.com/yl4579/StyleTTS2/issues/37)を参照してください。

## 参考文献
- [archinetai/audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch)
- [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- [rishikksh20/iSTFTNet-pytorch](https://github.com/rishikksh20/iSTFTNet-pytorch)
- [nii-yamagishilab/project-NN-Pytorch-scripts/project/01-nsf](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf)

## ライセンス

コード: MITライセンス

事前学習済みモデル: これらの事前学習済みモデルを使用する前に、合成する音声の使用許可がない限り、音声サンプルが事前学習済みモデルによって合成されたものであることをリスナーに通知することに同意するものとします。つまり、合成音声を公開する前に、話者が直接またはライセンスにより音声クローンの許可を与えている音声のみを使用するか、これらの音声の使用許可がない場合は合成音声であることを公表する必要があります。
