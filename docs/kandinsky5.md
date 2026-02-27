> 📝 Click on the language section to expand / 言語をクリックして展開

# Kandinsky 5

## Overview / 概要

This is an unofficial training and inference script for [Kandinsky 5](https://github.com/ai-forever/Kandinsky-5). The features are as follows:

- fp8 support and memory reduction by block swap
- Inference without installing Flash attention (using PyTorch's scaled dot product attention)
- LoRA training for text-to-video (T2V), image-to-video (I2V, Pro) models, and Image (T2I, Edit) models

This feature is experimental.

<details>
<summary>日本語</summary>

[Kandinsky 5](https://github.com/ai-forever/Kandinsky-5) の非公式の学習および推論スクリプトです。

以下の特徴があります：

- fp8対応およびblock swapによる省メモリ化
- Flash attentionのインストールなしでの実行（PyTorchのscaled dot product attentionを使用）
- テキストからビデオへの変換 (T2V)、画像からビデオへの変換 (I2V、Pro) モデル、および画像 (T2I、Edit) モデルの LoRA トレーニング

この機能は実験的なものです。

</details>

## Download the model / モデルのダウンロード

Download the model weights from the [Kandinsky 5.0 Collection](https://huggingface.co/collections/ai-forever/kandinsky-50) on Hugging Face.

### DiT Model / DiTモデル

This document focuses on **Pro** models. The trainer also works with **Lite** models.
本ドキュメントでは **Pro** モデルを中心に説明しますが、トレーナーは **Lite** モデルでも動作します。

Download a Pro DiT `.safetensors` checkpoint from the Kandinsky 5.0 Collection (e.g. `kandinsky5pro_t2v_pretrain_5s.safetensors` or `kandinsky5pro_i2v_sft_5s.safetensors`).

### VAE

Kandinsky 5 uses the HunyuanVideo 3D VAE for video tasks. Download `diffusion_pytorch_model.safetensors` (or `pytorch_model.pt`) from:
https://huggingface.co/hunyuanvideo-community/HunyuanVideo . Image generation/edit tasks use [Flux 1 VAE](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main/vae)

### Text Encoders / テキストエンコーダ

Kandinsky 5 uses Qwen2.5-VL-7B and CLIP for text encoding.

**Qwen2.5-VL-7B**: Download from https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct (or use the path to your local Qwen/Qwen2.5-VL-7B-Instruct model)

**CLIP**: Use the Hugging Face Transformers model `openai/clip-vit-large-patch14`.

Pass either the model ID (e.g., `--text_encoder_clip openai/clip-vit-large-patch14`) or a path to the locally cached snapshot directory.

### Directory Structure / ディレクトリ構造

Place them in your chosen directory structure:

```
weights/
├── model/
│   └── kandinsky5pro_t2v_pretrain_5s.safetensors
├── vae/
│   └── diffusion_pytorch_model.safetensors
├── text_encoder/
│   └── (Qwen2.5-VL-7B files)
└── text_encoder2/
    └── (openai/clip-vit-large-patch14 files)
```

<details>
<summary>日本語</summary>

Hugging Faceの[Kandinsky 5.0 Collection](https://huggingface.co/collections/ai-forever/kandinsky-50)からモデルの重みをダウンロードしてください。

このドキュメントは **Proモデル** を前提に説明しています。

**DiTモデル**: 上記のリポジトリから`.safetensors`ファイルをダウンロードしてください。

**VAE**: Kandinsky 5 は、ビデオ タスクに HunyuanVideo 3D VAE を使用します。以下から `diffusion_pytorch_model.safetensors` (または `pytorch_model.pt`) をダウンロードします。
https://huggingface.co/hunyuanvideo-community/HunyuanVideo 。画像生成/編集タスクでは[Flux 1 VAE](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main/vae)を使用します。

**テキストエンコーダ**: Qwen2.5-VL-7BとCLIPを使用します。

**Qwen2.5-VL-7B**: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct からダウンロードしてください（またはローカルの `Qwen/Qwen2.5-VL-7B-Instruct` を指定します）。

**CLIP**: Hugging Face Transformersの `openai/clip-vit-large-patch14` を使用してください（モデルIDまたはローカルにキャッシュされたsnapshotディレクトリへのパスを指定します）。

任意のディレクトリ構造に配置してください。

</details>

## List of Kandinsky 5 models / 利用可能なタスク

The `--task` option selects a model configuration (architecture, attention type, resolution, and default parameters).
The DiT checkpoint must be set explicitly via `--dit` (this overrides the task's default checkpoint path).

| # | Task | Checkpoint | Parameters | HF URL |
|---|---|---|---|---|
| 1 | k5-pro-t2v-5s-sd | kandinsky5pro_t2v_sft_5s.safetensors | T2V, 5s, 19B, Pro SFT | [kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s) |
| 2 | k5-pro-t2v-10s-sd | kandinsky5pro_t2v_sft_10s.safetensors | T2V, 10s, 19B, Pro SFT | [kandinskylab/Kandinsky-5.0-T2V-Pro-sft-10s](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-sft-10s) |
| 3 | k5-pro-i2v-5s-sd | kandinsky5pro_i2v_sft_5s.safetensors | I2V, 5s, 19B, Pro SFT | [kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s](https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s) |
| 4 | k5-pro-t2v-5s-sd | kandinsky5pro_t2v_pretrain_5s.safetensors | T2V, 5s, 19B, Pro Pretrain | [kandinskylab/Kandinsky-5.0-T2V-Pro-pretrain-5s](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-pretrain-5s) |
| 5 | k5-pro-t2v-10s-sd | kandinsky5pro_t2v_pretrain_10s.safetensors | T2V, 10s, 19B, Pro Pretrain | [kandinskylab/Kandinsky-5.0-T2V-Pro-pretrain-10s](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-pretrain-10s) |

[Kandinsky 5.0 Video Lite models](https://huggingface.co/collections/kandinskylab/kandinsky-50-video-lite) are technically supported, but were not extensively tested. Community feedback is welcome.

[Kandinsky 5.0 Image Lite models](https://huggingface.co/collections/kandinskylab/kandinsky-50-image-lite) are also supported but not extensively tested.

<details>
<summary>日本語</summary>

`--task` オプションでタスク設定（アーキテクチャ、attention、解像度、各種デフォルト値）を選択します。
DiTのチェックポイントは `--dit` で明示的に指定できます（タスクのデフォルトのパスを上書きします）。

[Kandinsky 5.0 Video Liteモデル](https://huggingface.co/collections/kandinskylab/kandinsky-50-video-lite) は技術的にはサポートされていますが、十分な動作確認はできていません。問題があればフィードバックをお願いします。

[Kandinsky 5.0 Image Lite モデル](https://huggingface.co/collections/kandinskylab/kandinsky-50-image-lite) もサポートされていますが、十分にテストされていません。

</details>

## Pre-caching / 事前キャッシュ

Pre-caching is required before training. This involves caching both latents and text encoder outputs. Note that caches created for Video Pro and Lite are NOT interchangeable with ones created for Image Lite - attempting to do this will create errors so please remake the cache when switching between image model/video model training e.g. Flux and Hunyuan VAE types.

### Notes for Kandinsky5 / Kandinsky5の注意点

- You must cache **text encoder outputs** with `kandinsky5_cache_text_encoder_outputs.py` before training.
- `--text_encoder_qwen` / `--text_encoder_clip` are Hugging Face Transformers models: pass a model ID (recommended) or a local HF snapshot directory.
- For I2V tasks, the latent cache stores both first and last frame latents (`latents_image`, always two frames) when running `kandinsky5_cache_latents.py`—one cache works for both first-only and first+last conditioning.
- If you want to train image models (T2I/I2I), you MUST use the Flux VAE and provide `--image_model_training` to `kandinsky5_cache_latents.py`!
- If you want to train image_edit (I2I), you MUST specify `--image_edit_training` to `'kandinsky5_cache_text_encoder_outputs.py` for the text encoder to see the image properly. Do NOT do this for any other mode including T2I or quality will degrade severely.

<details>
<summary>日本語</summary>

トレーニング前に事前キャッシュが必要です。これには、潜在出力とテキスト エンコーダー出力の両方のキャッシュが含まれます。 Video Pro および Lite 用に作成されたキャッシュは、Image Lite 用に作成されたキャッシュと互換性がないことに注意してください。これを実行しようとするとエラーが発生するため、画像モデルとビデオ モデルのトレーニングを切り替えるときにキャッシュを再作成してください。 Flux および Hunyuan VAE タイプ。

- 学習前に、`kandinsky5_cache_text_encoder_outputs.py` による **テキストエンコーダ出力のキャッシュ** が必須です。
- `--text_encoder_qwen` / `--text_encoder_clip` はHugging Face Transformersのモデルです。モデルID（推奨）またはローカルのHF snapshotディレクトリを指定してください。
- I2Vタスクでは、`kandinsky5_cache_latents.py` 実行時に最初と最後のフレームlatent（`latents_image`、常に2フレーム）もキャッシュされます。1回のキャッシュで first / first+last 両方のモードに対応できます。
- 画像モデル (T2I/I2I) をトレーニングする場合は、Flux VAE を使用し、`kandinsky5_cache_latents.py` に `--image_model_training` を指定する必要があります。
- image_edit (I2I) を学習させる場合、テキストエンコーダが画像を正しく認識できるように、`'kandinsky5_cache_text_encoder_outputs.py` に `--image_edit_training` を指定する必要があります。T2I を含む他のモードでは、この操作を行わないでください。そうしないと、画質が著しく低下します。

</details>

### Text Encoder Output Pre-caching / テキストエンコーダ出力の事前キャッシュ

Text encoder output pre-caching is required. Create the cache using the following command:

```bash
python kandinsky5_cache_text_encoder_outputs.py \
    --dataset_config path/to/dataset.toml \
    --text_encoder_qwen Qwen/Qwen2.5-VL-7B-Instruct \
    --text_encoder_auto \
    --text_encoder_clip openai/clip-vit-large-patch14 \
    --batch_size 4
```

Adjust `--batch_size` according to your available VRAM. Add `--image_edit_training` ONLY when training for image edit mode.

For additional options, use `python kandinsky5_cache_text_encoder_outputs.py --help`.

<details>
<summary>日本語</summary>

テキストエンコーダ出力の事前キャッシュは必須です。上のコマンド例を使用してキャッシュを作成してください。

使用可能なVRAMに合わせて `--batch_size` を調整してください。画像編集モードのトレーニングを行う場合のみ、`--image_edit_training` を追加します。

その他のオプションは `--help` で確認できます。

</details>

### Latent Pre-caching / latentの事前キャッシュ

Latent pre-caching is required. Create the cache using the following command:

```bash
python kandinsky5_cache_latents.py \
    --dataset_config path/to/dataset.toml \
    --vae path/to/vae/diffusion_pytorch_model.safetensors
```

For NABLA training, you may want to build NABLA-compatible latent caches:

```bash
python kandinsky5_cache_latents.py \
    --dataset_config path/to/dataset.toml \
    --vae path/to/vae/diffusion_pytorch_model.safetensors \
    --nabla_resize
```

If you're running low on VRAM, lower the `--batch_size`. If you want to train T2I/I2I, you MUST specify `--image_model_training` here! For image_edit (I2I) training, the `control_images` in the dataset config are used as the reference(ground truth) image. See [Dataset Config](./dataset_config.md#sample-for-image-dataset-with-control-images) for details.

For additional options, use `python kandinsky5_cache_latents.py --help`.

<details>
<summary>日本語</summary>

latentの事前キャッシュは必須です。上のコマンド例を使用してキャッシュを作成してください。

VRAMが足りない場合は、`--batch_size`を小さくしてください。T2I/I2I をトレーニングする場合は、ここでも `--image_model_training` を指定する必要があります。image_edit (I2I) トレーニングでは、データセット設定の `control_images` が参照画像（グラウンドトゥルース画像）として使用されます。詳細は [データセット設定](./dataset_config.md#sample-for-image-dataset-with-control-images) を参照してください。

NABLAで学習する場合は、NABLA互換のlatentキャッシュを作成することを推奨します：

```bash
python kandinsky5_cache_latents.py \
    --dataset_config path/to/dataset.toml \
    --vae path/to/vae/diffusion_pytorch_model.safetensors \
    --nabla_resize
```

その他のオプションは `--help` で確認できます。

</details>

## Training / 学習

Start training using the following command (input as a single line):

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    kandinsky5_train_network.py \
    --mixed_precision bf16 \
    --dataset_config path/to/dataset.toml \
    --task k5-pro-t2v-5s-sd \
    --dit path/to/kandinsky5pro_t2v_sft_5s.safetensors \
    --text_encoder_qwen Qwen/Qwen2.5-VL-7B-Instruct \
    --text_encoder_clip openai/clip-vit-large-patch14 \
    --vae path/to/vae/diffusion_pytorch_model.safetensors \
    --fp8_base --fp8_scaled \
    --sdpa \
    --gradient_checkpointing \
    --max_data_loader_n_workers 1 \
    --persistent_data_loader_workers \
    --learning_rate 1e-4 \
    --optimizer_type AdamW8Bit \
    --optimizer_args "weight_decay=0.001" "betas=(0.9,0.95)" \
    --max_grad_norm 1.0 \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_steps 100 \
    --network_module networks.lora_kandinsky \
    --network_dim 32 \
    --network_alpha 32 \
    --timestep_sampling shift \
    --discrete_flow_shift 5.0 \
    --output_dir path/to/output/folder \
    --output_name k5_lora \
    --save_every_n_epochs 1 \
    --max_train_epochs 50 
```

For I2V training, switch the task and checkpoint to an I2V preset (e.g., `k5-pro-i2v-5s-sd` with `kandinsky5pro_i2v_sft_5s.safetensors`). The latent cache already stores first and last frame latents (`latents_image`, two frames) when you run `kandinsky5_cache_latents.py`, so the same cache covers both first-only and first+last modes—no extra flags are needed beyond picking an I2V task. For image models (T2I or I2I), make sure to use the Flux VAE and set the appropriate task (`k5-lite-t2i-hd` or `k5-lite-i2i-hd`) here, as well as passing `--image_model_training` to `kandinsky5_cache_latents.py` when caching the latents in the previous step.

**Note on first+last frame conditioning**: First+last frame training support is experimental. The effectiveness and plausibility of this approach have not yet been thoroughly tested. Feedback and results from community testing are welcome.

The training settings are experimental. Appropriate learning rates, training steps, timestep distribution, etc. are not yet fully determined. Feedback is welcome.

For additional options, use `python kandinsky5_train_network.py --help`.

### Key Options / 主要オプション

- `--task`: Model configuration (architecture, attention type, resolution, sampling parameters). See Available Tasks above.
- `--dit`: Path to DiT checkpoint. **Overrides the task's default checkpoint path.** You can use any compatible checkpoint (SFT, pretrain, or your own) with any task config as long as the architecture matches.
- `--vae`: Path to VAE checkpoint (overrides task default)
- `--network_module`: Use `networks.lora_kandinsky` for Kandinsky5 LoRA

**Note**: The `--task` option only sets the model architecture and parameters, not the weights. Use `--dit` to specify which checkpoint to load.

**注意**: `--task`オプションはモデルのアーキテクチャとパラメータのみを設定し、重みは設定しません。`--dit`で読み込むチェックポイントを指定してください。

### Memory Optimization / メモリ最適化

`--gradient_checkpointing` enables gradient checkpointing to reduce VRAM usage.

`--fp8_base / --fp8_scaled` runs DiT in fp8 mode. This can significantly reduce memory consumption but may impact output quality.

If you're running low on VRAM, use `--blocks_to_swap` to offload some blocks to CPU. If you OOM on encoding prompts or caching for TE, try `--text_encoder_auto` or `--text_encoder_cpu` to run part or all of the Qwen TE on CPU.

`--gradient_checkpointing_cpu_offload` can be used to offload activations to CPU when using gradient checkpointing. This must be used together with `--gradient_checkpointing`.

### Attention / アテンション

Use `--sdpa`, `--flash_attn`, `--flash3`, `--sage_attn`, or `--xformers` to control the attention backend for Kandinsky5.

### Kandinsky5-specific Options / Kandinsky5固有オプション

- `text_encoder_auto`: Use device_map='auto' for Qwen TE to avoid OOM issues.
- `--i` / `--image`: Init image path for i2v-style seeding in `kandinsky5_generate_video.py`.

**NABLA attention (training):**

- `--use_nabla_attention`: Use NABLA attention.
- `--nabla_method`: NABLA binarization method (default `topcdf`).
- `--nabla_P`: CDF threshold (default `0.9`).
- `--nabla_wT`, `--nabla_wH`, `--nabla_wW`: STA window sizes (defaults `11`, `3`, `3`).
- `--nabla_add_sta` / `--no_nabla_add_sta`: Enable/disable STA prior when forcing NABLA.

**NABLA-compatible latent caching:**

- `kandinsky5_cache_latents.py --nabla_resize`: Resizes inputs to the next multiple of 128 before VAE encoding, which helps produce latents compatible with NABLA geometry constraints.

### Sample Generation During Training / 学習中のサンプル生成

Sample generation during training is supported. See [sampling during training](./sampling_during_training.md) for details.

<details>
<summary>日本語</summary>

上のコマンド例を使用して学習を開始してください（実際には一行で入力）。

日本語セクションの例（英語セクションと同じ内容）：

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    kandinsky5_train_network.py \
    --mixed_precision bf16 \
    --dataset_config path/to/dataset.toml \
    --task k5-pro-t2v-5s-sd \
    --dit path/to/kandinsky5pro_t2v_pretrain_5s.safetensors \
    --text_encoder_qwen Qwen/Qwen2.5-VL-7B-Instruct \
    --text_encoder_clip openai/clip-vit-large-patch14 \
    --vae path/to/vae/diffusion_pytorch_model.safetensors \
    --fp8_base --fp8_scaled \
    --sdpa \
    --gradient_checkpointing \
    --max_data_loader_n_workers 1 \
    --persistent_data_loader_workers \
    --learning_rate 1e-4 \
    --optimizer_type AdamW8Bit \
    --optimizer_args "weight_decay=0.001" "betas=(0.9,0.95)" \
    --max_grad_norm 1.0 \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_steps 100 \
    --network_module networks.lora_kandinsky \
    --network_dim 32 \
    --network_alpha 32 \
    --timestep_sampling shift \
    --discrete_flow_shift 5.0 \
    --output_dir path/to/output \
    --output_name k5_lora \
    --save_every_n_epochs 1 \
    --max_train_epochs 50
```

I2Vの学習を行う場合は、タスクとチェックポイントをI2V向けプリセットに変更してください（例: `k5-pro-i2v-5s-sd` と `kandinsky5pro_i2v_sft_5s.safetensors`）。`kandinsky5_cache_latents.py` でlatentをキャッシュする際に、最初のフレームlatent（`latents_image`）も保存されるため、I2V専用の追加フラグは不要です（I2Vタスクを選ぶだけで動作します）。画像モデル (T2I または I2I) の場合は、必ず Flux VAE を使用して適切なタスク (`k5-lite-t2i-hd` または `k5_lite_i2i_hd`) を設定し、前の手順で潜在変数をキャッシュするときに `--image_model_training` を `kandinsky5_cache_latents.py` に渡すようにしてください。

**最初と最後のフレーム条件付けについて**: 最初と最後のフレーム学習サポートは実験的なものです。このアプローチの有効性と妥当性はまだ十分にテストされていません。コミュニティからのフィードバックと結果をお待ちしています。

学習設定は実験的なものです。適切な学習率、学習ステップ数、タイムステップの分布などは、まだ完全には決まっていません。フィードバックをお待ちしています。

その他のオプションは `--help` で確認できます。

**主要オプション**

- `--task`: モデル設定（上記の利用可能なタスクを参照）
- `--dit`: DiTチェックポイントへのパス（タスクのデフォルトを上書き）
- `--vae`: VAEチェックポイントへのパス（タスクのデフォルトを上書き）
- `--network_module`: Kandinsky5 LoRAには `networks.lora_kandinsky` を使用

**メモリ最適化**

`--gradient_checkpointing`でgradient checkpointingを有効にし、VRAM使用量を削減できます。

`--fp8_base / --fp8_scaled`を指定すると、DiTがfp8で学習されます。消費メモリを大きく削減できますが、品質は低下する可能性があります。

VRAMが不足している場合は、`--blocks_to_swap` を使用して一部のブロックを CPU にオフロードしてください。エンコードプロンプトや TE のキャッシュでメモリオーバーフローが発生する場合は、`--text_encoder_auto` または `--text_encoder_cpu` を使用して、Qwen TE の一部またはすべてを CPU で実行してみてください。

`--gradient_checkpointing_cpu_offload`を指定すると、gradient checkpointing使用時にアクティベーションをCPUにオフロードします。`--gradient_checkpointing`と併用する必要があります。

**アテンション**

`--sdpa`/`--flash_attn`/`--flash3`/`--sage_attn`/`--xformers`はKandinsky5のattention backendに適用されます。

**Kandinsky5固有オプション**

- `text_encoder_auto`: OOM の問題を回避するには、Qwen TE に device_map='auto' を使用します。
- `--i` / `--image`: `kandinsky5_generate_video.py` でi2v風の初期画像（1フレーム目のシード）を指定します。

**NABLAアテンション（学習）**

- `--use_nabla_attention`: タスク設定に関係なくNABLAを強制します。
- `--nabla_method`: NABLAの二値化メソッド（デフォルト `topcdf`）。
- `--nabla_P`: CDFしきい値（デフォルト `0.9`）。
- `--nabla_wT`, `--nabla_wH`, `--nabla_wW`: STAウィンドウ（デフォルト `11`, `3`, `3`）。
- `--nabla_add_sta` / `--no_nabla_add_sta`: STA priorの有効/無効。

**NABLA互換latentキャッシュ**

- `kandinsky5_cache_latents.py --nabla_resize`: VAEエンコード前に入力を128の倍数へリサイズし、NABLAの幾何条件に合うlatentを生成しやすくします。

**学習中のサンプル生成**

学習中のサンプル生成がサポートされています。詳細は[学習中のサンプリング](./sampling_during_training.md)を参照してください。

</details>

## Inference / 推論

Generate videos using the following command:

```bash
python kandinsky5_generate_video.py \
    --task k5-pro-t2v-5s-sd \
    --dit path/to/kandinsky5pro_t2v_sft_5s.safetensors \
    --vae path/to/vae/diffusion_pytorch_model.safetensors \
    --text_encoder_qwen Qwen/Qwen2.5-VL-7B-Instruct \
    --text_encoder_auto \
    --text_encoder_clip openai/clip-vit-large-patch14 \
    --fp8_scaled \
    --dtype bfloat16 \
    --prompt "A cat walks on the grass, realistic style." \
    --negative_prompt "low quality, artifacts" \
    --video_length 121 \
    --steps 50 \
    --guidance_scale 5 \
    --scheduler_scale 10 \
    --seed 42 \
    --width 512 \
    --height 512 \
    --save_path path/to/output/folder/ \
    --lora_weight path/to/lora.safetensors \
    --lora_multiplier 1.0
```

### Options / オプション

- `--task`: Model configuration
- `--prompt`: Text prompt for generation
- `--negative_prompt`: Negative prompt (optional)
- `--save_path`: Output folder path
- `--width`, `--height`: Output resolution (defaults from task config). I2VI may override this if `--advanced_i2v` not specified!
- `--video_length`: Number of video frames to generate (exclusive of `--frames`)
- `--frames`: Number of latent frames to generate (exclusive of `--video_length`)
- `--steps`: Number of inference steps (defaults from task config)
- `--guidance_scale`: Guidance scale (defaults from task config)
- `--seed`: Random seed, can be an integer or a string! Yep, really!
- `--fp8_scaled`: Use fp8 scaled quantization to reduce size of DiT and save memory/VRAM
- `--fp8_fast`: Use fast fp8 math available on RTX 40X0 (Ada Lovelace) and potentially later GPUs to improve speed substantially for a small quality loss
- `--fp16_fast`: Use optimized fp16 math and fp16 accumulation available in PyTorch 2.7 or later to improve speed substantially. Quality loss is small for Video Pro but may be noticeable for Video Lite and Image!
- `--text_encoder_auto`: Auto split the text encoder between GPU and CPU. Use this if you OOM when encoding prompts!
- `--advanced_i2v`: Eases restrictions on size/shape for I2V/I2I modes and automatically scales input image to requested video size but pushing the model too far outside what it expects can cause issues so use smartly!
- `--blocks_to_swap`: Number of blocks to offload to CPU
- `--lora_weight`: Path(s) to LoRA weight file(s)
- `--lora_multiplier`: LoRA multiplier(s)
- `--optimized`: Overrides the default values of several command line args to provide an optimized but quality experience. Enables fp16_fast or fp8_fast depending on mode and hardware, fp8_scaled, sageattn and torch.compile. Requires SageAttention and Triton to be installed in addition to PyTorch 2.7.0 or higher!
- `--preview_latent_every`: If specified, enables previews (saved to output folder as latent_preview.mp4/png) of the current generation every N steps. By default uses latent2RGB (very fast, lower quality) but can optionally use `--preview_vae` to specify a [TinyAutoencoder](https://huggingface.co/Blyss/BlissfulModels/tree/main/taehv) for fast, high quality previews! Use taehv for Video Pro/Lite and taef1 for Image!

Additional tasks such as Lite and Image tasks are also available as well as various speed optimizations. For a complete list of available flags, please see `python kandinsky5_generate_video.py --help`.

<details>
<summary>日本語</summary>

上のコマンド例を使用して動画を生成します。

**オプション**

- `--task`: モデル設定
- `--prompt`: 生成用のテキストプロンプト
- `--negative_prompt`: ネガティブプロンプト（オプション）
- `--save_path`: 出力フォルダのパス
- `--width`, `--height`: 出力解像度（タスク設定からのデフォルト）。`--advanced_i2v` が指定されていない場合、I2VI はこれを上書きする可能性があります。
- `--video_length`: 生成するビデオフレーム数（`--frames` を除く）
- `--frames`: 生成する潜在フレーム数（`--video_length` を除く）
- `--steps`: 推論ステップ数（タスク設定からのデフォルト）
- `--guidance_scale`: ガイダンススケール（タスク設定からのデフォルト）
- `--seed`: ランダムシード。整数または文字列を指定できます。はい、本当にそうです！
- `--fp8_scaled`: fp8スケールの量子化を使用してDiTのサイズを縮小し、メモリ/VRAMを節約します
- `--fp8_fast`: RTX 40X0 (Ada Lovelace) およびそれ以降の GPU で利用可能な高速 fp8 演算を使用して、わずかな品質損失で速度を大幅に向上させます
- `--fp16_fast`: PyTorch 2.7 以降で利用可能な最適化された fp16 演算および fp16 累算を使用して、速度を大幅に向上させます。 Video Pro では品質の低下はわずかですが、Video Lite と Image では顕著になる可能性があります。
- `--text_encoder_auto`: テキスト エンコーダーを GPU と CPU の間で自動分割します。プロンプトをエンコードするときに OOM する場合は、これを使用してください。
- `--advanced_i2v`: I2V/I2I モードのサイズ/形状の制限を緩和し、入力画像を要求されたビデオ サイズに自動的にスケールしますが、モデルを期待値から大きく外しすぎると問題が発生する可能性があるため、賢く使用してください。
- `--blocks_to_swap`: CPUにオフロードするブロック数
- `--lora_weight`: LoRA重みファイルへのパス
- `--lora_multiplier`: LoRA係数
- `--optimized`: いくつかのコマンドライン引数のデフォルト値をオーバーライドし、最適化された高品質なエクスペリエンスを提供します。モードとハードウェアに応じて fp16_fast または fp8_fast、fp8_scaled、sageattn、torch.compile を有効にします。PyTorch 2.7.0 以降に加えて、SageAttention と Triton がインストールされている必要があります。
- `--preview_latent_every`: 指定すると、現在の世代のNステップごとのプレビュー（出力フォルダにlatent_preview.mp4/pngとして保存）が有効になります。デフォルトではlatent2RGB（非常に高速、低品質）を使用しますが、オプションで`--preview_vae`を使用して[TinyAutoencoder](https://huggingface.co/Blyss/BlissfulModels/tree/main/taehv)を指定し、高速で高品質のプレビューを実現できます。Video Pro/Liteの場合はtaehv、Imageの場合はtaef1を使用してください。

LiteタスクやImageタスクなどの追加タスクに加え、様々な速度最適化も利用可能です。利用可能なフラグの完全なリストについては、`python kandinsky5_generate_video.py --help` を参照してください。

</details>

## Dataset Configuration / データセット設定

Dataset configuration is the same as other architectures. See [dataset configuration](./dataset_config.md) for details.

<details>
<summary>日本語</summary>

データセット設定は他のアーキテクチャと同じです。詳細は[データセット設定](./dataset_config.md)を参照してください。

</details>
