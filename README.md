# ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding

*Jialiang Kang, Han Shu, Wenshuo Li, Yingjie Zhai, Xinghao Chen*


<a href="http://arxiv.org/abs/2509.15235"><img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red&logo=arxiv"></a>
<a href="https://huggingface.co/collections/JLKang/vispec-68cd460c5766dc65e908909f"><img src="https://img.shields.io/static/v1?label=HuggingFace&message=Collection&color=yellow&logo=huggingface"></a>

<p align="center">
  <img src="./figs/speedup_t0.png" alt="benchmark">
</p>

## Overview

Speculative decoding is a widely adopted technique for accelerating inference in large language models (LLMs), yet its application to vision-language models (VLMs) remains underexplored, with existing methods achieving only modest speedups ($<1.5\times$). This gap is increasingly significant as multimodal capabilities become central to large-scale models. We hypothesize that large VLMs can effectively filter redundant image information layer by layer without compromising textual comprehension, whereas smaller draft models struggle to do so. To address this, we introduce **Vision-Aware Speculative Decoding (ViSpec)**, a novel framework tailored for VLMs. ViSpec employs a lightweight vision adaptor module to compress image tokens into a compact representation, which is seamlessly integrated into the draft model's attention mechanism while preserving original image positional information. Additionally, we extract a global feature vector for each input image and augment all subsequent text tokens with this feature to enhance multimodal coherence. To overcome the scarcity of multimodal datasets with long assistant responses, we curate a specialized training dataset by repurposing existing datasets and generating extended outputs using the target VLM with modified prompts. Our training strategy mitigates the risk of the draft model exploiting direct access to the target model's hidden states, which could otherwise lead to shortcut learning when training solely on target model outputs. Extensive experiments validate ViSpec, achieving, to our knowledge, the first substantial speedup in VLM speculative decoding.

## Requirements

The code requires `python>=3.8` and `transformers==4.49`. You can install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Weights

| Base Model                                                                                    | ViSpec on Hugging Face                                                                                  |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [llava-hf/llava-v1.6-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf)   | [JLKang/ViSpec-llava-v1.6-vicuna-7b-hf](https://huggingface.co/JLKang/ViSpec-llava-v1.6-vicuna-7b-hf)   |
| [llava-hf/llava-v1.6-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf) | [JLKang/ViSpec-llava-v1.6-vicuna-13b-hf](https://huggingface.co/JLKang/ViSpec-llava-v1.6-vicuna-13b-hf) |
| [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)             | [JLKang/ViSpec-Qwen2.5-VL-3B-Instruct](https://huggingface.co/JLKang/ViSpec-Qwen2.5-VL-3B-Instruct)     |
| [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)             | [JLKang/ViSpec-Qwen2.5-VL-7B-Instruct](https://huggingface.co/JLKang/ViSpec-Qwen2.5-VL-7B-Instruct)     |
| [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)                   | [JLKang/ViSpec-llava-1.5-7b-hf](https://huggingface.co/JLKang/ViSpec-llava-1.5-7b-hf)                   |


## Usage

The workflow consists of three main stages: training data generation, model training, and evaluation.

We provide several pre-trained model checkpoints on Hugging Face (see the **Weights** section above). If you wish to use these, you can download them and skip the data generation and training sections, proceeding directly to **Stage 3: Evaluation**.

### 1. Training Data Generation

This process involves generating two distinct datasets for the two training stages.

#### 1.1. Generating Text-Only Data for Initial Training

First, generate the text-only data. This dataset will be used in the first stage of model training.

```bash
python -m vispec.ge_data.allocation_{llava,qwen}_shargpt \
  --outdir=<path_to_text_data_folder> \
  --start=0 \
  --end=67999 \
  --model={Qwen/Qwen2.5-VL-3B-Instruct,Qwen/Qwen2.5-VL-7B-Instruct,llava-hf/llava-v1.6-vicuna-7b-hf,llava-hf/llava-v1.6-vicuna-13b-hf}
```

#### 1.2. Generating Multimodal Data for ViSpec Training

Next, generate the multimodal data. This dataset is used in the second stage to train the ViSpec module.

```bash
python -m vispec.ge_data.allocation_{llava,qwen}_pretrain_gen \
  --outdir=<path_to_multimodal_data_folder> \
  --start=0 \
  --end=67999 \
  --model={Qwen/Qwen2.5-VL-3B-Instruct,Qwen/Qwen2.5-VL-7B-Instruct,llava-hf/llava-v1.6-vicuna-7b-hf,llava-hf/llava-v1.6-vicuna-13b-hf}
```

**Parameters**:

  - `--outdir`: The directory where the generated data will be stored.
  - `--start`/`--end`: The index range of the data to be generated.
  - `--model`: The base vision-language model to use for data generation.

### 2. Model Training

Model training is a two-stage process.

#### 2.1: Initial Training

This stage performs initial training on the draft model using the text-only data generated in Step 1.1.

```bash
accelerate launch --multi_gpu \
  -m --mixed_precision=bf16 \
  vispec.train.main \
  --cpdir=<path_to_output_checkpoints_folder> \
  --basepath={Qwen/Qwen2.5-VL-3B-Instruct,Qwen/Qwen2.5-VL-7B-Instruct,llava-hf/llava-v1.6-vicuna-7b-hf,llava-hf/llava-v1.6-vicuna-13b-hf} \
  --begin-epoch=0 \
  --bs=1 \
  --configpath=vispec/train/<yourconfig.json> \
  --lr=3e-6 \
  --max-len=4096 \
  --num-workers=8 \
  --tmpdir=<path_to_text_data_folder>
```

#### 2.2: Training with ViSpec

This stage continues the training using our proposed ViSpec method and the multimodal data from Step 1.2. It loads the checkpoint from Stage 2.1 to initialize the model weights.

```bash
accelerate launch --multi_gpu \
  -m --mixed_precision=bf16 \
  vispec.train.main_mtp \
  --cpdir=<path_to_output_checkpoints_folder> \
  --basepath={Qwen/Qwen2.5-VL-3B-Instruct,Qwen/Qwen2.5-VL-7B-Instruct,llava-hf/llava-v1.6-vicuna-7b-hf,llava-hf/llava-v1.6-vicuna-13b-hf} \
  --begin-epoch=0 \
  --bs=1 \
  --configpath=vispec/train/<yourconfig.json> \
  --loadpath=<path_to_stage1_checkpoint>/state_20/model.safetensors \
  --lr=3e-6 \
  --max-len=4096 \
  --mtp-steps=1 \
  --num-q=2 \
  --num-workers=8 \
  --tmpdir=<path_to_multimodal_data_folder> \
  --use-ours=True
```

**Key Parameters**:

  - `--cpdir`: The output directory for training checkpoints.
  - `--tmpdir`: The input directory containing the appropriate training data for each stage.
  - `--configpath`: Path to the model configuration file.
  - `--loadpath`: Path to load the pre-trained weights from Stage 2.1.
  - `--lr`: Learning rate (e.g., `3e-6`).
  - `--num-q`: Number of query vectors (e.g., `2`).
  - `--use-ours`: Flag to enable ViSpec.


### 3. Evaluation

Evaluate the inference speed of the model using both standard autoregressive decoding (baseline) and speculative decoding.

#### Baseline Speed Evaluation

```bash
python -m vispec.evaluation.gen_baseline_answer_xxx \
  --base-model-path={Qwen/Qwen2.5-VL-3B-Instruct,Qwen/Qwen2.5-VL-7B-Instruct,llava-hf/llava-v1.6-vicuna-7b-hf,llava-hf/llava-v1.6-vicuna-13b-hf} \
  --model-id test \
  --bench-name=<path_to_baseline_results_folder> \
  --spec-model-path=<path_to_your_model_directory> \
  --temperature=<value>
```

**Parameters**:

  - `--bench-name`: The output directory for evaluation results.
  - `--spec-model-path`: Path to the directory containing the ViSpec model checkpoint. This can be a model you trained or one downloaded from Hugging Face.
  - `--temperature`: Sampling temperature (e.g., `0.0` for greedy, `1.0` for stochastic).

#### Speculative Decoding Speed Evaluation

```bash
python -m vispec.evaluation.gen_spec_answer_xxx \
  --base-model-path={Qwen/Qwen2.5-VL-3B-Instruct,Qwen/Qwen2.5-VL-7B-Instruct,llava-hf/llava-v1.6-vicuna-7b-hf,llava-hf/llava-v1.6-vicuna-13b-hf} \
  --model-id test \
  --bench-name=<path_to_spec_results_folder> \
  --spec-model-path=<path_to_your_model_directory> \
  --num-q=2 \
  --depth=<value> \
  --top-k=<value> \
  --total-token=<value> \
  --use-ours=True \
  --temperature=<value>
```

**Specific Parameters**:

  - `--depth`: The depth for speculative decoding.
  - `--top-k`: The width for candidate token selection.
  - `--total-token`: The total number of tokens to generate for the evaluation.
  - `--num-q`: Number of query vectors; must be consistent with the training configuration (e.g., `2`).

## Evaluation Results

Speedup ratios and average acceptance lengths $\tau$ for different methods. Speedup ratios are computed based on the average time required to generate each token.

| Model                   | Method     |    SQA     |             |   MM-Vet   |             |  TextVQA   |             |    MME     |             | COCO Caps  |             |   VizWiz   |             |    GQA     |             | SEED-Bench |             |    Avg.    |             |
| :---------------------- | :--------- | :--------: | :---------: | :--------: | :---------: | :--------: | :---------: | :--------: | :---------: | :--------: | :---------: | :--------: | :---------: | :--------: | :---------: | :--------: | :---------: | :--------: | :---------: |
|                         |            | **$\tau$** | **Speedup** | **$\tau$** | **Speedup** | **$\tau$** | **Speedup** | **$\tau$** | **Speedup** | **$\tau$** | **Speedup** | **$\tau$** | **Speedup** | **$\tau$** | **Speedup** | **$\tau$** | **Speedup** | **$\tau$** | **Speedup** |
| **LLaVA-1.6 7B (T=0)**  | Medusa     |    0.72    |    1.41x    |    0.73    |    1.42x    |    0.77    |    1.46x    |    0.70    |    1.41x    |    0.66    |    1.61x    |    0.76    |    1.38x    |    0.73    |    1.29x    |    0.72    |    1.38x    |    0.72    |    1.42x    |
|                         | EAGLE-2    |    2.48    |    2.14x    |    0.63    |    1.48x    |    0.63    |    1.25x    |    1.25    |    1.68x    |    1.24    |    1.80x    |    1.15    |    1.40x    |    1.74    |    1.64x    |    1.40    |    1.59x    |    1.31    |    1.62x    |
|                         | **ViSpec** |  **2.86**  |  **2.37x**  |  **2.83**  |  **2.52x**  |  **2.95**  |  **2.90x**  |  **2.84**  |  **2.55x**  |  **3.30**  |  **3.22x**  |  **3.16**  |  **2.67x**  |  **2.88**  |  **2.22x**  |  **3.03**  |  **2.22x**  |  **2.98**  |  **2.58x**  |
| **LLaVA-1.6 13B (T=0)** | Medusa     |    0.84    |    1.61x    |    0.80    |    1.47x    |    0.89    |    1.51x    |    0.79    |    1.47x    |    0.75    |    1.48x    |    0.81    |    1.45x    |    0.85    |    1.45x    |    0.82    |    1.40x    |    0.82    |    1.48x    |
|                         | EAGLE-2    |    2.02    |    2.12x    |    1.64    |    1.59x    |    1.71    |    1.91x    |    1.81    |    1.85x    |    1.83    |    2.01x    |    1.98    |    1.90x    |    2.10    |    1.82x    |    2.03    |    1.66x    |    1.89    |    1.86x    |
|                         | **ViSpec** |  **2.76**  |  **2.57x**  |  **2.73**  |  **2.34x**  |  **2.78**  |  **2.43x**  |  **2.78**  |  **2.36x**  |  **3.18**  |  **2.82x**  |  **2.93**  |  **2.26x**  |  **2.95**  |  **2.12x**  |  **3.04**  |  **2.16x**  |  **2.89**  |  **2.38x**  |
| **Qwen2.5-VL 3B (T=0)** | Medusa     |    0.57    |    1.07x    |    0.60    |    1.12x    |    0.66    |    1.08x    |    0.59    |    1.12x    |    0.62    |    1.21x    |    0.60    |    1.16x    |    0.65    |    1.21x    |    0.61    |    1.15x    |    0.61    |    1.14x    |
|                         | EAGLE-2    |    1.18    |    1.41x    |    1.03    |    1.30x    |    0.98    |    1.26x    |    1.07    |    1.38x    |    1.40    |    1.60x    |    1.11    |    1.32x    |    1.39    |    1.52x    |    1.11    |    1.32x    |    1.16    |    1.39x    |
|                         | **ViSpec** |  **1.99**  |  **1.87x**  |  **2.13**  |  **1.81x**  |  **2.15**  |  **1.85x**  |  **1.96**  |  **1.82x**  |  **2.37**  |  **2.15x**  |  **2.22**  |  **1.71x**  |  **2.28**  |  **2.01x**  |  **2.37**  |  **1.78x**  |  **2.19**  |  **1.87x**  |
| **Qwen2.5-VL 7B (T=0)** | Medusa     |    0.60    |    1.13x    |    0.59    |    1.06x    |    0.58    |    1.05x    |    0.59    |    1.19x    |    0.61    |    1.11x    |    0.59    |    1.09x    |    0.64    |    1.19x    |    0.62    |    1.05x    |    0.60    |    1.11x    |
|                         | EAGLE-2    |    1.40    |    1.49x    |    1.19    |    1.36x    |    1.14    |    1.23x    |    1.29    |    1.54x    |    1.46    |    1.50x    |    1.27    |    1.20x    |    1.53    |    1.54x    |    1.42    |    1.32x    |    1.34    |    1.40x    |
|                         | **ViSpec** |  **2.19**  |  **1.84x**  |  **2.16**  |  **1.74x**  |  **2.21**  |  **1.72x**  |  **2.15**  |  **1.96x**  |  **2.27**  |  **1.99x**  |  **2.31**  |  **1.71x**  |  **2.30**  |  **1.91x**  |  **2.34**  |  **1.55x**  |  **2.24**  |  **1.80x**  |
| **LLaVA-1.6 7B (T=1)**  | Medusa     |    0.58    |    1.36x    |    0.58    |    1.37x    |    0.57    |    1.32x    |    0.56    |    1.35x    |    0.58    |    1.67x    |    0.57    |    1.29x    |    0.60    |    1.19x    |    0.59    |    1.32x    |    0.58    |    1.36x    |
|                         | EAGLE-2    |    1.78    |    2.17x    |    0.51    |    1.34x    |    0.41    |    1.11x    |    1.02    |    1.53x    |    1.03    |    1.78x    |    0.77    |    1.32x    |    1.33    |    1.47x    |    0.98    |    1.57x    |    0.98    |    1.54x    |
|                         | **ViSpec** |  **2.06**  |  **2.20x**  |  **1.94**  |  **1.99x**  |  **1.78**  |  **1.93x**  |  **1.96**  |  **1.98x**  |  **2.36**  |  **3.05x**  |  **2.32**  |  **2.21x**  |  **2.11**  |  **1.83x**  |  **2.16**  |  **1.94x**  |  **2.09**  |  **2.14x**  |
| **LLaVA-1.6 13B (T=1)** | Medusa     |    0.68    |    1.41x    |    0.67    |    1.44x    |    0.66    |    1.42x    |    0.66    |    1.40x    |    0.67    |    1.40x    |    0.64    |    1.37x    |    0.70    |    1.37x    |    0.68    |    1.37x    |    0.67    |    1.40x    |
|                         | EAGLE-2    |    1.51    |    1.98x    |    1.29    |    1.73x    |    1.26    |    1.72x    |    1.45    |    1.78x    |    1.54    |    1.83x    |    1.46    |    1.72x    |    1.64    |    1.73x    |    1.60    |    1.79x    |    1.47    |    1.79x    |
|                         | **ViSpec** |  **2.02**  |  **2.25x**  |  **1.98**  |  **2.15x**  |  **1.90**  |  **2.08x**  |  **2.07**  |  **2.08x**  |  **2.43**  |  **2.39x**  |  **2.04**  |  **2.01x**  |  **2.19**  |  **2.03x**  |  **2.22**  |  **2.07x**  |  **2.11**  |  **2.13x**  |
| **Qwen2.5-VL 3B (T=1)** | Medusa     |    0.52    |    1.02x    |    0.48    |    1.02x    |    0.46    |    0.99x    |    0.46    |    1.02x    |    0.51    |    1.03x    |    0.46    |    0.99x    |    0.55    |    1.13x    |    0.49    |    1.03x    |    0.49    |    1.03x    |
|                         | EAGLE-2    |    0.92    |    1.25x    |    0.70    |    1.19x    |    0.70    |    1.06x    |    0.84    |    1.26x    |    0.97    |    1.28x    |    0.84    |    1.19x    |    1.02    |    1.31x    |    0.86    |    1.16x    |    0.86    |    1.21x    |
|                         | **ViSpec** |  **1.49**  |  **1.49x**  |  **1.23**  |  **1.39x**  |  **1.32**  |  **1.38x**  |  **1.45**  |  **1.58x**  |  **1.42**  |  **1.50x**  |  **1.39**  |  **1.43x**  |  **1.49**  |  **1.59x**  |  **1.55**  |  **1.42x**  |  **1.42**  |  **1.47x**  |
| **Qwen2.5-VL 7B (T=1)** | Medusa     |    0.56    |    1.05x    |    0.51    |    0.95x    |    0.49    |    0.96x    |    0.51    |    1.02x    |    0.52    |    1.00x    |    0.50    |    1.02x    |    0.53    |    1.02x    |    0.53    |    1.02x    |    0.52    |    1.01x    |
|                         | EAGLE-2    |    1.19    |    1.52x    |    0.92    |    1.19x    |    0.88    |    1.08x    |    1.00    |    1.23x    |    1.08    |    1.22x    |    0.94    |    1.13x    |    1.11    |    1.32x    |    1.04    |    1.19x    |    1.02    |    1.18x    |
|                         | **ViSpec** |  **1.82**  |  **1.62x**  |  **1.57**  |  **1.47x**  |  **1.51**  |  **1.37x**  |  **1.61**  |  **1.49x**  |  **1.63**  |  **1.50x**  |  **1.88**  |  **1.53x**  |  **1.61**  |  **1.56x**  |  **1.70**  |  **1.38x**  |  **1.66**  |  **1.49x**  |

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{vispec,
  title={ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding},
  author={Kang, Jialiang and Shu, Han and Li, Wenshuo and Zhai, Yingjie and Chen, Xinghao},
  booktitle={Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

## License

This project is licensed under <a rel="license" href="LICENSE"> Apache License 2.0</a>. Redistribution and use should follow this license.

## Acknowledgements

This work is supported by Huawei Noah's Ark Lab. We would like to acknowledge the foundational work of previous projects that inspired our approach, especially [EAGLE](https://github.com/SafeAILab/EAGLE) and [Medusa](https://github.com/FasterDecoding/Medusa). We also thank the anonymous NeurIPS reviewers for their insightful comments and valuable feedback.
