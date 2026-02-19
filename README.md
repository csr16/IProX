# IProX

## Project Introduction

IProX (Influence-Preserving Proxies) is a two-stage framework for scalable gradient-based data selection in LLM fine-tuning.

Gradient-based selectors (for example, TracIn and influence-function variants) can improve supervised fine-tuning data quality, but they are expensive on large models. IProX addresses this by constructing a smaller proxy directly from a target LLM while preserving influence-related behavior.
This repository currently uses a TracIn-based implementation for gradient-based influence estimation.

The pipeline has two stages:

1. Influence-Preserving SVD (IPSVD): compresses target linear layers into low-rank factors with influence-aware initialization.
2. Gradient Alignment: aligns proxy gradients with target gradients in factor space, and anchors proxy logits for stability.

This repository provides code for proxy initialization and training with gradient alignment.

## Environment Setup and Training

### 1. Create and activate an environment

```bash
cd IProX
conda create -y -n Iprox python=3.10
conda activate Iprox
```

### 2. Install dependencies

Install PyTorch first, then install project dependencies:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 3. Run training

Example command:

```bash
python init_iprox.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --train_files data/train/processed/dolly/dolly_data.jsonl \
  --sparsity 0.5 \
  --init_method IPSVD \
  --epochs 5 \
  --batch_size 4 \
  --output_dir ../models
```

## Directory Acquisition

### 1. Repository structure

```text
IProX/
├── init_iprox.py
├── requirements.txt
├── data/
│   └── train/processed/dolly/dolly_data.jsonl
└── utils/
    ├── get_training_dataset.py
    ├── init_with_ipsvd.py
    ├── grad_align.py
    └── util.py
```

### 2. Expected data location

The default training file path is:

```text
data/train/processed/dolly/dolly_data.jsonl
```

If you use your own data, place it anywhere accessible and pass it with `--train_files`.

## Citation Format

If you use this codebase, please cite:

```bibtex
@inproceedings{chen2026influencepreserving,
  title={Influence-Preserving Proxies for Gradient-Based Data Selection in {LLM} FineTuning},
  author={Sirui Chen and Yunzhe Qi and Mengting Ai and Yifan Sun and Ruizhong Qiu and Jiaru Zou and Jingrui He},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=PDNpRLxDlI}
}
```
