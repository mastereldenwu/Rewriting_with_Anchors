# Code for "Sentence-Level Rewriting with Structural Anchors: Balancing Safety and Diversity in Semi-Synthetic Data Generation"

This repository contains the experimental code for the paper **"Sentence-Level Rewriting with Structural Anchors: Balancing Safety and Diversity in Semi-Synthetic Data Generation"**. It covers the full pipeline: data preparation, augmentation, model training, and evaluation.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── server.sh                  # llama-server launch script
├── nov_preprocess.py          # novel cleaning and chunking
├── mark_id.py                 # assign sequential ids to JSONL records
├── cut.py                     # sampling; train/test split
├── generate_with_id.py        # pure synthetic generation (vLLM)
├── toedit_with_id.py          # ToEdit semi-synthetic generation (vLLM)
├── double_paragraph_edit.py   # sentence-level augmentation (llama-server)
├── train_and_eva.py           # training + PPL / diversity evaluation
├── eva_with_rain.py           # full pipeline: training + multi-level evaluation
└── uni.py                     # unique N-gram extraction
```

## Experimental Environment

### Hardware

| Component | Configuration |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| CPU | AMD Ryzen 5 7500F (6 cores / 12 threads) |
| RAM | 32 GiB |
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB) |
| GPU Driver | 595.58.03 |

All experiments were conducted on a single consumer-grade GPU.

### Software

| Component | Version |
|---|---|
| Python | 3.13.11 |
| PyTorch | 2.9.1 (CUDA Runtime 12.8) |
| transformers | 4.57.6 |
| accelerate | 1.12.0 |
| vLLM | 0.15.0 |
| sentence-transformers | 5.4.1 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.0 |
| jieba | 0.42.1 |

The driver reports CUDA 13.2 while PyTorch is built against CUDA Runtime 12.8. The driver is backward-compatible; this is not a conflict.

## Data

The corpus consists of Chinese online literary texts with a mixed range of genres. The exact source is not disclosed due to privacy constraints. Cleaning and chunking are implemented in `nov_preprocess.py`.

Chunks are formed by accumulating sentences until the next punctuation mark after 100 characters, with a hard cut at 256 characters. Input files are JSONL, one JSON object per line:

```json
{"text": "..."}
```

## Pipeline

```
raw TXT
   │
   ├─ nov_preprocess.py        → clean, chunk, merge into JSONL
   ├─ mark_id.py               → assign sequential ids
   ├─ cut.py                   → sample + split into train / test
   │
   ├─ generate_with_id.py      → Pure group
   ├─ toedit_with_id.py        → ToEdit group
   └─ double_paragraph_edit.py → New group (sentence-level augmentation)
```

## Usage

### 1. Prepare the data

```bash
python nov_preprocess.py -i ./data/raw -o ./data/processed
python mark_id.py -i ./data/processed/all_novels_merged.jsonl.gz -o ./data/marked.jsonl.gz
python cut.py --input_file ./data/marked.jsonl.gz --num_samples 95000
```

`--num_samples 95000` is the total number of samples used for the ablation study; the main experiments use 5,000 paragraphs as drafts.

### 2. Launch the inference server (required for the New group)

```bash
bash server.sh   # starts llama-server on port 8080
```

The model path is configured via an environment variable; see `server.sh`.

### 3. Generate augmented data

```bash
python generate_with_id.py                        # Pure
python toedit_with_id.py                          # ToEdit
python double_paragraph_edit.py -s marked.jsonl   # New
```

### 4. Train and evaluate

```bash
python train_and_eva.py --real_file ... --output_dir ./exp_results
python eva_with_rain.py --real_file ... --output_dir ./exp_results_enhanced
```

### 5. Unique N-gram analysis

```bash
python uni.py
```

## Script-to-Paper Mapping

| Script | Corresponds to |
|---|---|
| `nov_preprocess.py` | Data cleaning, chunking, and merging |
| `cut.py` | Sampling and train/test split |
| `generate_with_id.py` | Pure baseline |
| `toedit_with_id.py` | ToEdit baseline |
| `double_paragraph_edit.py` | Proposed method |
| `train_and_eva.py` | PPL and diversity metrics |
| `eva_with_rain.py` | Data-level metrics |
| `uni.py` | Unique N-gram analysis |

## Installation

```bash
pip install -r requirements.txt
```

`vLLM` is only required by `generate_with_id.py` and `toedit_with_id.py`. `server.sh` requires a self-compiled [llama.cpp](https://github.com/ggerganov/llama.cpp); update the binary path and model path inside the script before running.
The New generation model is a community-quantized DeepSeek 14B model, loaded via llama-server in server.sh.

## Known Limitations

- The dataset is not publicly released due to privacy constraints.
- The experiments were run on a Blackwell (sm_120) GPU; some older CUDA libraries are incompatible.

## Citation

```bibtex
@misc{anonymous2026repo,
  title  = {Code for "Sentence-Level Rewriting with Structural Anchors"},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```
