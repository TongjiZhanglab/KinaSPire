# KinaSPire

![label1](https://img.shields.io/badge/license-MIT-green)

**| [Overview](#overview) | [Installation](#installation) | [Quick Start](#quick-start) | [Reproduction](#reproduction) |**

---

## Overview

**KinaSPire** is a prior-constrained deep-learning framework for sample-specific kinase activity inference from phosphoproteomics data under incomplete substrate knowledge. It uses a kinase–substrate prior to constrain an autoencoder, then extracts per-sample kinase activity from the trained encoder.

![Figure 1](https://raw.githubusercontent.com/Yizhi-Zhang/MarkdownPicture/master/KinaSPire/model.png)

The repository includes the core package, configuration files, a toy example, and resources for reproducing the benchmark analyses:

```text
KinaSPire/
├── kinaspire/          # Python package and CLI
├── configs/            # YAML configs
├── examples/           # toy dataset and notebook
├── scripts/            # thin wrappers around the CLI
├── reproduction/       # perturbation- and tumor-based benchmarks
├── pyproject.toml
└── README.md
```

A typical end-to-end workflow consists of three steps:

1. **Preprocess** a phosphosite matrix and a kinase–substrate prior.
2. **Train** a KinaSPire model from a YAML config.
3. **Infer** kinase activity from a saved model.

---

## Installation

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or a Python 3.10+ virtual environment
- [PyTorch](https://pytorch.org) installed separately to match your CUDA or CPU setup

### Install the package

From the `KinaSPire/` directory:

```bash
pip install -e .
```

This installs the `kinaspire` command-line tool. Check the install with:

```bash
kinaspire -v
kinaspire -h
```

> **Note:** PyTorch is not listed in `requirements.txt`. Install it first according to your CUDA version. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the appropriate command. For example:
> ```bash
> pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Quick Start

KinaSPire can be run in two ways: through the provided Jupyter notebook for a complete toy example, or through the command-line interface for step-by-step execution.

### Jupyter notebook

A complete toy example is provided in `examples/toy_example.ipynb`, together with the accompanying toy dataset in `examples/example_data/`:

```text
examples/
├── toy_example.ipynb
└── example_data/
    ├── toy_phosphoproteome.feather   # phosphosites × samples (20 phosphosites, 600 samples)
    ├── toy_prior.tsv                 # phosphosites × kinases (K1, K2, K3)
    └── toy_metadata.tsv              # sample metadata and simulated true kinase activity
```

- **`toy_phosphoproteome.feather`**

    Matrix with phosphosites as rows and samples as columns. `DataProcessor` expects this orientation and writes `data.parquet` as samples × phosphosites.

- **`toy_prior.tsv`**

    Tab-separated kinase–substrate prior with phosphosites as rows and kinases as columns. Known relationships are non-zero; others are 0. After preprocessing, `prior.parquet` is kinases × phosphosites.

- **`toy_metadata.tsv`**

    Simulated ground-truth kinase activity (`K1`, `K2`, `K3`) and binary kinase-state labels used only in the notebook visualization step.

### Command line

The same workflow can be run from the command line using `kinaspire <command>`, `kinaspire-<command>`, or the wrapper scripts in `scripts/`.

#### 1. Preprocess

Load a phosphosite matrix (phosphosites × samples, `.feather`) and a kinase–substrate prior (phosphosites × kinases, `.tsv`). `RobustMinScaler` is applied per phosphosite, the prior is transposed to kinases × phosphosites, kinases with no substrates are removed, and the processed tables are written to `<output>/processed_data/`.

```bash
kinaspire preprocess \
  --exp examples/example_data/toy_phosphoproteome.feather \
  --ksr examples/example_data/toy_prior.tsv \
  --output examples/outputs
```

Equivalent:

```bash
python scripts/preprocess_data.py \
  --exp examples/example_data/toy_phosphoproteome.feather \
  --ksr examples/example_data/toy_prior.tsv \
  --output examples/outputs
```

For a full list of arguments, run `kinaspire preprocess -h`.

#### 2. Train

Train KinaSPire from the preprocessed tables using a YAML config. Model checkpoints, `performance.csv`, and the resolved `config.yaml` are written to `--output`.

```bash
kinaspire train \
  --data examples/outputs/processed_data/data.parquet \
  --prior examples/outputs/processed_data/prior.parquet \
  --config configs/toy.yaml \
  --output examples/outputs/toy_example
```

Configs live in `configs/`:

| File | Use |
| --- | --- |
| `configs/toy.yaml` | toy example |
| `configs/perturbation.yaml` | perturbation experiments |
| `configs/pdc/<COHORT>.yaml` | tumor cohorts (BIC, CA, CCRCC, GBM, HCC, HNSCC, LSCC, LUAD, OSC, PDA, UCEC) |

Set `runtime.device` in the YAML to `cpu` or a CUDA device.

For a full list of arguments, run `kinaspire train -h`.

#### 3. Infer kinase activity

Use a saved `.pth` checkpoint with `--model`, or pass `--performance` to select the best model by R<sup>2</sup>. The command writes a kinase × sample activity table.

```bash
kinaspire infer \
  --data examples/outputs/processed_data/data.parquet \
  --prior examples/outputs/processed_data/prior.parquet \
  --model examples/outputs/toy_example/model/cv0_wd1e-10_refit3.pth \
  --output examples/outputs/toy_example/kinaspire_act.csv
```

For a full list of arguments, run `kinaspire infer -h`.

---

## Reproduction

`reproduction/` contains scripts and input tables for reproducing the paper benchmarks from precomputed kinase activity matrices.

### Perturbation-based benchmark

Compares KinaSPire, z-score, mean, PTM-SEA, PhosX and KSTAR across kinase-perturbation experiments. The benchmark can be reproduced using the provided script:

```bash
python reproduction/perturbation_based_benchmark/scripts/run_benchmark.py
```

The required activity matrices, metadata, and benchmark scripts, and generated results are organized as follows:

```text
reproduction/perturbation_based_benchmark/
├── activity/            # one CSV per method (kinases × experiments)
├── metadata/
│   ├── meta.csv
│   └── curated_sub_count.csv
├── scripts/
│   ├── run_benchmark.py
│   └── utils.py
└── results/             # written by the script
    ├── benchmark_all_result.pkl
    └── benchmark_all_result.summary.csv
```

### Tumor-based benchmark

Compares the same methods across 11 cancer types using activating phosphosite labels. The benchmark can be reproduced using:

```bash
python reproduction/tumor_based_benchmark/scripts/run_benchmark.py
```

The corresponding activity matrices, metadata, benchmark scripts, and generated results are organized as follows:

```text
reproduction/tumor_based_benchmark/
├── activity/            # one subdirectory per method, one table per cohort
├── metadata/
│   ├── activating_site_labels/
│   ├── cohort_sparse_list.pkl
│   └── tumor_sample_case_idmap.csv
├── scripts/
│   ├── run_benchmark.py
│   └── utils.py
└── results/             # written by the script
    └── benchmark_all_result.csv
```
