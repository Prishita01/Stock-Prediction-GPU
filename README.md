# StockNet — GPU-Accelerated Stock Price Prediction on HPC

> 72.6× GPU speedup over CPU. 97.5% directional accuracy. 
> Trained on S&P 500 data using LSTM, CNN-1D, and CNN-2D with 
> PyTorch DDP on CWRU Markov HPC cluster.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.1.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What This Does

StockNet is a parallelized deep learning framework for stock price prediction using S&P 500 historical data. It benchmarks three neural architectures 
(LSTM, CNN-1D, CNN-2D) across CPU, single-GPU, and multi-GPU configurations on the CWRU Markov HPC cluster — quantifying the real-world benefits of GPU acceleration and distributed training with PyTorch DDP.

**Core finding:** GPU acceleration delivers up to **72.6× speedup** over CPU-only training with zero accuracy loss. Multi-GPU DDP achieves near-linear scaling (101.9% efficiency for CNN-2D) while maintaining high inference throughput suitable for real-time stock screening.

---

## Results

### Training Time & Prediction Quality

| Model | Device | GPUs | Time (s) | R² | Directional Acc. |
|---|---|---|---|---|---|
| LSTM | CPU | 0 | 5,515s | 0.9774 | 97.5% |
| LSTM | GPU | 1 | **76s** | 0.9774 | 97.5% |
| LSTM | GPU | 2 | **36.7s** | 0.9858 | **98.7%** |
| CNN-1D | CPU | 0 | 599s | 0.9767 | 62.4% |
| CNN-1D | GPU | 1 | **41s** | 0.9767 | 62.4% |
| CNN-1D | GPU | 2 | **23.6s** | 0.8967 | 62.7% |
| CNN-2D | CPU | 0 | 3,680s | 0.9702 | 62.4% |
| CNN-2D | GPU | 1 | **153s** | 0.9702 | 62.4% |
| CNN-2D | GPU | 2 | **75.1s** | 0.8522 | 62.7% |

### GPU Speedup Over CPU

| Model | CPU→GPU Speedup | 1GPU→2GPU Speedup | Parallel Efficiency |
|---|---|---|---|
| LSTM | **72.6×** | 2.07× | 103.5% |
| CNN-1D | 14.6× | 1.74× | 86.8% |
| CNN-2D | 24.1× | **2.04×** | **101.9%** |

### Inference Throughput (samples/sec)

| Model | Throughput |
|---|---|
| CNN-1D | **41,336 samples/sec** |
| LSTM | 17,110 samples/sec |
| CNN-2D | 9,866 samples/sec |

---

## Models

- **LSTM** — Stacked 2-layer LSTM with hidden size 64, dropout 0.2; best accuracy (R²=0.977, 97.5% directional accuracy)
- **CNN-1D** — Temporal convolutional filters; fastest inference (41K samples/sec); best for real-time screening
- **CNN-2D** — Cross-feature convolutional kernels modeling joint OHLCV patterns; best multi-GPU scaling efficiency (101.9%)

All models trained on S&P 500 (500 stocks, 1000 trading days each, 6 features: Open, High, Low, Close, Volume, Adj. Close)

---

## Parallelization Strategy

- **Single-GPU:** PyTorch `cuda` device, SLURM `--gres=gpu:1`, 32GB RAM, 8 CPU cores
- **Multi-GPU:** PyTorch **Distributed Data Parallel (DDP)** via `torch.multiprocessing.spawn`, NCCL backend, gradient all-reduce across devices, SLURM `--gres=gpu:2`, 64GB RAM
- **Cluster:** CWRU Markov HPC, PyTorch 2.1.2 + CUDA 12.1.1, Intel Xeon Gold 6230 CPU baselines

---

## Quickstart

```bash
git clone https://github.com/Prishita01/Stock-Prediction-GPU.git
cd Stock-Prediction-GPU
git checkout feature/gpu-training
pip install -r utils/requirements.txt
```

**Single-GPU training:**
```bash
python main.py --mode train --model cnn --cnn_type 1d \
  --epochs 1000 --batch_size 64 --device cuda
```

**Multi-GPU distributed training:**
```bash
python train_distributed.py --model cnn --cnn_type 1d \
  --epochs 1000 --batch_size 64 --world_size 2
```

**On HPC cluster:**
```bash
sbatch slurm_scripts/train_cnn1d_1gpu.slurm
sbatch slurm_scripts/train_cnn1d_2gpu.slurm
```

---

## Tech Stack

Python · PyTorch 2.1.2 · CUDA 12.1.1 · PyTorch DDP · 
SLURM · CWRU Markov HPC · NumPy · Pandas

---

## Note

> This repository contains my individual GPU training and distributed computing contribution to a collaborative stock prediction project.
> Full collaborative project: [ZhengBenjamin/Stock-Prediction-Network](https://github.com/ZhengBenjamin/Stock-Prediction-Network/tree/feature/gpu-training)

---

## Author

**Prishita Ghanathe Krishna**

