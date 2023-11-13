# FewShot Coreset based Anomaly Detection

## Description
This project presents a deep learning-based category-agnostic model capable of detecting anomalies using few support samples. The results have been tested on two benchmark datasets: MVTec and MPDD.

## Installation Instructions

### 1. Download Datasets
- [MVTec Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads)
- [MPDD Dataset](https://github.com/stepanje/MPDD)

### 2. Download Support Sets
- MVTec: [LINK]
- MPDD: [LINK]

### 3. Setup Config File
Here's a sample configuration file (`config.yaml`):

```yaml
dataset:
  data_type: mpdd
  data_path: ./MPDD
  obj: bracket_black
  shot: 2
  batch_size: 32
  img_size: 224
  input_channel: 3
  supp_set: ./mpdd_supp_set/2/b_b_2_1.pt

model:
  coreset_sampling_ratio: 0.05
  num_neighbors: 1

project:
  seed: 668
  save_dir: results/mpdd-bb-2/

trainer:
  epochs: 50
  inferences: 1
  lr: 0.0001
  momentum: 0.9
  stn_mode: rotation_scale
