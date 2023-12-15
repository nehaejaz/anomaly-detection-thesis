## Few-Shot-Anomaly-Detection-Thesis

Author: Neha Ejaz
Keywords: Anomaly Detection, Few-Shot Learning, Computer Vision, Machine Learning


## Description
This is my Masters thesis which is based on visual anomaly detection focused on Industries using few-shot technique at Ontario Tech university under the supervision of Dr.Faisal Qureshi.

## Get Started

### 1. Clone the repository:
```
git clone https://github.com/nehaejaz/Few-Shot-Anomaly-Detection-Thesis.git
cd Few-Shot-Anomaly-Detection-Thesis
```

### 2. Create environment
Create a virtual envionment and install the necessary dependencies, using `pip` and the `requirements.txt` file:

```
pip install -r requirements.txt
```

### 3. Download Datasets
- [MVTec Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads)
- [MPDD Dataset](https://github.com/stepanje/MPDD)

### 4. Download Support Sets
- MVTec: [MVTec Support Set](https://drive.google.com/file/d/1AZcc77cmDfkWA8f8cs-j-CUuFFQ7tPoK/view)
- MPDD: [LINK]

### 3. Setup Config File
Here's a sample configuration file (`config.yaml`):

```yaml
dataset:
  data_type: mpdd
  data_path: ./MPDD
  obj: metal_plate
  shot: 2
  batch_size: 32
  img_size: 224
  input_channel: 3
  supp_set: ./mpdd_supp_set/2/m_2_1.pt
  include_maddern_transform: false
  alpha: 0.48
  ilumination_data: false

model:
  backbone: convnext #[resnet_stn, resnet, convnext, convnext_stn]
  coreset_sampling_ratio: 0.01
  num_neighbors: 1
  drop_path_rate: 0.7

project:
  seed: 668
  save_dir: results/m-convnext/

trainer:
  epochs: 50
  inferences: 1
  lr: 0.0001
  momentum: 0.9
  stn_mode: rotation_scale

```

### 4. Training
Run the following command to train the model:

```
python train.py --config path/to/config.yaml

```

### 4. Testing
Run the following command to test the model:


```
python test.py --config path/to/config.yaml --CKPT_name path/to/model_checkpoint.pth

```

## Results
Results of few-shot anomaly detection and localization with k=2:


