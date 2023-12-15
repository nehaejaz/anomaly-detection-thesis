## Few-Shot-Anomaly-Detection-Thesis

**Author**: Neha Ejaz

**Keywords**: Anomaly Detection, Few-Shot Learning, Computer Vision, Machine Learning

## Description
This repository represnts my master's thesis, conducted at Ontario Tech University under the guidance of Dr. Faisal Qureshi, delves into the realm of visual anomaly detection within industrial settings. This research is specifically centered on leveraging the innovative few-shot technique. The aim is to enhance anomaly detection capabilities in industrial environments, contributing to the advancement of  quality control systems. By applying state-of-the-art methodologies, this thesis seeks to make a significant impact in the field of visual anomaly detection, addressing challenges unique to industries and fostering advancements in the intersection of computer vision and industrial applications

## Get Started

### 1. Clone the repository:
```
git clone https://github.com/nehaejaz/Few-Shot-Anomaly-Detection-Thesis.git
cd Few-Shot-Anomaly-Detection-Thesis
```

### 2. Create environment
Create a virtual envionment and install the necessary dependencies, using `pip` and the `requirements.txt` file:

```
conda create -n myenv python=3.9
conda activate myenv
pip install -r requirements.txt

```

### 3. Download Datasets
- [MVTec Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads)
- [MPDD Dataset](https://github.com/stepanje/MPDD)

### 4. Download Support Sets
Download the following support sets and move them in a folder named "support sets"
- MVTec: [MVTec Support Set](https://drive.google.com/file/d/1AZcc77cmDfkWA8f8cs-j-CUuFFQ7tPoK/view)
- MPDD: [MPDD Support Set](https://drive.google.com/drive/folders/12W0gmljbQU8f8MzO52EQJ_4fv41WZexX?usp=sharing)


### 5. Setup Config File
Create a configuration file (`config.yaml`) from the sample below:

```yaml
dataset:
  data_type: mpdd
  data_path: ./MPDD #path to data set dir
  obj: metal_plate
  shot: 2
  batch_size: 32
  img_size: 224
  input_channel: 3
  supp_set: .support_sets/MPDD/2/m_2_1.pt #path to support set dir
  include_maddern_transform: false #illumination transformation
  alpha: 0.48
  ilumination_data: false

model:
  backbone: convnext #[resnet_stn, resnet, convnext] name of the model
  coreset_sampling_ratio: 0.01
  num_neighbors: 1
  drop_path_rate: 0.7

project:
  seed: 668
  save_dir: results/m-convnext/ #path to dir where results should be stored

trainer:
  epochs: 50
  inferences: 1
  lr: 0.0001
  momentum: 0.9
  stn_mode: rotation_scale

```

### 6. Project Structure
After these steps your project structure should look like this:

  ```
  ./Few-Shot-Anomaly-Detection-Thesis
  ├── README.md
  ├── train.py                                  # training code
  ├── test.py                                   # testing code
  ├── config.yaml                               # config file
  ├── create_tensor.py                          # to create custome support set tensors 
  ├── requirments.txt                           # requirments file
  ├── MVTec                                     # MVTec dataset files
  │   ├── bottle
  │   ├── cable
  │   ├── ...                  
  │   └── zippper
  ├── MPDD                                     # MPDD dataset files
  │   ├── bracket_black
  │   ├── bracket_white
  │   ├── ...                  
  │   └── tube
  ├── support set                               # support dataset files
  │   ├── MVTec
  │       ├── 2
  │       ├── 4                 
  │       └── 8
  │   ├── MPDD
  │       ├── 2
  │       ├── 4                 
  │       └── 8                 
  ├── models                                    # models and backbones
  │   ├── stn.py 
  │   ├── convnext_stn.py 
  │   ├── hf_convnext.py 
  │   ├── hf_resnet.py  
  │   └── siamese.py
  ├── losses                                    # losses
  │   └── norm_loss.py  
  ├── datasets                                  # dataset                      
  │   └── mvtec.py
 files                  
  └── utils                                     # utils
      ├── utils.py
      └── funcs.py
  ```

### 7. Training
Run the following command to train the model:

```
python train.py --config path/to/config.yaml

```

### 8. Testing
Run the following command to test the model:


```
python test.py --config path/to/config.yaml --CKPT_name path/to/model_checkpoint.pth

```
## Custome Support Sets
You can create your own custome support set tensors by running create_tensor.py script
```
python create_tensor.py 
```
## Run TensorBorad
Open a terminal window in your root project directory run this command to see the loss and accuracy plots on tensor board

```
tensorboard --logdir=runs
```

Go to the URL it provides OR on windows:

```
http://localhost:6006/

```

## Acknowledgement
We borrow some codes from [SimSiam](https://github.com/facebookresearch/simsiam), [STN](https://github.com/YotYot/CalibrationNet/blob/2446a3bcb7ff4aa1e492adcde62a4b10a33635b4/models/configurable_stn_no_stereo.py), [PaDiM](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master) and [RegAd](https://github.com/MediaBrain-SJTU/RegAD)

## Contact
If you have any problem with this code, please feel free to contact **nehaejaz29@gmail.com**



