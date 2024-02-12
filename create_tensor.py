import torch
import torchvision.transforms as transforms
import os
import cv2
from PIL import Image
from torchvision.utils import make_grid, save_image

# define the transformations to be applied to each image
transform = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# set the path to the folder containing the images
folder_path = '/home/nejaz/Few-Shot-Anomaly-Detection-Thesis/MVTec/bottle/train/good_20'

# create an empty list to store the tensors
tensor_list = []

    
# loop through the images in the folder
for img_name in os.listdir(folder_path):
    if img_name.endswith('.jpg') or img_name.endswith('.png'):
        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img)
        tensor_list.append(tensor.unsqueeze(0))  # add a batch dimension
        
# concatenate the tensors along the batch dimension
tensor_list = torch.cat(tensor_list, dim=0)

# split the concatenated tensor into a list of tensors with shape [2, 3, 224, 224]
tensor_list = tensor_list.split(2, dim=0)
torch.save(tensor_list, "22_10.pt")

# to save the support test images
output_dir = os.path.join("/home/roya/neha/RegAD/support_set/bottle/pattern")
if not os.path.exists(output_dir):
        os.makedirs(output_dir)
count = 0
for tensor in tensor_list:
    output_file = os.path.join(output_dir, f'{count}.png')
    grid = make_grid(tensor)
    save_image(grid, output_file)
    print(tensor.shape)
    count += 1

