import torch
import torchvision.transforms as transforms
import os
import cv2
from PIL import Image
from torchvision.utils import make_grid, save_image
import streamlit as st
from io import StringIO

# define the transformations to be applied to each image
transform = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def check_supp_set_exit(support_set_new_name):
    support_set_folder = "./support_sets"
    for supp_set_name in os.listdir(support_set_folder):
        supp_set_name = os.path.splitext(supp_set_name)[0]
        if supp_set_name == support_set_new_name:
            st.write("Name already exit")

# create an empty list to store the tensors
tensor_list = []
support_set_name = st.text_input('Support Set Name')
check_supp_set_exit(support_set_name)
        
uploaded_files = st.file_uploader(f"Choose images for {support_set_name} - Support Set", accept_multiple_files=True, type=["png", "jpg"])
img_num = 1
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    # Display images in rows of 3
    if img_num % 3 == 1:
        col1, col2, col3 = st.columns(3)

    if img_num % 3 == 1:
        with col1:
            st.image(uploaded_file, caption=uploaded_file.name)
            
    elif img_num % 3 == 2:
        with col2:
            st.image(uploaded_file, caption=uploaded_file.name)
    else:
        with col3:
            st.image(uploaded_file, caption=uploaded_file.name)
    img_num += 1
    img = Image.open(uploaded_file).convert('RGB')
    tensor = transform(img)
    tensor_list.append(tensor.unsqueeze(0))  # add a batch dimension

if tensor_list:
    # concatenate the tensors along the batch dimension
    tensor_list = torch.cat(tensor_list, dim=0)
    tensor_length = len(tensor_list)

    # split the concatenated tensor into a list of tensors with shape [2, 3, 224, 224]
    tensor_list = tensor_list.split(tensor_length, dim=0)
    torch.save(tensor_list, f"./support_sets/{support_set_name}_{tensor_length}.pt")
    st.balloons()
    st.success('Created Support Set!', icon='🎉')
    # st.write("Created Support Set ✅")

