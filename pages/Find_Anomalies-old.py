import os
import io
import cv2
import time
from zipfile import ZipFile
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.funcs import embedding_concat, mahalanobis_torch, apply_augmentations, maddern_transform, nearest_neighbors, reshape_embedding, subsample_embedding, compute_anomaly_score, print_with_loader
from datasets.dataset import FSAD_Dataset_streamlit, FSAD_Dataset_inference
from models.hf_convnext import HF_Convnext
from models.siamese import Encoder, Predictor
from losses.norm_loss import CosLoss
from utils.AnomalyMapGenerator import AnomalyMapGenerator

import streamlit as st
from io import StringIO

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def main():
    
    # Initialize session state
    if 'loaded' not in st.session_state:
        st.session_state.loaded = True


    class DotDict(dict):
        def __getattr__(self, attr):
            if attr in self:
                if isinstance(self[attr], dict):
                    return DotDict(self[attr])  # Convert nested dictionary to DotDict
                else:
                    return self[attr]
            else:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    config = DotDict({
        'model': {
            'drop_path_rate': 0.7
        },
        'dataset': {
            'data_path': './VisA'
        }
    })

    input_channel = 3
    
    if use_cuda:
        torch.cuda.manual_seed_all(668)
    # args.prefix = time_file_str()
  
    STN = HF_Convnext(config).to(device)
    ENC = Encoder(768,768).to(device)
    PRED = Predictor(768,768).to(device)
    
    # load models
    CKPT_name = "/home/nejaz/few-shot-visual-anomaly-detection/final_model_convnext.pt"
    model_CKPT = torch.load(CKPT_name)
    STN.load_state_dict(model_CKPT['STN'])
    ENC.load_state_dict(model_CKPT['ENC'])
    PRED.load_state_dict(model_CKPT['PRED'])
    models = [STN, ENC, PRED]

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

   
    #pick up the supprt set from drowndown
    supp_set = pick_supp_set()
    fixed_fewshot_list = torch.load(f'./support_sets/{supp_set}.pt')
    visualize_supp_set(fixed_fewshot_list)
    query_images = upload_test_images()

    #TODO: Take test images as input and then pass it to the test loader
    #Pass the uploaded image here, object name
    query_dataset = FSAD_Dataset_inference("./MVTec", class_name="bottle", is_train=False, resize=224, shot=2, data_type="mpdd")
    query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=1, shuffle=False, **kwargs)

    image_auc_list = []
    start_time = time.time()
    st.divider()
    
    if query_images:
        st.write(st.session_state.loaded)
        if st.session_state.loaded:
            st.toast('Generating Results...', icon='⏳')
            with st.spinner('Generating Results...'):
                scores_list, test_imgs = test(models, fixed_fewshot_list, query_loader)
                scores = np.asarray(scores_list)
            st.success('Done! 🤗')
        
            st.balloons()
        
        
        # Normalization
        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()
        scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
        # Ensure scores has the shape (N, 224, 224)
        if len(scores.shape) == 2:  # Single image case
            scores = scores[np.newaxis, ...]  # Add a new axis at the beginning
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        # gt_list = np.asarray(gt_list)
        
        #Set a threshold value to display high regions, this value is adjustable
        threshold = 0.5  # This is an example value, adjust it according to your needs
        
        st.header("Heat Maps:")
        heat_maps(test_imgs, scores, threshold, obj="mvtec")
        
        st.header("Classification Results:")
        classifi_visual(test_imgs, img_scores, obj="mvtec")
        if st.session_state.loaded:
            st.toast('Finished Processing!', icon='✅')

        # Folder containing images
        image_folder = './visuals'
        zip_name = 'results.zip'
        
        # Create a ZIP file
        create_zip_from_folder(image_folder, zip_name)
            
        st.divider()
        st.session_state.loaded = False
        query_images = []
        st.write(st.session_state.loaded)

        # Download button for the ZIP file
        with open(zip_name, "rb") as file:
            st.download_button(
                    label="Download Images",
                    data=file,
                    file_name=zip_name,
                    mime="application/zip",
                    help="Click to download a ZIP file containing the classification results and heatmap visualizations for each image",
                    use_container_width=True,
                    type="secondary"
                )
        


def visualize_supp_set(image_tensor):
    image_tensor = image_tensor[0]
    print(image_tensor.shape)

    # Assuming your tensor is in range [0, 1]
    img_np = image_tensor.permute(0, 2, 3, 1).numpy()
    
    img_num = 1

    # Display the images
    plt.figure(figsize=(10, 5))
    for i in range(len(image_tensor)):  # Displaying 2 images from the batch
        plt.subplot(1, 2, i + 1)
        plt.imshow(img_np[i])
        plt.axis('off')
        
        # Convert the plot to an image buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Display images in rows of 3
        if img_num % 3 == 1:
            col1, col2, col3 = st.columns(3)

        if img_num % 3 == 1:
            with col1:
                st.image(buf, caption=f'Image {img_num}')
        elif img_num % 3 == 2:
            with col2:
                st.image(buf, caption=f'Image {img_num}')
        else:
            with col3:
                st.image(buf, caption=f'Image {img_num}')
        
        img_num = 1
        plt.close() 

def heat_maps(test_imgs, scores, threshold, obj="mvtec"):
    img_num = 1
    for index, img in enumerate(test_imgs):
       
        orig_image = img.transpose(1, 2, 0)
        print(scores.shape)
        # Apply the threshold to the scores array
        thresholded_scores = np.where(scores[index] > threshold, scores[index], 0)
        print(thresholded_scores.shape)

        # Plot the original image
        plt.imshow(orig_image)  # Assuming the original image is grayscale
      
        # Overlay the heatmap. Use the 'alpha' parameter for transparency.
        plt.imshow(thresholded_scores, cmap='hot', alpha=0.5)  
        plt.axis('off')  # Hide axis
        plt.savefig(f"visuals/heatmaps/{img_num}.png") 

        # Convert the plot to an image buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Display images in rows of 3
        if img_num % 3 == 1:
            col1, col2, col3 = st.columns(3)

        if img_num % 3 == 1:
            with col1:
                st.image(buf, caption=f'Image {img_num}')
        elif img_num % 3 == 2:
            with col2:
                st.image(buf, caption=f'Image {img_num}')
        else:
            with col3:
                st.image(buf, caption=f'Image {img_num}')

        plt.close() 
        img_num +=1

def classifi_visual(test_imgs, img_scores, obj="mvtec"):
    """save test images for classification """
    img_num = 1

    for index, img in enumerate(test_imgs):
        # Transpose from [channels, height, width] to [height, width, channels]
        img = np.transpose(img, (1, 2, 0))

        # The number you want to write
        # ground_truth_lab = gt_list[index]
        score_num = img_scores[index]

        fig, ax = plt.subplots()

        # Visualize the image
        ax.imshow(img)

        # Add text at the bottom left (x=0, y=image height)
        # ax.text(0, img.shape[0], str(ground_truth_lab), color='white', fontsize=16, weight='bold', verticalalignment='bottom')
            
        # Add text at the bottom right (x=image width, y=image height)
        ax.text(img.shape[1], img.shape[0], str(score_num), color='red', fontsize=16, weight='bold', verticalalignment='bottom', horizontalalignment='right')

        # Visualize the image
        plt.imshow(img)
        plt.savefig(f"visuals/classification/{img_num}.png") 

        # Convert the plot to an image buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Display images in rows of 3
        if img_num % 3 == 1:
            col1, col2, col3 = st.columns(3)

        if img_num % 3 == 1:
            with col1:
                st.image(buf, caption=f'Image {img_num}')
        elif img_num % 3 == 2:
            with col2:
                st.image(buf, caption=f'Image {img_num}')
        else:
            with col3:
                st.image(buf, caption=f'Image {img_num}')

        plt.close(fig)
        img_num +=1 

# Function to create a ZIP file from a folder of images
def create_zip_from_folder(folder_path, zip_name):
    with ZipFile(zip_name, 'w') as zip_file:
        for foldername, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                zip_file.write(file_path, os.path.relpath(file_path, folder_path))
                       
def upload_test_images():
    uploaded_files = st.file_uploader("Upload test images", accept_multiple_files=True, type=["png"])
    img_num = 1  # Initialize image number
    img_list = []
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert('RGB')
        img_list.append(img)
        
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

        img_num += 1  # Increment image number
    return img_list
        
def pick_supp_set():
    supp_set_names_tuple = ()
    support_set_folder = "./support_sets"
    for supp_set_name in os.listdir(support_set_folder):
        supp_set_name = os.path.splitext(supp_set_name)[0]
        supp_set_names_tuple += (supp_set_name,)
            
    choosen_supp_set = st.selectbox(
    'Select a support set',
    supp_set_names_tuple)

    st.write('You selected:', choosen_supp_set)   
    return choosen_supp_set

def test(models, fixed_fewshot_list, test_loader):
    model = models[0]
    ENC = models[1]
    PRED = models[2]

    model.eval()
    ENC.eval()
    PRED.eval()
    
    memory_bank = torch.Tensor()
    
    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('layer4', [])])
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []),('layer4', [])])

    #The shape support_img should be [2,3,224,224] [k, C, H, W]
    support_img = fixed_fewshot_list[0]
    augment_support_img = support_img
    
    #Apply Augmentations
    augment_support_img = apply_augmentations(augment_support_img,support_img)
  
    # torch version
    with torch.no_grad():
            output = model(augment_support_img.to(device))
            support_feat = output.last_hidden_state
            out_features = output.hidden_states
     
    support_feat = torch.mean(support_feat, dim=0, keepdim=True)
        
    train_outputs['layer1'].append(out_features[1])
    train_outputs['layer2'].append(out_features[2])
    train_outputs['layer3'].append(out_features[3])
    train_outputs['layer4'].append(out_features[4])


    for k, v in train_outputs.items():
        train_outputs[k] = torch.cat(v, 0)
    
    embedding_vectors = train_outputs['layer1']
    for layer_name in ['layer2', 'layer3','layer4']:
        embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name], True)
    
 
    embedding_vectors = reshape_embedding(embedding_vectors)
    # print("embedding_vectors reshaped",embedding_vectors.shape)

    #Applying core-set subsampling to get the embedding
    memory_bank = subsample_embedding(embedding_vectors, coreset_sampling_ratio= 0.01)
    # print("memory_bank",memory_bank.shape)

    # torch version
    query_imgs = []
    # gt_list = []
    mask_list = []
    score_map_list = []

    for (query_img,y) in tqdm(test_loader):
        
        # print(query_img.shape)
        # print(mask.shape)
        
        # gt_list.extend(y.cpu().detach().numpy())
        # mask_list.extend(mask.cpu().detach().numpy())

        query_imgs.extend(query_img.cpu().detach().numpy()) 


        output = model(query_img.to(device))
        query_feat = output.last_hidden_state
        out_features = output.hidden_states

        z1 = ENC(query_feat)
        z2 = ENC(support_feat)
        p1 = PRED(z1)
        p2 = PRED(z2)
        # print("z1",z1.shape,"z2",z2.shape,"p1",p1.shape,"p2",p2.shape)
        # exit()

        loss = CosLoss(p1,z2, Mean=False)/2 + CosLoss(p2,z1, Mean=False)/2
        loss_reshape = F.interpolate(loss.unsqueeze(1), size=query_img.size(2), mode='bilinear',align_corners=False).squeeze(0)
        score_map_list.append(loss_reshape.cpu().detach().numpy())
        
        test_outputs['layer1'].append(out_features[1])
        test_outputs['layer2'].append(out_features[2])
        test_outputs['layer3'].append(out_features[3])
        test_outputs['layer4'].append(out_features[4])

    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)

    # Embedding concat
    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3', 'layer4']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name], True)
    
    batch_size, _, width, height = embedding_vectors.shape

    #apply reshaping on query embeddings 
    embedding_vectors = reshape_embedding(embedding_vectors)
    # print(embedding_vectors.shape)

    #apply nearest neighbor search on query embeddings 
    patch_scores, locations = nearest_neighbors(embedding=embedding_vectors, n_neighbors=1, memory_bank=memory_bank)
    
    # reshape to batch dimension
    patch_scores = patch_scores.reshape((batch_size, -1))
    locations = locations.reshape((batch_size, -1))
    # print("A-patch_scores", patch_scores.shape)
    # print("A-locations", locations.shape)
    
    #compute anomaly score
    anomaly_score = compute_anomaly_score(patch_scores, locations, embedding_vectors, memory_bank)
    # print("anomaly_score", anomaly_score.shape)

    #reshape to w, h
    patch_scores = patch_scores.reshape((batch_size, 1, width, height))
    
    #Generate anomaly map
    anomaly_map_generator = AnomalyMapGenerator(224)
    anomaly_map = anomaly_map_generator(patch_scores)
    
    #Put it on CPU and convert to numpy
    # score_map = anomaly_map.cpu().numpy()
    score_map = anomaly_map.cpu().detach().numpy()
    
    score_map = np.squeeze(score_map)

    return score_map, query_imgs   




#show result images

if __name__ == '__main__':
    main()
