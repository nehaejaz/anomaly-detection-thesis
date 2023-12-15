import os
import random
from argparse import ArgumentParser, Namespace
import time
import torch
import numpy as np
from torch.optim import optimizer
import torch.nn.functional as F
from tqdm import tqdm
from datasets.dataset import FSAD_Dataset_train, FSAD_Dataset_test
from utils.utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
from models.siamese import Encoder, Predictor

from models.stn import stn_net
from models.hf_convnext import HF_Convnext
from models.hf_resnet import HF_Resnet
from models.convnext_stn import convnext_tiny

from losses.norm_loss import CosLoss
from utils.funcs import embedding_concat, mahalanobis_torch, apply_augmentations, maddern_transform, nearest_neighbors, reshape_embedding, subsample_embedding, compute_anomaly_score
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from collections import OrderedDict
import warnings
import csv
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from config import get_configurable_parameters
from utils.AnomalyMapGenerator import AnomalyMapGenerator

warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser(description='Registration based Few-Shot Anomaly Detection')
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--CKPT_name", type=str, required=False, help="Path to a model checkpoint")

    return parser

def main():

    args = get_parser().parse_args()

    """Read the arguments from Config File"""
    config = get_configurable_parameters(config_path=args.config)

    args.input_channel = 3
    
    if config.project.get("seed") is None:
        config.project.seed = random.randint(1, 10000)
        random.seed(config.project.seed)
        torch.manual_seed(config.project.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(config.project.seed)
    args.prefix = time_file_str()
    
    model_names = {
        "resnet_stn": stn_net, 
        "resnet": HF_Resnet, 
        "convnext": HF_Convnext, 
        "convnext_stn": convnext_tiny, 
        }
    
    input_planes = {
        "resnet_stn": 256, 
        "resnet": 256, 
        "convnext": 768, 
        "convnext_stn": 768, 
        }
    
    STN = model_names[config.model.backbone](config).to(device)
    ENC = Encoder(input_planes[config.model.backbone],input_planes[config.model.backbone]).to(device)
    PRED = Predictor(input_planes[config.model.backbone],input_planes[config.model.backbone]).to(device)
    
    # load models
    CKPT_name = args.CKPT_name
    model_CKPT = torch.load(CKPT_name)
    model.load_state_dict(model_CKPT['STN'])
    ENC.load_state_dict(model_CKPT['ENC'])
    PRED.load_state_dict(model_CKPT['PRED'])
    models = [STN, ENC, PRED]

    print('Loading Datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    test_dataset = FSAD_Dataset_test(config.dataset.data_path, class_name=config.dataset.obj, is_train=False, resize=config.dataset.img_size, shot=config.dataset.shot, data_type=config.dataset.data_type)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    print('Loading Fixed Support Set')

    fixed_fewshot_list = torch.load(config.dataset.supp_set)

    print('Start Testing:')
    image_auc_list = []
    pixel_auc_list = []
    for inference_round in range(config.trainer.inferences):
        print('Round {}:'.format(inference_round))
        scores_list, test_imgs, gt_list, gt_mask_list = test(config, models, inference_round, fixed_fewshot_list, test_loader, **kwargs)
        
        scores = np.asarray(scores_list)
        
        # Normalization
        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()
        scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
        """The shape of scores is [83,224,224]"""

        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        image_auc_list.append(img_roc_auc)
        print("img_roc_auc", img_roc_auc)
        print("image_auc_list",image_auc_list)

        # calculate per-pixel level ROCAUC
        gt_mask = np.asarray(gt_mask_list)
        gt_mask = (gt_mask > 0.5).astype(np.int_)
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        pixel_auc_list.append(per_pixel_rocauc)
        print("per_pixel_rocauc", per_pixel_rocauc)
        print("pixel_auc_list", pixel_auc_list)



    image_auc_list = np.array(image_auc_list)
    pixel_auc_list = np.array(pixel_auc_list)
    mean_img_auc = np.mean(image_auc_list, axis = 0)
    mean_pixel_auc = np.mean(pixel_auc_list, axis = 0)
    print('Img-level AUC:',mean_img_auc)
    print('Pixel-level AUC:', mean_pixel_auc)


def test(config, models, cur_epoch, fixed_fewshot_list, test_loader, **kwargs):
    model = models[0]
    ENC = models[1]
    PRED = models[2]

    model.eval()
    ENC.eval()
    PRED.eval()
    
    memory_bank = torch.Tensor()
    
    if config.model.backbone == "resnet_stn":
        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    if config.model.backbone == "convnext":
        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('layer4', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []),('layer4', [])])

    #The shape support_img should be [2,3,224,224] [k, C, H, W]
    support_img = fixed_fewshot_list[cur_epoch]

    augment_support_img = support_img
    
    #Apply Augmentations
    augment_support_img = apply_augmentations(config,augment_support_img,support_img)
    
    #Create Maddern Transform of support images
    if config.dataset.include_maddern_transform is True:
        augment_support_img_mt = maddern_transform(augment_support_img, config.dataset.alpha)    
    
    # torch version
    with torch.no_grad():
        if config.model.backbone == "resnet_stn":
            support_feat = model(augment_support_img.to(device))
            if config.dataset.include_maddern_transform is True:
                #calculate MT Features here
                support_feat_mt = model(augment_support_img_mt.to(device))

        '''For HF Convnext Model'''
        if config.model.backbone == "convnext":
            output = model(augment_support_img.to(device))
            support_feat = output.last_hidden_state
            out_features = output.hidden_states
            if config.dataset.include_maddern_transform is True:
                #calculate MT Features here
                output = model(augment_support_img_mt.to(device))
                support_feat_mt = output.last_hidden_state
                out_features_mt = output.hidden_states
     
    support_feat = torch.mean(support_feat, dim=0, keepdim=True)
    
    if config.dataset.include_maddern_transform is True:
        support_feat_mt = torch.mean(support_feat, dim=0, keepdim=True)
        #Concat Supp Feat + Supp Feat MT
        support_feat = torch.cat([support_feat_mt, support_feat], dim=0)

    if config.model.backbone == "resnet_stn":
        train_outputs['layer1'].append(model.stn1_output)
        train_outputs['layer2'].append(model.stn2_output)
        train_outputs['layer3'].append(model.stn3_output)
        
    if config.model.backbone == "convnext":
        train_outputs['layer1'].append(out_features[1])
        train_outputs['layer2'].append(out_features[2])
        train_outputs['layer3'].append(out_features[3])
        train_outputs['layer4'].append(out_features[4])


    for k, v in train_outputs.items():
        train_outputs[k] = torch.cat(v, 0)

    # Embedding concat
    if config.model.backbone == "resnet_stn":
        embedding_vectors = train_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name], True)
    
    if config.model.backbone == "convnext":
        embedding_vectors = train_outputs['layer1']
        for layer_name in ['layer2', 'layer3','layer4']:
            embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name], True)
    
    """The shape of embedding_vectors is [44, 448, 56, 56]""" 
    print("embedding_vectors",embedding_vectors.shape)
 
    embedding_vectors = reshape_embedding(embedding_vectors)
    print("embedding_vectors reshaped",embedding_vectors.shape)

    #Applying core-set subsampling to get the embedding
    memory_bank = subsample_embedding(embedding_vectors, coreset_sampling_ratio= config.model.coreset_sampling_ratio)
    print("memory_bank",memory_bank.shape)

    # torch version
    query_imgs = []
    gt_list = []
    mask_list = []
    score_map_list = []

    for (query_img, support_img, mask, y) in tqdm(test_loader):
        
        print(query_img.shape)
       
        
        gt_list.extend(y.cpu().detach().numpy())
        mask_list.extend(mask.cpu().detach().numpy())

        query_imgs.extend(query_img.cpu().detach().numpy())
        
        """Calulate MT"""
        if config.dataset.include_maddern_transform is True:
            query_img_mt = maddern_transform(query_img, config.dataset.alpha)

        # model prediction
        if config.model.backbone == "resnet_stn":
            query_feat = model(query_img.to(device))
            if config.dataset.include_maddern_transform is True:
                query_feat_mt = model(query_img_mt.to(device))
                #concat Query Feat + Query MT Feat
                query_feat = torch.cat([query_feat_mt, query_feat], dim=0) 


        if config.model.backbone == "convnext":
            output = model(query_img.to(device))
            query_feat = output.last_hidden_state
            out_features = output.hidden_states
            """TODO: Find MT Feat and Concat"""
            

        z1 = ENC(query_feat)
        z2 = ENC(support_feat)
        p1 = PRED(z1)
        p2 = PRED(z2)

        loss = CosLoss(p1,z2, Mean=False)/2 + CosLoss(p2,z1, Mean=False)/2
        loss_reshape = F.interpolate(loss.unsqueeze(1), size=query_img.size(2), mode='bilinear',align_corners=False).squeeze(0)
        score_map_list.append(loss_reshape.cpu().detach().numpy())

        if config.model.backbone == "resnet_stn":
            test_outputs['layer1'].append(model.stn1_output)
            test_outputs['layer2'].append(model.stn2_output)
            test_outputs['layer3'].append(model.stn3_output)
        
        if config.model.backbone == "convnext":
            test_outputs['layer1'].append(out_features[1])
            test_outputs['layer2'].append(out_features[2])
            test_outputs['layer3'].append(out_features[3])
            test_outputs['layer4'].append(out_features[4])

    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)

    # Embedding concat
    if config.model.backbone == "resnet_stn":
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name], True)
        """The shape of embedding_vectors is [83, 448, 56, 56]""" 
    
    if config.model.backbone == "convnext":
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3', 'layer4']:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name], True)
        """The shape of embedding_vectors is [83, 448, 56, 56]""" 
    
    batch_size, _, width, height = embedding_vectors.shape

    #apply reshaping on query embeddings 
    embedding_vectors = reshape_embedding(embedding_vectors)
    print(embedding_vectors.shape)

    #apply nearest neighbor search on query embeddings 
    patch_scores, locations = nearest_neighbors(embedding=embedding_vectors, n_neighbors=config.model.num_neighbors, memory_bank=memory_bank)
    
    # reshape to batch dimension
    """The shape of patch_scores and locations is [83, 260288]"""
    patch_scores = patch_scores.reshape((batch_size, -1))
    locations = locations.reshape((batch_size, -1))
    print("A-patch_scores", patch_scores.shape)
    print("A-locations", locations.shape)
    
    #compute anomaly score
    anomaly_score = compute_anomaly_score(patch_scores, locations, embedding_vectors, memory_bank)
    print("anomaly_score", anomaly_score.shape)

    #reshape to w, h
    patch_scores = patch_scores.reshape((batch_size, 1, width, height))
    """The shape of patch_scores is [83, 1,  56, 56]"""
    
    #Generate anomaly map
    anomaly_map_generator = AnomalyMapGenerator(input_size=config.dataset.img_size)
    anomaly_map = anomaly_map_generator(patch_scores)
    """The shape of anomaly_map is [83, 1,  24, 24]"""
    
    #Put it on CPU and convert to numpy
    # score_map = anomaly_map.cpu().numpy()
    score_map = anomaly_map.cpu().detach().numpy()

        
    """To Generate the Heat Maps. Basically the score_map
    is the score of the patchesvwhere the anomalies are present."""   
    print(score_map.shape) 
    """The shape of score_map is (83,1, 224, 224)"""
    
    score_map = np.squeeze(score_map)
    """The shape of score_map is (83,224, 224)"""

    print(score_map.shape) 

    # for i in range(score_map.shape[0]):
    #     image = score_map[i, :, :]
    #     plt.imshow(image, cmap='gray')
    #     plt.colorbar()
    #     plt.savefig("rounds/"+str(cur_epoch)+"/" + str(i)+".jpg")
    #     plt.close()
    #     print("image", image.shape)

    """The shape of score_map is (83, 224, 224)"""
    return score_map, query_imgs, gt_list, mask_list
    
if __name__ == '__main__':
    main()
