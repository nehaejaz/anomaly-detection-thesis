import os
import random
import time
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datasets.dataset import FSAD_Dataset_train, FSAD_Dataset_test
from utils.utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
from models.siamese import Encoder, Predictor

from models.stn import stn_net
from models.hf_convnext import HF_Convnext
from models.hf_resnet import HF_Resnet
from models.convnext_stn import convnext_tiny

from losses.norm_loss import CosLoss
from utils.funcs import embedding_concat, mahalanobis_torch, maddern_transform
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from collections import OrderedDict
from test import test
from argparse import ArgumentParser, Namespace
from config import get_configurable_parameters

import warnings
warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

import torch
from torch.utils.tensorboard import SummaryWriter

def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser(description='Registration based Few-Shot Anomaly Detection')
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    return parser

def main():
    args = get_parser().parse_args()

    """Read the arguments from Config File"""
    config = get_configurable_parameters(config_path=args.config)


    if config.project.get("seed") is None:
        config.project.seed = random.randint(1, 10000)
        random.seed(config.project.seed)
        torch.manual_seed(config.project.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(config.project.seed)

    args.prefix = time_file_str()
    # if not os.path.exists(config.project.save_dir):
    #     os.makedirs("results/"+config.project.save_dir)

    """Create a Save Dir for MODEL """
    args.save_model_dir = config.project.save_dir + config.trainer.stn_mode + '/' + str(config.dataset.shot) + '/' + config.dataset.obj + '/'
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    
    writer = SummaryWriter()

    """Create a LOG file to save the results """
    log = open(os.path.join(config.project.save_dir, 'log_{}_{}.txt'.format(str(config.dataset.shot),config.dataset.obj)), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    # load model and dataset
    """Select which model you need to train on"""
    model_names = {
        "resnet_stn": stn_net, 
        "resnet": HF_Resnet, 
        "convnext": HF_Convnext, 
        "convnext_stn": convnext_tiny, 
        }
    input_planes = {
        "resnet_stn": 256, 
        "resnet": 512, 
        "convnext": 768, 
        "convnext_stn": 768, 
        }

    model = model_names[config.model.backbone](config).to(device)
    ENC = Encoder(input_planes[config.model.backbone],input_planes[config.model.backbone]).to(device)
    PRED = Predictor(input_planes[config.model.backbone],input_planes[config.model.backbone]).to(device)

    model_optimizer = optim.SGD(model.parameters(), lr=config.trainer.lr, momentum=config.trainer.momentum)
    ENC_optimizer = optim.SGD(ENC.parameters(), lr=config.trainer.lr, momentum=config.trainer.momentum)
    PRED_optimizer = optim.SGD(PRED.parameters(), lr=config.trainer.lr, momentum=config.trainer.momentum)
    models = [model, ENC, PRED]
    optimizers = [model_optimizer, ENC_optimizer, PRED_optimizer]
    init_lrs = [config.trainer.lr, config.trainer.lr, config.trainer.lr]

    print('Loading Datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    
    """Create Synthetic data fol illumination"""
    if config.dataset.ilumination_data is True:
        pass;#Future Work
    train_dataset = FSAD_Dataset_train(config.dataset.data_path, class_name=config.dataset.obj, is_train=True, resize=config.dataset.img_size, shot=config.dataset.shot, batch=config.dataset.batch_size, data_type=config.dataset.data_type )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
    test_dataset = FSAD_Dataset_test(config.dataset.data_path, class_name=config.dataset.obj, is_train=False, resize=config.dataset.img_size, shot=config.dataset.shot, data_type=config.dataset.data_type)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    """Create a name for Model"""
    save_name = os.path.join(args.save_model_dir, '{}_{}_{}_model.pt'.format(config.dataset.obj, config.dataset.shot, config.trainer.stn_mode))
    start_time = time.time()
    epoch_time = AverageMeter()
    img_roc_auc_old = 0.0
    per_pixel_rocauc_old = 0.0
    print('Loading Fixed Support Set')
    
    fixed_fewshot_list = torch.load(config.dataset.supp_set)
    print_log((f'---------{config.trainer.stn_mode}--------'), log)

    for epoch in range(1, config.trainer.epochs + 1):
        adjust_learning_rate(optimizers, init_lrs, epoch, config)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (config.trainer.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, config.trainer.epochs, time_string(), need_time), log)

        if epoch <= config.trainer.epochs:
            image_auc_list = []
            pixel_auc_list = []
            
            train(config, models, epoch, train_loader, optimizers, log, writer)
            train_dataset.shuffle_dataset()
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
            
            for inference_round in tqdm(range(config.trainer.inferences)):
                scores_list, test_imgs, gt_list, gt_mask_list = test(config, models, inference_round, fixed_fewshot_list,
                                                                     test_loader, **kwargs)
                scores = np.asarray(scores_list)
                # Normalization
                max_anomaly_score = scores.max()
                min_anomaly_score = scores.min()
                scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

                # calculate image-level ROC AUC score
                img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
                gt_list = np.asarray(gt_list)
                img_roc_auc = roc_auc_score(gt_list, img_scores)
                image_auc_list.append(img_roc_auc)

                # calculate per-pixel level ROCAUC
                gt_mask = np.asarray(gt_mask_list)
                gt_mask = (gt_mask > 0.5).astype(np.int_)
                per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
                pixel_auc_list.append(per_pixel_rocauc)

            image_auc_list = np.array(image_auc_list)
            pixel_auc_list = np.array(pixel_auc_list)
            mean_img_auc = np.mean(image_auc_list, axis = 0)
            mean_pixel_auc = np.mean(pixel_auc_list, axis = 0)

            if mean_img_auc + mean_pixel_auc > per_pixel_rocauc_old + img_roc_auc_old:
                state = {'STN': model.state_dict(), 'ENC': ENC.state_dict(), 'PRED':PRED.state_dict()}
                torch.save(state, save_name)
                per_pixel_rocauc_old = mean_pixel_auc
                img_roc_auc_old = mean_img_auc
            print('Img-level AUC:',img_roc_auc_old)
            print('Pixel-level AUC:', per_pixel_rocauc_old)
            
            writer.add_scalar("Test/Accuracy",mean_img_auc, epoch-1)
            print_log(('Test Epoch(img, pixel): {} ({:.6f}, {:.6f}) best: ({:.3f}, {:.3f})'
            .format(epoch-1, mean_img_auc, mean_pixel_auc, img_roc_auc_old, per_pixel_rocauc_old)), log)

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        
    
    writer.close()
    log.close()

def train(config, models, epoch, train_loader, optimizers, log, writer):
    model = models[0]
    ENC = models[1]
    PRED = models[2]

    model_optimizer = optimizers[0]
    ENC_optimizer = optimizers[1]
    PRED_optimizer = optimizers[2]

    model.train()
    ENC.train()
    PRED.train()
    

    total_losses = AverageMeter()
    for (query_img, support_img_list, _) in tqdm(train_loader):
        model_optimizer.zero_grad()
        ENC_optimizer.zero_grad()
        PRED_optimizer.zero_grad()

        query_img = query_img.squeeze(0).to(device)

        support_img = support_img_list.squeeze(0).to(device)
        B,K,C,H,W = support_img.shape 

        support_img = support_img.view(B * K, C, H, W) 
                   
        """ Create Maddern Transform of the support image and query images"""
        if config.dataset.include_maddern_transform == True:
            query_img_mt = maddern_transform(query_img, config.dataset.alpha)
            supp_img_mt = maddern_transform(support_img, config.dataset.alpha)                    
        
        """The shape of query_feat is [32, 256, 14, 14] [B,C,H,W]"""
        """The shape of support_feat is [64, 256, 14, 14] because we ARE DOING B*K images"""

        if config.model.backbone == "resnet_stn":
            query_feat = model(query_img)
            support_feat = model(support_img)
            if config.dataset.include_maddern_transform is True:
                #calculate features of MT
                query_feat_mt = model(query_img_mt)
                support_feat_mt = model(supp_img_mt)

        if config.model.backbone == "convnext" or config.model.backbone == "resnet":
            query_feat = model(query_img).last_hidden_state
            support_feat = model(support_img).last_hidden_state
            if config.dataset.include_maddern_transform is True:
                #calculate features of MT
                query_feat_mt = model(query_img_mt).last_hidden_state
                support_feat_mt = model(supp_img_mt).last_hidden_state
                

        print("query_feat", query_feat.shape)
        print("support_feat", support_feat.shape)

        """Because we have k-shot images"""
        support_feat = support_feat / K
            
        _, C, H, W = support_feat.shape
        support_feat = support_feat.view(B, K, C, H, W)
        
        support_feat = torch.sum(support_feat, dim=1)
        
        if config.dataset.include_maddern_transform is True:
            support_feat_mt = support_feat_mt / K
            _, C, H, W = support_feat_mt.shape
            support_feat_mt = support_feat_mt.view(B, K, C, H, W)
            support_feat_mt = torch.sum(support_feat_mt, dim=1)
            
            """Concatinate the Query + Query MT"""
            query_feat = torch.cat([query_feat_mt, query_feat], dim=0) 
            """Concatinate the Supp + Supp MT"""
            support_feat = torch.cat([support_feat_mt, support_feat], dim=0) 

        """The shape of z1 is [32, 256, 14, 14]"""
        z1 = ENC(query_feat)
        print("z1=>", z1.shape)
        """The shape of z2 is [32, 256, 14, 14]"""
        z2 = ENC(support_feat)
        print("z2", z2.shape)
        """The shape of p1 is [32, 256, 14, 14]"""
        p1 = PRED(z1)
        print("p1", p1.shape)
        """The shape of p2 is [32, 256, 14, 14]"""
        p2 = PRED(z2)
        print("p2", p2.shape)
        
        total_loss = CosLoss(p1,z2, Mean=True)/2 + CosLoss(p2,z1, Mean=True)/2
        total_losses.update(total_loss.item(), query_img.size(0))

        total_loss.backward()

        model_optimizer.step()
        ENC_optimizer.step()
        PRED_optimizer.step()
        
    writer.add_scalar("Train/Loss",total_losses.avg, epoch)
    print_log(('Train Epoch: {} Total_Loss: {:.6f}'.format(epoch, total_losses.avg)), log)

def adjust_learning_rate(optimizers, init_lrs, epoch, config):
    """Decay the learning rate based on schedule"""
    for i in range(3):
        cur_lr = init_lrs[i] * 0.5 * (1. + math.cos(math.pi * epoch / config.trainer.epochs))
        for param_group in optimizers[i].param_groups:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()
