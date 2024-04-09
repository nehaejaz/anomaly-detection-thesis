import os
import io
from PIL import Image
import random
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
import streamlit as st

client_id = st.secrets['AZURE_CLIENT_ID']
tenant_id = st.secrets['AZURE_TENANT_ID']
client_secret = st.secrets['AZURE_CLIENT_SECRET']
account_url = st.secrets["AZURE_STORAGE_URL"]

# create a credential 
credentials = ClientSecretCredential(
    client_id = client_id, 
    client_secret= client_secret,
    tenant_id= tenant_id
)

class FSAD_Dataset_train(Dataset):
    def __init__(self,
                 dataset_path='../data/mvtec_anomaly_detection',
                 class_name='bottle',
                 is_train=True,
                 resize=256,
                 shot=2,
                 batch=32,
                 data_type="mvtec"
                 ):
        
        # Set CLASS_NAMES based on data_type
        if data_type == "mvtec":
            self.CLASS_NAMES = [
                'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
            ]
        elif data_type == "mpdd":
            self.CLASS_NAMES = [
                'bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes'
            ]
        elif data_type == "axiom":
            self.CLASS_NAMES = [
                'mc9', 'mc27'
            ]
        else:
            raise ValueError("This datatype is not supported by us.")
        assert class_name in self.CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, self.CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.shot = shot
        self.batch = batch
        # load dataset
        self.query_dir, self.support_dir = self.load_dataset_folder()
        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize(resize, Image.LANCZOS),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        query_list, support_list = self.query_dir[idx], self.support_dir[idx]
        query_img = None
        support_sub_img = None
        support_img = None

        for i in range(len(query_list)):
            image = Image.open(query_list[i])
            # Check if the image is grayscale
            if image.mode == 'L':
                print("The image is grayscale.")
                rgb_image = image.convert('RGB')
            else:
                print("The image is color.")
                image = Image.open(query_list[i]).convert('RGB')
            image = self.transform_x(image) #image_shape torch.Size([3, 224, 224])
            image = image.unsqueeze(dim=0) #image_shape torch.Size([1, 3, 224, 224])
            if query_img is None:
                query_img = image
            else:
                query_img = torch.cat([query_img, image],dim=0)

            for k in range(self.shot):
                image = Image.open(support_list[i][k])
                # Check if the image is grayscale
                if image.mode == 'L':
                    print("The image is grayscale.")
                    rgb_image = image.convert('RGB')
                else:
                    print("The image is color.")
                    image = Image.open(support_list[i][k]).convert('RGB')
                image = self.transform_x(image)
                image = image.unsqueeze(dim=0) #image_shape torch.Size([1, 3, 224, 224])
                if support_sub_img is None:
                    support_sub_img = image
                else:
                    support_sub_img = torch.cat([support_sub_img, image], dim=0)

            support_sub_img = support_sub_img.unsqueeze(dim=0)
            if support_img is None:
                support_img = support_sub_img
            else:
                support_img = torch.cat([support_img, support_sub_img], dim=0)

            support_sub_img = None

        mask = torch.zeros([self.batch, self.resize, self.resize])
        return query_img, support_img, mask

    def __len__(self):
        return len(self.query_dir)
    
    def shuffle_dataset(self):
        phase = 'train' if self.is_train else 'test'

        data_img = {}
        # data_img includes all image pathes, key: class_name like wood, zipper. value: each image path.
        for class_name_one in self.CLASS_NAMES:
            if class_name_one != self.class_name:
                data_img[class_name_one] = []
                img_dir = os.path.join(self.dataset_path, class_name_one, phase, 'good')
                img_types = sorted(os.listdir(img_dir))
                for img_type in img_types:
                    img_type_dir = os.path.join(img_dir, img_type)
                    data_img[class_name_one].append(img_type_dir)
                random.shuffle(data_img[class_name_one])

        query_dir, support_dir = [], []
        for class_name_one in data_img.keys():
            for image_index in range(0, len(data_img[class_name_one]), self.batch):
                query_sub_dir = []
                support_sub_dir = []
                for batch_count in range(0, self.batch):
                    if image_index + batch_count >= len(data_img[class_name_one]):
                        break
                    image_dir_one = data_img[class_name_one][image_index + batch_count]
                    support_dir_one = []
                    query_sub_dir.append(image_dir_one)
                    for k in range(self.shot):
                        random_choose = random.randint(0, (len(data_img[class_name_one]) - 1))
                        while data_img[class_name_one][random_choose] == image_dir_one:
                            random_choose = random.randint(0, (len(data_img[class_name_one]) - 1))
                        support_dir_one.append(data_img[class_name_one][random_choose])
                    support_sub_dir.append(support_dir_one)
                
                query_dir.append(query_sub_dir)
                support_dir.append(support_sub_dir)

        assert len(query_dir) == len(support_dir), 'number of query_dir and support_dir should be same'
        self.query_dir = query_dir
        self.support_dir = support_dir

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'

        data_img = {}
        for class_name_one in self.CLASS_NAMES:
            if class_name_one != self.class_name:
                data_img[class_name_one] = []
                img_dir = os.path.join(self.dataset_path, class_name_one, phase, 'good')
                img_types = sorted(os.listdir(img_dir))
                for img_type in img_types:
                    if img_type != ".DS_Store":
                        img_type_dir = os.path.join(img_dir, img_type)
                        data_img[class_name_one].append(img_type_dir)
                random.shuffle(data_img[class_name_one])

        query_dir, support_dir = [], []
        for class_name_one in data_img.keys():

            for image_index in range(0, len(data_img[class_name_one]), self.batch):
                query_sub_dir = []
                support_sub_dir = []

                for batch_count in range(0, self.batch):
                    if image_index + batch_count >= len(data_img[class_name_one]):
                        break
                    image_dir_one = data_img[class_name_one][image_index + batch_count]
                    support_dir_one = []
                    query_sub_dir.append(image_dir_one)
                    for k in range(self.shot):
                        random_choose = random.randint(0, (len(data_img[class_name_one]) - 1))
                        while data_img[class_name_one][random_choose] == image_dir_one:
                            random_choose = random.randint(0, (len(data_img[class_name_one]) - 1))
                        support_dir_one.append(data_img[class_name_one][random_choose])
                    support_sub_dir.append(support_dir_one)
                
                query_dir.append(query_sub_dir)
                support_dir.append(support_sub_dir)

        assert len(query_dir) == len(support_dir), 'number of query_dir and support_dir should be same'
        return query_dir, support_dir


class FSAD_Dataset_test(Dataset):
    def __init__(self,
                 dataset_path='../mvtec_loco_anomaly_detection',
                 class_name='bottle',
                 is_train=True,
                 resize=256,
                 shot=2,
                 data_type="mvtec"
                 ):
        
        # Set CLASS_NAMES based on data_type
        if data_type == "mvtec":
            self.CLASS_NAMES = [
                'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
            ]
        elif data_type == "mpdd":
            self.CLASS_NAMES = [
                'bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes'
            ]
        elif data_type == "axiom":
            self.CLASS_NAMES = [
                'mc27', "mc9"
            ]
        else:
            raise ValueError("This datatype is not supported by us.")
        assert class_name in self.CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, self.CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.shot = shot
        # load dataset
        self.query_dir, self.support_dir, self.query_mask = self.load_dataset_folder()
        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize((resize,224), Image.LANCZOS),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_mask = transforms.Compose(
            [transforms.Resize((resize,224), Image.NEAREST),
             transforms.ToTensor()])

    def __getitem__(self, idx):
        query_one, support_one, mask_one = self.query_dir[idx], self.support_dir[idx], self.query_mask[idx]
        if not ("/.DS_Store" in query_one):
            query_img = Image.open(query_one).convert('RGB')
            query_img = self.transform_x(query_img)

        support_img = []
        for k in range(self.shot):
            support_img_one = Image.open(support_one[k]).convert('RGB')
            support_img_one = self.transform_x(support_img_one)
            support_img.append(support_img_one)

        if 'good' in mask_one:
            mask = torch.zeros([1, self.resize, self.resize])
            y = 0
        else:
            mask = Image.open(mask_one)
            mask = self.transform_mask(mask)
            y = 1
        return query_img, support_img, mask, y

    def __len__(self):
        return len(self.query_dir)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        query_dir, support_dir = [], []
        data_img = {}
        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            data_img[img_type] = []
            img_type_dir = os.path.join(img_dir, img_type)
            if ".DS_Store" not in os.path.basename(img_type_dir):
                img_num = sorted(os.listdir(img_type_dir))
                for img_one in img_num:
                    if img_type != ".DS_Store":
                        img_dir_one = os.path.join(img_type_dir, img_one)
                        data_img[img_type].append(img_dir_one)
        img_dir_train = os.path.join(self.dataset_path, self.class_name, 'train', 'good')
        img_num = sorted(os.listdir(img_dir_train))

        data_train = []
        for img_one in img_num:
            img_dir_one = os.path.join(img_dir_train, img_one)
            data_train.append(img_dir_one)

        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
        query_dir, support_dir, query_mask = [], [], []
        for img_type in data_img.keys():
            for image_dir_one in data_img[img_type]:
                support_dir_one = []
                if not ("/.DS_Store" in image_dir_one):
                    query_dir.append(image_dir_one)
                    query_mask_dir = image_dir_one.replace('test', 'ground_truth')
                    query_mask_dir = query_mask_dir[:-4] + '_mask.png'
                    query_mask.append(query_mask_dir)
                    for k in range(self.shot):
                        random_choose = random.randint(0, (len(data_train) - 1))
                        support_dir_one.append(data_train[random_choose])
                    support_dir.append(support_dir_one)

        assert len(query_dir) == len(support_dir) == len(
            query_mask), 'number of query_dir and support_dir should be same'
        return query_dir, support_dir, query_mask

class FSAD_Dataset_inference(Dataset):
    def __init__(self,
                 dataset_path='../mvtec_loco_anomaly_detection',
                 class_name='bottle',
                is_train=True,
                 resize=256,
                 shot=2,
                 data_type="mvtec"
                 ):
       
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.resize = resize
        self.shot = shot
        # load dataset
        self.query_dir = self.load_dataset_folder()
        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize((resize,224), Image.LANCZOS),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        query_one = self.query_dir[idx]
        if not ("/.DS_Store" in query_one):
            query_img = Image.open(query_one).convert('RGB')
            query_img = self.transform_x(query_img)

        # support_img = []
        # for k in range(self.shot):
        #     support_img_one = Image.open(support_one[k]).convert('RGB')
        #     support_img_one = self.transform_x(support_img_one)
        #     support_img.append(support_img_one)
        
        if 'good' in query_one:
            y = 0
        else:
            y = 1
        
        return query_img, y

    def __len__(self):
        return len(self.query_dir)

    def load_dataset_folder(self):
        phase = 'test'
        query_dir, support_dir = [], []
        data_img = {}
        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            data_img[img_type] = []
            img_type_dir = os.path.join(img_dir, img_type)
            if ".DS_Store" not in os.path.basename(img_type_dir):
                img_num = sorted(os.listdir(img_type_dir))
                for img_one in img_num:
                    if img_type != ".DS_Store":
                        img_dir_one = os.path.join(img_type_dir, img_one)
                        data_img[img_type].append(img_dir_one)
        # img_dir_train = os.path.join(self.dataset_path, self.class_name, 'train', 'good')
        # img_num = sorted(os.listdir(img_dir_train))

        # data_train = []
        # for img_one in img_num:
        #     img_dir_one = os.path.join(img_dir_train, img_one)
        #     data_train.append(img_dir_one)

        for img_type in data_img.keys():
            for image_dir_one in data_img[img_type]:
                support_dir_one = []
                if not ("/.DS_Store" in image_dir_one):
                    query_dir.append(image_dir_one)
                    # for k in range(self.shot):
                    #     random_choose = random.randint(0, (len(data_train) - 1))
                    #     support_dir_one.append(data_train[random_choose])
                    # support_dir.append(support_dir_one)

        # assert len(query_dir) == len(support_dir), 'number of query_dir and support_dir should be same'
        return query_dir


class FSAD_Dataset_streamlit(Dataset):
    def __init__(self,
                dataset_path,
                is_train=True,
                resize=256,
                shot=2,
                 ):
        self.dataset_path = dataset_path
        self.resize = resize
        self.shot = shot
        self.query_dir = self.load_dataset_folder()

   
        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize((resize,224), Image.LANCZOS),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        query_one = self.query_dir[idx]
        query_img = Image.open(query_one).convert('RGB')
        query_img = self.transform_x(query_img)
        
        y = 1
        
        return query_img, y

    def __len__(self):
        return len(self.query_dir)

    
    def load_dataset_folder(self):  
        query_dir_paths = []
        query_dir="./visuals/inputs"
        for filename in os.listdir(query_dir):
            full_path = os.path.join(query_dir, filename)
            query_dir_paths.append(full_path)
        return query_dir_paths
