import os
import argparse
import json

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torchvision
import torchfile
from PIL import Image
import clip

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
import clip
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random
from vdt_utils import read_split, read_json
import pandas as pd
from utils_nat import *

seed = 2103
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

cub_taxonomy_data = pd.read_csv("./assets/cub_taxonomy_2022.csv")
cub_taxonomy_data = cub_taxonomy_data.drop_duplicates(subset='cub_id')
nabirds_taxonomy_data = pd.read_csv("./assets/nabirds_taxonomy_2022.csv")
nabirds_taxonomy_data = nabirds_taxonomy_data.drop_duplicates(subset='nabirds_id')

BATCH_SIZE = 1024
EPOCH = 15


def ft_clip(opt):
    save_dir = opt.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_losses = []

    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model, preprocess = clip.load(opt.arch,device=device,jit=False) #Must set jit=False for training

    im_dir = opt.im_dir
    
    if opt.dataset == "CUB":
        with open('./assets/class_names_cub.txt') as f:
            all_classes = f.readlines()
        all_classes = [line.rstrip('\n') for line in all_classes]
        dataset = CUBImageLabelDataset(
            mode='train',
            class_range_train = np.arange(0, 100),
            all_classes = all_classes,
            im_dir = im_dir,
            desc_path_viz = opt.text_dir_viz,
            desc_path_loc = opt.text_dir_loc,
            preprocess = preprocess,
            taxonomy_data = cub_taxonomy_data,
            )
    elif opt.dataset == "Flowers102":
        with open('./assets/cat_to_name.json', 'r') as f:
            all_classes = json.load(f)
        dataset = FlowersImageLabelDataset(
            mode='train',
            preprocess = preprocess,
            all_classes = all_classes,
            im_dir = im_dir,
            desc_path_viz = opt.text_dir_viz,
            desc_path_loc = opt.text_dir_loc,
            class_range_train = np.arange(0,math.ceil(102 / 2)),
            )
    elif opt.dataset == "INaturalist21":
        with open(os.path.join(os.path.dirname(os.path.normpath(im_dir)), 'categories.json'), 'r') as f:
            inat_data = json.load(f)
        with open("./assets/inat_ids.txt", "r") as f:
            inat_data_id_subset = f.readlines()
        inat_data_id_subset = [int(id_i.rstrip('\n')) for id_i in inat_data_id_subset]
        dataset = INatImageLabelDataset(
            mode='train',
            inat_data = inat_data,
            inat_data_id_subset = inat_data_id_subset,
            im_dir = im_dir,
            desc_path_viz = opt.text_dir_viz,
            desc_path_loc = opt.text_dir_loc,
            preprocess = preprocess,
            )
    elif opt.dataset == "NABirds":
        parent_dir = os.path.dirname(os.path.normpath(im_dir))
        with open(os.path.join(parent_dir, 'species_names_and_ids.json'), 'r') as f:
            species_ids = json.load(f)

        children = {}
        with open(os.path.join(parent_dir, 'hierarchy.txt')) as f:
            lines_h = f.readlines()
        for line in lines_h:
            line = line.rstrip('\n')
            item, key = line.split(" ")
            if key not in children.keys():
                children[key] = [item.zfill(4)]
            else:
                children[key].append(item.zfill(4))

        cub_data_filtered = cub_taxonomy_data[cub_taxonomy_data['cub_id'] > 100]
        cub_scientific_names = cub_data_filtered['scientific_name'].tolist()
        nab_scientific_names = nabirds_taxonomy_data['scientific_name'].tolist()
        overlapping_names = set(nab_scientific_names).intersection(cub_scientific_names)
        ov_df = nabirds_taxonomy_data[nabirds_taxonomy_data['scientific_name'].isin(overlapping_names)]
        overlap_ids = ov_df['nabirds_id'].tolist()
        overlap_ids = [str(i) for i in overlap_ids]

        keys_filtered = list(set(species_ids.keys())-set(overlap_ids))

        dataset = NABirdsImageLabelDataset(
            mode='train',
            keys_filtered = keys_filtered,
            children = children,
            im_dir = im_dir,
            preprocess = preprocess,
            species_ids = species_ids,
            desc_path_viz = opt.text_dir_viz,
            desc_path_loc = opt.text_dir_loc,
            )
        
    all_classes = all_classes[0:math.ceil(len(all_classes) / 2)]

    train_dataloader = DataLoader(dataset,batch_size = BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=False) #Define your own dataloader
    temperature = nn.Parameter(torch.tensor(opt.tau))

    main_parameters = [param for name, param in model.named_parameters() if "proj" not in name and "text_projection" not in name]

    optimizer = optim.AdamW([
        {'params': main_parameters, 'lr': opt.main_lr, 'weight_decay' : opt.main_wd},
        {'params': [model.text_projection, model.visual.proj], 'lr': opt.proj_lr, 'weight_decay' : opt.proj_wd},
        {'params': temperature, 'lr': 1e-2, 'weight_decay' : 1e-6},
    ], betas=(0.9,0.98),eps=1e-6) 
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    for epoch in range(EPOCH):
        model.train()
        total_train_loss = 0
        train_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Training Epoch {epoch + 1}/{EPOCH}")
        for num_b, (images, class_ids, texts) in train_bar:
            optimizer.zero_grad()
            texts = clip.tokenize(texts)
            images = images.to(device)
            texts = texts.to(device)
            class_ids = class_ids.to(device)
            logits_per_image_unscaled, logits_per_text_unscaled = model(images, texts)
            logits_per_image = logits_per_image_unscaled/temperature
            logits_per_text = logits_per_text_unscaled/temperature
            list_probs_image = []
            for i in range(images.shape[0]):
                class_indx = class_ids[i,0]
                indices_ci = (class_ids[:, 0] == class_indx).nonzero(as_tuple=False)
                this_sm = logits_per_image[i,:].log_softmax(-1)
                list_probs_image.append(-this_sm.index_select(-1, indices_ci[:,0]).mean())

            list_probs_text = []
            for i in range(images.shape[0]):
                class_indx = class_ids[i,0]
                indices_ci = (class_ids[:, 0] == class_indx).nonzero(as_tuple=False)
                this_sm = logits_per_text[i,:].log_softmax(-1)
                list_probs_text.append(-this_sm.index_select(-1, indices_ci[:,0]).mean())

            gt_loss = torch.stack(list_probs_image).mean() + torch.stack(list_probs_text).mean()
            total_train_loss += gt_loss.item()
            gt_loss.backward()

            if device == "cpu":
                optimizer.step()
            else : 
                optimizer.step()

            train_bar.set_postfix(train_loss=(total_train_loss / (num_b + 1)))
        scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(temperature)    
        torch.save(model.state_dict(), os.path.join(save_dir,'model_'+str(epoch)+'.pth'))


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CUB', choices=['CUB', 'Flowers102', 'INaturalist21', 'NABirds'])
    parser.add_argument('--im_dir', type=str, required=True, help="dataset image directory")
    parser.add_argument('--save_dir', type=str, help="checkpoint saving path", default="./ft_clip")  
    parser.add_argument('--text_dir_viz', type=str, help="where generated visual gpt descriptions are saved", default="./gpt4_0613_api_CUB_viz")
    parser.add_argument('--text_dir_loc', type=str, help="where generated location gpt descriptions are saved", default="./gpt4_0613_api_CUB_loc")    
    parser.add_argument('--main_lr', type=float, help="main lr", default=5e-7)  
    parser.add_argument('--main_wd', type=float, help="main wd", default=1e-2) 
    parser.add_argument('--proj_lr', type=float, help="proj lr", default=1e-7)  
    parser.add_argument('--proj_wd', type=float, help="proj wd", default=1e-2)  
    parser.add_argument('--tau', type=float, help="temperature", default=2.0)  
    parser.add_argument('--fewshot', action='store_true', help="whether to train using 16 samples or full train set")
    parser.add_argument('--arch', type=str, help="vit architecture", default="ViT-B/32", choices=["ViT-B/16", "ViT-B/32"])
    opt = parser.parse_args()
    ft_clip(opt)