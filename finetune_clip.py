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

seed = 2103
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


BATCH_SIZE = 1024
EPOCH = 15

CUSTOM_TEMPLATES = {
    'OxfordIIITPets': 'a photo of a {}, a type of pet',
    'Flowers102': 'a photo of a {}, a type of flower',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft',
    'DTD': '{} texture',
    'EuroSAT': 'a centered satellite photo of {}',
    'StanfordCars': 'a photo of a {}',
    'Food101': 'a photo of {}, a type of food',
    'Sun397': 'a photo of a {}',
    'CalTech101': 'a photo of a {}',
    'UCF101': 'a photo of a person doing {}',
    'ImageNet': 'a photo of a {}',
    'CUB': 'a photo of a {} bird',
}



def ft_clip(opt):
    save_dir = opt.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_losses = []
    val_losses = []

    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model, preprocess = clip.load(opt.arch,device=device,jit=False) #Must set jit=False for training

    im_dir = opt.im_dir
    train, val, test = read_split(opt.json_file, im_dir)
    all_classes = []
    labels = []
    for ob in test:
        if ob.classname not in all_classes:
            all_classes.append(ob.classname)
            labels.append(ob.label)

    all_classes = all_classes[0:math.ceil(len(all_classes) / 2)]

    transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                ])

    class ImageLabelDataset(Dataset):
        def __init__(
                self,
                mode,
                text_dir,
                img_size=(224, 224),
                classes_sublist=None
        ):
            self.img_path_list = []
            self.lbl_list = []
            for ob in train:
                if ob.classname in all_classes:
                    count = self.lbl_list.count(ob.classname)
                    if not opt.fewshot:
                        self.img_path_list.append(ob.impath)
                        self.lbl_list.append(ob.classname)
                    else:
                        if count < 16:
                            self.img_path_list.append(ob.impath)
                            self.lbl_list.append(ob.classname)
            self.classes_sublist = classes_sublist
            self.img_size = img_size
            self.mode = mode
            self.text_dir = text_dir

        def __len__(self):
            return len(self.img_path_list)

        def __getitem__(self, index):
            im_path = os.path.join(self.img_path_list[index])
            im = Image.open(im_path).convert('RGB')
            class_id = np.asarray([all_classes.index(self.lbl_list[index])])
            if self.mode == 'train':
                im = transform_train(im)
            im = preprocess(im)
            with open(os.path.join(self.text_dir,self.lbl_list[index]+'.txt')) as f:
                texts_class = f.readlines()
            texts_class = [CUSTOM_TEMPLATES[opt.dataset].format(self.lbl_list[index].replace("_", " ")) + " " + ' '.join(line.rstrip('\n').split(" ")[2:]) for line in texts_class if line.strip()]
            text_i = texts_class[np.random.randint(0,len(texts_class))]
            return im, torch.from_numpy(class_id), text_i

    dataset = ImageLabelDataset(mode='train', text_dir=opt.text_dir)
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
    parser.add_argument('--dataset', type=str, default='StanfordCars', choices=['StanfordCars', 'FGVCAircraft', 'Food101', 'ImageNet', 'EuroSAT', 'DTD', 'Sun397', 'UCF101', 'CalTech101', 'OxfordIIITPets'])
    parser.add_argument('--im_dir', type=str, required=True, help="dataset image directory")
    parser.add_argument('--json_file', type=str, required=True, help="dataset split json") 
    parser.add_argument('--save_dir', type=str, help="checkpoint saving path", default="./ft_clip")  
    parser.add_argument('--text_dir', type=str, help="where generated gpt descriptions are saved", default="./gpt4_0613_api_StanfordCars")  
    parser.add_argument('--main_lr', type=float, help="main lr", default=5e-7)  
    parser.add_argument('--main_wd', type=float, help="main wd", default=1e-2) 
    parser.add_argument('--proj_lr', type=float, help="proj lr", default=1e-7)  
    parser.add_argument('--proj_wd', type=float, help="proj wd", default=1e-2)  
    parser.add_argument('--tau', type=float, help="temperature", default=2.0)  
    parser.add_argument('--fewshot', action='store_true', help="whether to train using 16 samples or full train set")
    parser.add_argument('--arch', type=str, help="vit architecture", default="ViT-B/32", choices=["ViT-B/16", "ViT-B/32"])
    opt = parser.parse_args()
    ft_clip(opt)