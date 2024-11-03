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
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
import clip
import math
from vdt_utils import read_split, read_json
from utils_nat import *
import pandas as pd

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
    'CUB': 'a photo of a {} bird.',
}


def test(opt):
    im_dir = opt.im_dir
    if opt.dataset == "CUB":
        with open('./assets/class_names_cub.txt') as f:
            all_classes = f.readlines()
        all_classes = [line.rstrip('\n') for line in all_classes]
        all_classes = all_classes[100:]
        cub_taxonomy_data = pd.read_csv("./assets/cub_taxonomy_2022.csv")
        cub_taxonomy_data = cub_taxonomy_data.drop_duplicates(subset='cub_id')
    elif opt.dataset == "Flowers102":
        with open(os.path.join(im_dir,'cat_to_name.json'), 'r') as f:
            classes = json.load(f)
        all_classes = [
            v for k, v in sorted(classes.items(), key=lambda x: int(x[0]))
            if int(k) > math.ceil(102/ 2)
        ]
    else:
        train, val, test = read_split(opt.json_file, im_dir)
        all_classes = []
        labels = []
        for ob in test:
            if ob.classname not in all_classes:
                all_classes.append(ob.classname)
                labels.append(ob.label)

        all_classes = all_classes[math.ceil(len(all_classes) / 2):]

    class ImageLabelDataset(Dataset):
        def __init__(
                self,
                mode,
                img_size=(224, 224),
                classes_sublist=None
        ):
            self.img_path_list = []
            self.lbl_list = []
            for ob in test:
                if ob.classname in all_classes:
                    self.img_path_list.append(ob.impath)
                    self.lbl_list.append(ob.classname)
            self.classes_sublist = classes_sublist
            self.img_size = img_size
            self.mode = mode

        def __len__(self):
            return len(self.img_path_list)

        def __getitem__(self, index):
            im_path = os.path.join(self.img_path_list[index])
            im = Image.open(im_path).convert('RGB')
            class_id = np.asarray([all_classes.index(self.lbl_list[index])])
            im = self.transform(im)
            return im, torch.from_numpy(class_id)

        def transform(self, img):
            img = torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)(img)
            img = torchvision.transforms.CenterCrop(224)(img)
            img = transforms.ToTensor()(img)
            img = resnet_transform(img)
            return img
        
    resnet_transform = torchvision.transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])

    @torch.no_grad()
    def run(model, loader, texts, text, texts_dict):
        train_feat = []
        train_labels = []
        for (inp, target) in loader:
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            with torch.no_grad():
                logits_per_image, _ = model(inp, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            probs_classes = []
            for class_i in texts_dict.keys():
                texts_thisclass = texts_dict[class_i]
                probs_thisclass = []
                for text_j in texts_thisclass:
                    probs_thisclass.append(probs[:,texts.index(text_j)])
                probs_classes.append(np.stack(probs_thisclass, -1).mean(-1))
            probs_classes = np.stack(probs_classes, -1)
            
            train_feat.append(probs_classes.argmax(-1))
            train_labels.append(target.squeeze(-1))   
        train_feat = np.concatenate(train_feat,axis=0)
        train_labels = torch.cat(train_labels,dim=0).detach().cpu().numpy()
        return train_feat, train_labels


    cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(opt.arch, device=device)
    model.cuda()
    model.eval()
    
    if opt.dataset == "CUB":
        class_range_test = list(np.arange(100, 200))
        dataset_val = CUBImageLabelDatasetTest(mode='val', im_dir=im_dir, class_range_test=class_range_test)
    elif opt.dataset == "Flowers102":
        class_range_test = np.arange(math.ceil(102/ 2), 102)
        dataset_val = FlowersImageLabelDatasetTest(mode='val', im_dir=im_dir, class_range_test=class_range_test)
    else:
        dataset_val = ImageLabelDataset(mode='val', classes_sublist=None)
    
    val_loader = DataLoader(dataset_val, batch_size=256, shuffle=False, num_workers=8, pin_memory=False)
    
    texts = []
    texts_dict = {}
    for i, class_i in enumerate(all_classes):
        if opt.attributes:
            with open(os.path.join(opt.text_dir,class_i+".txt")) as f:
                texts_class = f.readlines()
            texts_class = [CUSTOM_TEMPLATES[opt.dataset].format(str(class_i).replace("_", " ")) + " " + ' '.join(line.rstrip('\n').split(" ")[2:]) for line in texts_class if line.strip()]
            texts.extend(texts_class)
            texts_dict[str(class_i).replace("_", " ")] = texts_class
            if opt.text_dir_loc != "":
                with open(os.path.join(opt.text_dir_loc,class_i+".txt")) as f:
                    texts_class_loc = f.readlines()
                texts_class_loc = [line.replace('"', '').replace("'", '') for line in texts_class_loc]
                texts_class_loc = [CUSTOM_TEMPLATES[opt.dataset].format(str(class_i).replace("_", " ")) + " " + ' '.join(line.rstrip('\n').split(" ")[2:]) for line in texts_class_loc if line.strip()]
                texts.extend(texts_class_loc)
                texts_dict[str(class_i).replace("_", " ")].extend(texts_class_loc)
            if opt.dataset == "CUB":
                texts.append("a photo of a " + str(class_i) + " bird, with scientific name " + str(get_scientific_name(i+100+1, cub_taxonomy_data)))
                texts.append("a photo of a " + str(class_i) + " bird, with family name " + str(get_family(i+100+1, cub_taxonomy_data)))
                texts.append("a photo of a " + str(class_i) + " bird, of the order " + str(get_order(i+100+1, cub_taxonomy_data)))
                texts_dict[str(class_i)].append("a photo of a " + str(class_i) + " bird, with scientific name " + str(get_scientific_name(i+100+1, cub_taxonomy_data)))
                texts_dict[str(class_i)].append("a photo of a " + str(class_i) + " bird, with family name " + str(get_family(i+100+1, cub_taxonomy_data)))
                texts_dict[str(class_i)].append("a photo of a " + str(class_i) + " bird, of the order " + str(get_order(i+100+1, cub_taxonomy_data)))
        else:
            texts_class = CUSTOM_TEMPLATES[opt.dataset].format(str(class_i).replace("_", " ")) + "."
            texts.append(texts_class)
            texts_dict[str(class_i).replace("_", " ")] = [texts_class]
        
    
    text = clip.tokenize(texts).to("cuda")
    if not opt.vanillaCLIP:
        state_dict = torch.load(opt.ckpt_path)
        model.load_state_dict(state_dict, strict=True)
    model.cuda()
    model.eval()
    val_feat, val_labels = run(model, val_loader, texts, text, texts_dict)
    print("Accuracy Val = "+str((val_feat==val_labels).mean()*100))



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CUB', choices=['StanfordCars', 'FGVCAircraft', 'Food101', 'ImageNet', 'EuroSAT', 'DTD', 'Sun397', 'UCF101', 'CalTech101', 'OxfordIIITPets', 'CUB', 'Flowers102'])
    parser.add_argument('--im_dir', type=str, required=True, help="dataset image directory")
    parser.add_argument('--json_file', type=str, help="dataset split json") 
    parser.add_argument('--ckpt_path', type=str, help="checkpoint path", default="./ft_clip/model_9.pth")  
    parser.add_argument('--text_dir', type=str, help="where generated gpt descriptions are saved", default="./gpt4_0613_api_CUB")  
    parser.add_argument('--text_dir_loc', type=str, help="where generated gpt descriptions of location are saved", default="")
    parser.add_argument('--arch', type=str, help="vit architecture", default="ViT-B/32", choices=["ViT-B/16", "ViT-B/32", "ViT-L/14"])
    parser.add_argument('--vanillaCLIP', action='store_true', help="if testing vanilla CLIP")
    parser.add_argument('--attributes', action='store_true', help="if testing with LLM attributes")
    opt = parser.parse_args()
    test(opt)
