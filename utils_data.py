from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch
import glob
import os
import numpy as np
import xml.etree.ElementTree as ET

transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            ]) 

resnet_transform = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])

class AircraftTrainDataset(Dataset):
    def __init__(
            self,
            class_range_train,
            all_classes,
            data_dir,
            desc_path,
            fewshot,
            preprocess,
            img_size=(224, 224),
            classes_sublist=None
    ):
       
        with open(os.path.join(data_dir,"fgvc-aircraft-2013b/data/images_variant_train.txt")) as f:
            data = f.readlines()
        self.class_range = class_range_train
        data = [line.rstrip('\n') for line in data]
        self.img_path_list = []
        self.lbl_list = []
        for data_i in data:
            if data_i.split(" ", 1)[1] in np.array(all_classes)[self.class_range].tolist():
                self.img_path_list.append(data_i.split(" ", 1)[0]+'.jpg')
                self.lbl_list.append(data_i.split(" ", 1)[1])
        self.img_size = img_size
        self.classes_sublist = classes_sublist
        self.im_dir = os.path.join(data_dir,"fgvc-aircraft-2013b/data/images")
        self.preprocess = preprocess
        self.all_classes = all_classes
        self.desc_path = desc_path

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = os.path.join(self.im_dir, self.img_path_list[index])
        im = Image.open(im_path).convert('RGB')
        class_id = np.asarray([self.all_classes.index(self.lbl_list[index])])

        im = transform_train(im)
        im = self.preprocess(im)

        with open(os.path.join(self.desc_path, self.lbl_list[index].replace('/','SLASH')+'.txt')) as f:
            texts_class = f.readlines()
        texts_class = ["a photo of a " + self.lbl_list[index] + ", a type of" + line.rstrip('\n')[2:] for line in texts_class if line.strip()]
        text_i = texts_class[np.random.randint(0,len(texts_class))]
        return im, torch.from_numpy(class_id), text_i
    

class ImageNetTrainDataset(Dataset):
    def __init__(
            self,
            class_range_train,
            all_classes_ids,
            all_classes_names,
            data_dir,
            desc_path,
            fewshot,
            preprocess,
            img_size=(224, 224),
            classes_sublist=None
    ):
        self.im_dir = os.path.join(data_dir, 'ILSVRC/Data/CLS-LOC/train')
        self.class_range = class_range_train
        
        self.label_list = []
        self.img_path_list = []
        for idx in self.class_range:
            if fewshot:
                files_folder = glob.glob(os.path.join(self.im_dir,all_classes_ids[idx]+"/*"))[0:16]
            else:
                files_folder = glob.glob(os.path.join(self.im_dir,all_classes_ids[idx]+"/*"))
            self.img_path_list.extend(files_folder)
            self.label_list.extend([idx]*len(files_folder))
        self.img_size = img_size
        self.classes_sublist = classes_sublist
        self.all_classes_ids = all_classes_ids
        self.all_classes_names = all_classes_names
        self.desc_path = desc_path
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = os.path.join(self.img_path_list[index])
        im = Image.open(im_path).convert('RGB')
        class_id = np.asarray([self.label_list[index]])
        im = transform_train(im)
        im = self.preprocess(im)

        with open(os.path.join(self.desc_path, self.all_classes_ids[self.label_list[index]]+'.txt')) as f:
            texts_class = f.readlines()
        texts_class = ["a photo of a " + self.all_classes_names[self.label_list[index]] + line.rstrip('\n')[9:] for line in texts_class if line.strip()]
        text_i = texts_class[np.random.randint(0,len(texts_class))]
        if len(text_i.split())>30:
            text_i = min((text_i.split(delim)[0] for delim in ";:,"), key=len) + "."
        return im, torch.from_numpy(class_id), text_i
    
class ImageNetTestDataset(Dataset):
    def __init__(
            self,
            all_classes_ids,
            data_dir,
            img_size=(224, 224),
            classes_sublist=None
    ):
        self.im_dir = os.path.join(data_dir, 'ILSVRC/Data/CLS-LOC/val')
        self.label_list = []
        self.img_path_list = []
        xml_dir = os.path.join(data_dir, 'ILSVRC/Annotations/CLS-LOC/val/')

        self.all_classes_ids = all_classes_ids
        files_folder = glob.glob(self.im_dir+"/*")
        for file_i in files_folder:            
            tree = ET.parse(os.path.join(xml_dir, file_i.split("/")[-1][:-4]+"xml"))
            root = tree.getroot()
            name_element = root.find('.//name')
            name_text = name_element.text if name_element is not None else "None"
            if name_text not in all_classes_ids:
                continue
            self.img_path_list.append(file_i)
            self.label_list.append(name_text)
        self.img_size = img_size
        self.classes_sublist = classes_sublist

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = os.path.join(self.img_path_list[index])
        im = Image.open(im_path).convert('RGB')
        class_id = np.asarray([self.all_classes_ids.index(self.label_list[index])])
        im = self.transform(im)
        # import pdb;pdb.set_trace()
        return im, torch.from_numpy(np.asarray(class_id))
    
    def transform(self, img):
        im_shape = (min(int(img.size[0]*0.995), int(img.size[1]*0.995)), min(int(img.size[0]*0.995), int(img.size[1]*0.995)))
        img = transforms.CenterCrop(im_shape)(img) 
        img = img.resize(self.img_size)     
        img = transforms.ToTensor()(img)
        img = resnet_transform(img)
        return img
    

class AircraftTestDataset(Dataset):
    def __init__(
            self,
            all_classes,
            data_dir,
            img_size=(224, 224),
            classes_sublist=None
    ):
        with open(os.path.join(data_dir, "fgvc-aircraft-2013b/data/images_variant_test.txt")) as f:
            data = f.readlines()
        data = [line.rstrip('\n') for line in data]
        self.img_path_list = []
        self.lbl_list = []
        for data_i in data:
            if data_i.split(" ", 1)[1] in np.array(all_classes).tolist():
                self.img_path_list.append(data_i.split(" ", 1)[0]+'.jpg')
                self.lbl_list.append(data_i.split(" ", 1)[1])
        self.img_size = img_size
        self.im_dir = os.path.join(data_dir, "fgvc-aircraft-2013b/data/images")
        self.all_classes = all_classes

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = os.path.join(self.im_dir, self.img_path_list[index])
        im = Image.open(im_path).convert('RGB')
        class_id = np.asarray([self.all_classes.index(self.lbl_list[index])])
        im = self.transform(im)
        return im, torch.from_numpy(class_id)

    def transform(self, img):
        im_shape = (min(int(img.size[0]*0.995), int(img.size[1]*0.995)), min(int(img.size[0]*0.995), int(img.size[1]*0.995)))
        img = transforms.CenterCrop(im_shape)(img) 
        img = img.resize(self.img_size)     
        img = transforms.ToTensor()(img)
        img = resnet_transform(img)
        return img