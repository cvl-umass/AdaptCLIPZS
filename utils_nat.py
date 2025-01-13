from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch
import glob
import os
import numpy as np
import torchfile
import json
import torchvision


supercat_mapping = {"Amphibians" : "amphibian",
"Animalia" : "animal",
"Arachnids" : "arachnid",
"Birds" : "bird",
"Fungi" : "fungus",
"Insects" : "insect",
"Mammals" : "mammal",
"Mollusks" : "mollusk",
"Plants" : "plant",
"Ray-finned Fishes" : "ray-finned fish",
"Reptiles" : "reptile",
}

def get_scientific_name(cub_id, cub_taxonomy_data):
    """
    Return the scientific_name for a given cub_id from the cub_taxonomy_data dataframe.
    """
    try:
        row = cub_taxonomy_data[cub_taxonomy_data['cub_id'] == cub_id]
    except:
        row = cub_taxonomy_data[cub_taxonomy_data['nabirds_id'] == cub_id]
    if not row.empty:
        return row['scientific_name'].values[0]
    else:
        return "No data found for the given cub_id."

def get_family(cub_id, cub_taxonomy_data):
    """
    Return the family for a given cub_id from the cub_taxonomy_data dataframe.
    """
    try:
        row = cub_taxonomy_data[cub_taxonomy_data['cub_id'] == cub_id]
    except:
        row = cub_taxonomy_data[cub_taxonomy_data['nabirds_id'] == cub_id]
    if not row.empty:
        return row['family'].values[0]
    else:
        return "No data found for the given cub_id."

def get_order(cub_id, cub_taxonomy_data):
    """
    Return the order for a given cub_id from the cub_taxonomy_data dataframe.
    """
    try:
        row = cub_taxonomy_data[cub_taxonomy_data['cub_id'] == cub_id]
    except:
        row = cub_taxonomy_data[cub_taxonomy_data['nabirds_id'] == cub_id]
    if not row.empty:
        return row['order'].values[0]
    else:
        return "No data found for the given cub_id."
    
def get_genus(cub_id, cub_taxonomy_data):
    """
    Return the genus for a given cub_id from the cub_taxonomy_data dataframe.
    """
    try:
        row = cub_taxonomy_data[cub_taxonomy_data['cub_id'] == cub_id]
    except:
        row = cub_taxonomy_data[cub_taxonomy_data['nabirds_id'] == cub_id]
    if not row.empty:
        return row['genus'].values[0]
    else:
        return "No data found for the given cub_id."

transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            ]) 

resnet_transform = torchvision.transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])

class INatImageLabelDataset(Dataset):
    def __init__(
            self,
            mode,
            inat_data,
            inat_data_id_subset,
            im_dir,
            desc_path_viz,
            desc_path_loc,
            preprocess,
            img_size=(224, 224),
            classes_sublist=None
    ):
        self.img_path_list = []
        self.lbl_list = []
        for data_i in inat_data:
            if data_i['id'] not in inat_data_id_subset:
                continue
            # if data_i['supercategory'] == 'Birds':
            #     continue
            files_folder = glob.glob(os.path.join(im_dir,"train",data_i['image_dir_name'])+"/*")
            self.img_path_list.extend(files_folder)
            self.lbl_list.extend([data_i['id']]*len(files_folder))
        self.img_size = img_size
        self.mode = mode
        self.inat_data = inat_data
        self.preprocess = preprocess
        self.desc_path_viz = desc_path_viz
        self.desc_path_loc = desc_path_loc

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = os.path.join(self.img_path_list[index])
        im = Image.open(im_path).convert('RGB')
        class_id = np.asarray([self.lbl_list[index]])
        if self.mode == 'train':
            im = transform_train(im)
        im = self.preprocess(im)
        inat_dict = self.inat_data[self.lbl_list[index]]
        organism1 = inat_dict["common_name"]
        sn1 = inat_dict["name"]
        type1 = supercat_mapping[inat_dict["supercategory"]].lower()
        or1 = inat_dict["order"]
        fa1 = inat_dict["family"]
        organism1_sv = organism1.replace('/','SLASH')
        organism1_sv = organism1_sv + "_" + sn1
        with open(os.path.join(self.desc_path_viz, organism1_sv+'.txt')) as f:
            texts_class = f.readlines()
        texts_class = [text for text in texts_class if 'scientific name' not in text.lower()]

        texts_class = [text[4:] if text.lower().startswith('the ') else (text[3:] if text.lower().startswith('an ') else (text[2:] if text.lower().startswith('a ') else text)) for text in texts_class]

        texts_class = ["a photo of a " + organism1 + " " + line.rstrip('\n')[0].lower() + line.rstrip('\n')[1:] for line in texts_class if line.strip()]
        
        with open(os.path.join(self.desc_path_loc,organism1_sv+'.txt')) as f:
            texts_class_loc = f.readlines()
        texts_class_loc = [text for text in texts_class_loc if 'scientific name' not in text.lower()]
        texts_class_loc = [line.replace('"', '').replace("'", '') for line in texts_class_loc]
        texts_class_loc = [text[4:] if text.lower().startswith('the ') else (text[3:] if text.lower().startswith('an ') else (text[2:] if text.lower().startswith('a ') else text)) for text in texts_class_loc]

        texts_class_loc = ["a photo of a " + organism1 + " " + line.rstrip('\n')[0].lower() + line.rstrip('\n')[1:] for line in texts_class_loc if line.strip()]
        texts_class.extend(texts_class_loc)
        texts_class.append("a photo of a " + organism1 + " " + type1 + ", of the order " + or1)
        texts_class.append("a photo of a " + organism1 + " " + type1 + ", with family name " + fa1)
        texts_class.append("a photo of a " + organism1 + " " + type1 + ", with scientific name " + sn1)

        text_i = texts_class[np.random.randint(0,len(texts_class))]
        if len(text_i.split())>30:
            text_i = min((text_i.split(delim)[0] for delim in ";:,"), key=len) + "."
        return im, torch.from_numpy(class_id), text_i
    
class CUBImageLabelDataset(Dataset):
    def __init__(
            self,
            mode,
            class_range_train,
            all_classes,
            im_dir,
            desc_path_viz,
            desc_path_loc,
            preprocess,
            taxonomy_data,
            img_size=(224, 224),
            classes_sublist=None
    ):
        datfile = torchfile.load('./assets/train.dat')
        self.class_range = class_range_train
        with open('./assets/cub_classes.json', 'r') as f:
            self.class_list = json.load(f)
        self.img_path_list = []
        for name, _ in datfile.items():
            if name.decode() == 'Black_Tern_0079_143998.jpg':
                continue
            if self.class_list[name.decode()]-1 in self.class_range:
                self.img_path_list.append(name.decode())
        self.img_size = img_size
        self.classes_sublist = classes_sublist
        self.im_dir = im_dir
        self.mode = mode
        self.preprocess = preprocess
        self.all_classes = all_classes
        self.desc_path_viz = desc_path_viz
        self.desc_path_loc = desc_path_loc
        self.taxonomy_data = taxonomy_data

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = os.path.join(self.im_dir, "images_extracted" ,self.img_path_list[index])
        im = Image.open(im_path).convert('RGB')
        if self.mode == 'train':
            im = transform_train(im)
        im = self.preprocess(im)
        if self.classes_sublist!=None:
            class_id = np.asarray([self.classes_sublist.index(self.class_list[self.img_path_list[index]]-1)])
        else:
            class_id = np.asarray([self.class_list[self.img_path_list[index]]-1])

        with open(os.path.join(self.desc_path_viz, self.all_classes[class_id[0]]+'.txt')) as f:
            texts_class_i = f.readlines()
        texts_class = ["a photo of a " + str(self.all_classes[class_id[0]]) + line.rstrip('\n')[1:] for line in texts_class_i if line.strip()]
        with open(os.path.join(self.desc_path_loc, self.all_classes[class_id[0]]+'.txt')) as f:
            texts_class_i = f.readlines()
        texts_class_i = [line.replace('"', '').replace("'", '') for line in texts_class_i]
        texts_class.extend("a photo of a " + str(self.all_classes[class_id[0]]) + line.rstrip('\n')[1:] for line in texts_class_i if line.strip())
        texts_class.append("a photo of a " + str(self.all_classes[class_id[0]]) + " bird, with scientific name " + str(get_scientific_name(class_id[0]+1, self.taxonomy_data)))
        texts_class.append("a photo of a " + str(self.all_classes[class_id[0]]) + " bird, with family name " + str(get_family(class_id[0]+1, self.taxonomy_data)))
        texts_class.append("a photo of a " + str(self.all_classes[class_id[0]]) + " bird, of the order " + str(get_order(class_id[0]+1, self.taxonomy_data)))
        text_i = texts_class[np.random.randint(0,len(texts_class))]
        # print(self.img_path_list[index], text_i)
        return im, torch.from_numpy(class_id), text_i
    
class FlowersImageLabelDataset(Dataset):
    def __init__(
            self,
            mode,
            preprocess,
            all_classes,
            im_dir,
            desc_path_viz,
            desc_path_loc,
            class_range_train,
            img_size=(224, 224),
            classes_sublist=None
    ):
        self.im_dir = os.path.join(im_dir, 'train')
        self.class_range = class_range_train
        self.label_list = []
        self.img_path_list = []
        for idx in self.class_range:
            files_folder = glob.glob(os.path.join(self.im_dir,str(idx+1)+"/*"))
            self.img_path_list.extend(files_folder)
            self.label_list.extend([idx+1]*len(files_folder))
        self.img_size = img_size
        self.classes_sublist = classes_sublist
        self.mode = mode
        self.preprocess = preprocess
        self.all_classes = all_classes
        self.desc_path_viz = desc_path_viz
        self.desc_path_loc = desc_path_loc

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = os.path.join(self.img_path_list[index])
        im = Image.open(im_path).convert('RGB')
        class_id = np.asarray([self.label_list[index]])
        if self.mode == 'train':
            im = transform_train(im)
        im = self.preprocess(im)

        with open(os.path.join(self.desc_path_viz, self.all_classes[str(self.label_list[index])]+'.txt')) as f:
            texts_class_i = f.readlines()
        texts_class = ["a photo of a " + self.all_classes[str(self.label_list[index])] + ", a type of" + line.rstrip('\n')[1:] for line in texts_class_i if line.strip()]
        with open(os.path.join(self.desc_path_loc, self.all_classes[str(self.label_list[index])]+'.txt')) as f:
            texts_class_i = f.readlines()
        texts_class_i = [line.replace('"', '').replace("'", '') for line in texts_class_i]
        texts_class.extend("a photo of a " + self.all_classes[str(self.label_list[index])] + ", a type of" + line.rstrip('\n')[1:] for line in texts_class_i if line.strip())
        text_i = texts_class[np.random.randint(0,len(texts_class))]
        return im, torch.from_numpy(class_id), text_i
    

class NABirdsImageLabelDataset(Dataset):
    def __init__(
            self,
            mode,
            keys_filtered,
            children,
            im_dir,
            preprocess,
            species_ids,
            desc_path_viz,
            desc_path_loc,
            taxonomy_data,
            img_size=(224, 224),
            classes_sublist=None
    ):
        self.img_path_list = []
        self.id_list = []
        self.class_label_list = []
        for ind_k, class_id in enumerate(keys_filtered):
            for child_id in children[class_id]:
                folder_path = os.path.join(im_dir, child_id)
                files_folder = glob.glob(folder_path+"/*")
                self.img_path_list.extend(files_folder)
                self.id_list.extend([class_id]*len(files_folder))
                self.class_label_list.extend([ind_k]*len(files_folder))
            
        self.img_size = img_size
        self.mode = mode
        self.preprocess = preprocess
        self.species_ids = species_ids
        self.keys_filtered = keys_filtered
        self.desc_path_viz = desc_path_viz
        self.desc_path_loc = desc_path_loc
        self.taxonomy_data = taxonomy_data

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = os.path.join(self.img_path_list[index])
        im = Image.open(im_path).convert('RGB')
        class_id = np.asarray([self.class_label_list[index]]).astype(int)
        real_id = np.asarray([self.id_list[index]]).astype(int)
        if self.mode == 'train':
            im = transform_train(im)
        im = self.preprocess(im)
        with open(os.path.join(self.desc_path_viz, self.species_ids[self.keys_filtered[class_id[0]]]+'.txt')) as f:
            texts_class = f.readlines()
        texts_class = ["a photo of a " + str(self.species_ids[self.keys_filtered[class_id[0]]]) + line.rstrip('\n')[1:] for line in texts_class if line.strip()]

        with open(os.path.join(self.desc_path_loc, self.species_ids[self.keys_filtered[class_id[0]]]+'.txt')) as f:
            texts_class_loc = f.readlines()
        texts_class_loc = [line.replace('"', '').replace("'", '') for line in texts_class_loc]
        texts_class_loc = ["a photo of a " + str(self.species_ids[self.keys_filtered[class_id[0]]]) + line.rstrip('\n')[1:] for line in texts_class_loc if line.strip()]
        texts_class.extend(texts_class_loc)

        texts_class.append("a photo of a " + str(self.species_ids[self.keys_filtered[class_id[0]]]) + " bird, with scientific name " + str(get_scientific_name(real_id[0], self.taxonomy_data)))
        texts_class.append("a photo of a " + str(self.species_ids[self.keys_filtered[class_id[0]]]) + " bird, with family name " + str(get_family(real_id[0], self.taxonomy_data)))
        texts_class.append("a photo of a " + str(self.species_ids[self.keys_filtered[class_id[0]]]) + " bird, of the order " + str(get_order(real_id[0], self.taxonomy_data)))
        text_i = texts_class[np.random.randint(0,len(texts_class))]
        
        return im, torch.from_numpy(class_id), text_i

class CUBImageLabelDatasetTest(Dataset):
    def __init__(
            self,
            mode,
            im_dir,
            class_range_test,
            img_size=(224, 224),
            classes_sublist=None
    ):
        if mode == 'train':
            datfile = torchfile.load('./assets/train.dat')
            self.class_range = class_range_test
        else:
            datfile = torchfile.load('./assets/val.dat')
            self.class_range = class_range_test
        with open('./assets/cub_classes.json', 'r') as f:
            self.class_list = json.load(f)
        self.img_path_list = []
        self.lbl_list = []
        for name, _ in datfile.items():
            if name.decode() == 'Black_Tern_0079_143998.jpg':
                continue
            if self.class_list[name.decode()]-1 in self.class_range:
                self.img_path_list.append(name.decode())
                self.lbl_list.append(self.class_list[name.decode()]-1)
        self.img_size = img_size
        self.classes_sublist = classes_sublist
        self.mode = mode
        self.im_dir = im_dir

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = os.path.join(self.im_dir, self.img_path_list[index])
        im = Image.open(im_path).convert('RGB')
        im = self.transform(im)
        return im, torch.from_numpy(np.array([self.class_range.index(self.lbl_list[index])]))

    def transform(self, img):
        im_shape = (min(int(img.size[0]*0.875), int(img.size[1]*0.875)), min(int(img.size[0]*0.875), int(img.size[1]*0.875)))
        img = torchvision.transforms.CenterCrop(im_shape)(img) 
        img = img.resize(self.img_size)     
        img = transforms.ToTensor()(img)
        img = resnet_transform(img)
        return img
    
class FlowersImageLabelDatasetTest(Dataset):
    def __init__(
            self,
            mode,
            im_dir,
            class_range_test,
            img_size=(224, 224),
            classes_sublist=None
    ):
        if mode == 'train':
            self.im_dir = os.path.join(im_dir, 'train')
            self.class_range = class_range_test
        elif mode == 'val':
            self.im_dir = os.path.join(im_dir, 'valid')
            self.class_range = class_range_test
        elif mode == 'test':
            self.im_dir = os.path.join(im_dir, 'test')
            self.class_range = class_range_test
        self.label_list = []
        self.img_path_list = []
        for idx in self.class_range:
            files_folder = glob.glob(os.path.join(self.im_dir,str(idx+1)+"/*"))
            self.img_path_list.extend(files_folder)
            self.label_list.extend([idx]*len(files_folder))
        self.img_size = img_size
        self.classes_sublist = classes_sublist
        self.mode = mode

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = os.path.join(self.img_path_list[index])
        im = Image.open(im_path).convert('RGB')
        class_id = np.asarray([self.label_list[index] - self.class_range.min()])
        im = self.transform(im)
        return im, torch.from_numpy(class_id)

    def transform(self, img):
        im_shape = (min(int(img.size[0]*0.875), int(img.size[1]*0.875)), min(int(img.size[0]*0.875), int(img.size[1]*0.875)))
        img = torchvision.transforms.CenterCrop(im_shape)(img) 
        img = img.resize(self.img_size)     
        img = transforms.ToTensor()(img)
        img = resnet_transform(img)
        return img