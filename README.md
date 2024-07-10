# AdaptCLIPZS

This is the code-base for the 14 dataset benchmark for zero-shot classification proposed in

### Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions

[Oindrila Saha](http://oindrilasaha.github.io), [Grant Van Horn](https://gvh.codes), [Subhransu Maji](http://people.cs.umass.edu/~smaji/) 

CVPR'24

<h3 align="center">
  <a href="https://arxiv.org/abs/2401.02460">[arXiv]</a> | 
  <a href="https://cvl-umass.github.io/AdaptCLIPZS/">[Visualize the data]</a> |
    <a href="https://github.com/cvl-umass/AdaptCLIPZS/tree/main">[Project Page]</a> |
  <a href="https://github.com/cvl-umass/AdaptCLIPZS/blob/main/%E2%80%8EVisionGPT_poster.png">[Poster]</a> |
  <a href="https://www.youtube.com/watch?v=H-I0SFuRGxU">[Video]</a>
</h3>

![visiongptmethod](https://github.com/cvl-umass/AdaptCLIPZS/assets/20623465/7bdb90f6-8aaf-4091-991e-70302530e9da)


## Preparation

Create a conda environment with the specifications
```
conda env create -f environment.yml
conda activate adaptclipzs
```

Follow [DATASETS.md](https://github.com/mayug/VDT-Adapter/blob/main/DATASETS.md) of VDT-Adapter to download datasets and json files. Further download [iNaturalist21](https://github.com/visipedia/inat_comp/tree/master/2021), [NABirds](https://dl.allaboutbirds.org/nabirds), [CUB](https://www.vision.caltech.edu/datasets/cub_200_2011/) and [Flowers102](https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset) from these specified links.
Extract all images of CUB into a single folder by running:
```
cd <path to cub data>/images/ 
for folder in *; do; mv $folder/* ../images_extracted/.; done
```

## Generate attributes from OpenAI GPT (optional)

We provide our generated attributes for all datasets in "gpt_descriptions" folder. The folder contains folders for every dataset named in the format `<gpt_version>_<Dataset Name>`. Each of the dataset folder contains text files for each class named after the classname. You can also reproduce the process by running
```
python generate_gpt.py --api_key <your_api_key> --dataset StanfordCars --location --im_dir <path to directory containing images of StanfordCars> --json_file <path to json file of StanfordCars from VDT-Adapter> --gpt_version gpt4_0613_api
``` 

The above command will generate attributes for the StanfordCars dataset. The same command can be used to generate descriptions for all 14 datasets by changing the dataset, im_dir and json_file arguments. You do not need to provide json_file for CUB, NABirds and iNaturalist datasets. the location argument indicicates whether you want to generate attributes pertaining to where a certain category is found. We use this for natural domains in the paper i.e. CUB, NABirds. iNaturalist21 and Flowers102.

This will save the attributes in a folders named `<gpt_version>_<dataset>` inside AdaptCLIPZS.

## Fine-tuning CLIP

For non-natural domains run
```
python finetune_clip.py --dataset StanfordCars --im_dir <path to directory containing StanfordCars> --json_file <path to json file of StanfordCars from VDT-Adapter> --fewshot --arch ViT-B/16 --save_dir ./ft_clip_cars --text_dir ./gpt4_0613_api_StanfordCars
```

For natural domains i.e. CUB, iNaturalist, Flowers102 and NABirds run
```
python finetune_clip_nat.py --dataset CUB --im_dir <path to directory containing CUB> --fewshot --arch ViT-B/16 --save_dir ./ft_clip_cub --text_dir_viz ./gpt4_0613_api_CUB --text_dir_loc ./gpt4_0613_api_CUB_location
```

The fewshot argument indicates whether you want use 16 images per class for training or the whole dataset. You can also specify hyperparmeters including `main_lr, main_wd, proj_lr, proj_wd, tau`.


## Testing

Following command performs evaluation for CLIPFT+A setup

```
python test_AdaptZS.py --dataset StanfordCars --im_dir <path to directory containing images of StanfordCars> --json_file <path to json file of StanfordCars from VDT-Adapter> --fewshot --arch ViT-B/16 --ckpt_path <path to fine-tuned checkpoints> --text_dir ./gpt4_0613_api_StanfordCars --arch ViT-B/16 --attributes
```

For testing vanilla CLIP add --vanillaCLIP argument and for testing without GPT attributes omit --attributes. For natural domains also provide path to location attributes in text_dir_loc argument. 

## Pre-trained Checkpoints

We provide pre-trained checkpoints for iNaturist21, NABirds and CUB datasets for both ViT-B/16 and ViT-B/32 architectures, which can be downloaded [here](https://drive.google.com/drive/folders/1EGtnjHZSEUe-BY-v9r_5Zecbadv4E7vk?usp=share_link).

You can run the following command with pre-trained checkpoints to reproduce performance testing on CUB dataset.

```
python test_AdaptZS.py --im_dir <path to CUB extracted images> --ckpt_path ./INaturalist21_b16.pth --text_dir ./gpt_descriptions/gpt4_0613_api_CUB/ --text_dir_loc ./gpt_descriptions/gpt4_0613_api_CUB_location/ --arch ViT-B/16 --attributes
```
 You can modify the --ckpt_path with any of the other checkpoints making sure you provide the corresponding architecture in --arch. Following table shows the accuracies for the various checkpoints.
 
 Model | Accuracy
 --- | ---
 INaturalist21_b32.pth | 54.54
 INaturalist21_b16.pth | 56.76
 NABirds_b32.pth | 55.46
 NABirds_b16.pth | 56.59
 CUB_b32.pth | 54.23
 CUB_b16.pth | 56.01


## Citation
If you find our work useful, please consider citing:

```
@inproceedings{saha2024improved,
  title={Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions},
  author={Saha, Oindrila and Van Horn, Grant and Maji, Subhransu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17542--17552},
  year={2024}
}
```

Thanks to [CoOP](https://github.com/KaiyangZhou/CoOp) and [VDT-Adapter](https://github.com/mayug/VDT-Adapter) for releasing the code base which our code is built upon.
