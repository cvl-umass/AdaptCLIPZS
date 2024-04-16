# AdaptCLIPZS

This the code-base for the 14 dataset benchmark for zero-shot classification proposed in

#### Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions

[Oindrila Saha](http://oindrilasaha.github.io), [Grant Van Horn](https://gvh.codes), [Subhransu Maji](http://people.cs.umass.edu/~smaji/) 

CVPR'24

[[arXiv]](https://arxiv.org/abs/2401.02460)

## Preparation

Create a conda environment with the specifications
```
conda env create -f environment.yml
conda activate adaptclipzs
```

Follow [DATASETS.md](https://github.com/mayug/VDT-Adapter/blob/main/DATASETS.md) of VDT-Adapter to download datasets and json files. Further download [iNaturalist21](https://github.com/visipedia/inat_comp/tree/master/2021) and [NABirds](https://dl.allaboutbirds.org/nabirds).

## Generate attributes from OpenAI GPT4

We provide our generated attributes for all datasets in "gpt_descriptions" folder. You can also reproduce the process by running
```
python generate_gpt.py --api_key <your_api_key> --dataset StanfordCars --location --im_dir <path to directory containing images of StanfordCars> --json_file <path to json file of StanfordCars from VDT-Adapter> --gpt_version gpt4_0613_api
``` 

The above command will generate attributes for the StanfordCars dataset. The same command can be used to generate descriptions for all 14 datasets by changing the dataset, im_dir and json_file arguments. You do not need to provide json_file for CUB, NABirds and iNaturalist datasets. the location argument indicicates whether you want to generate attributes pertaining to where a certain category is found. We use this for natural domains in the paper i.e. CUB, NABirds. iNaturalist21 and Flowers102.

This will save the attributes in a folders named `<gpt_version>_<dataset>` inside AdaptCLIPZS.

## Fine-tuning CLIP

For non-natural domains run
```
python finetune_clip.py --dataset StanfordCars --im_dir <path to directory containing images of StanfordCars> --json_file <path to json file of StanfordCars from VDT-Adapter> --fewshot --arch ViT-B/16 --save_dir ./ft_clip_cars --text_dir ./gpt4_0613_api_StanfordCars
```

For natural domains i.e. CUB, iNaturalist, Flowers102 and NABirds run
```
python finetune_clip.py --dataset CUB --im_dir <path to directory containing images of CUB> --fewshot --arch ViT-B/16 --save_dir ./ft_clip_cub --text_dir_viz ./gpt4_0613_api_CUB --text_dir_loc ./gpt4_0613_api_CUB_location
```

The fewshot argument indicates whether you want use 16 images per class for training or the whole dataset. You can also specify hyperparmeters including `main_lr, main_wd, proj_lr, proj_wd, tau`.


## Testing

Following command performs evaluation for CLIPFT+A setup

```
python test_AdaptZS.py --dataset StanfordCars --im_dir <path to directory containing images of StanfordCars> --json_file <path to json file of StanfordCars from VDT-Adapter> --fewshot --arch ViT-B/16 --ckpt_dir <path to folder containing fine-tuning checkpoints> --text_dir ./gpt4_0613_api_StanfordCars --arch ViT-B/16 --vanillaCLIP False --attributes True
```

For testing vanilla CLIP set vanillaCLIP argument to True and for testing without GPT attributes set attributes to False. For natural domains also provide path to location attributes in text_dir_loc argument. 


## Citation
If you use this code for your research, please cite the following paper.

```
@article{saha2024improved,
  title={Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions},
  author={Saha, Oindrila and Horn, Grant Van and Maji, Subhransu},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

Thanks to [CoOP](https://github.com/KaiyangZhou/CoOp) and [VDT-Adapter](https://github.com/mayug/VDT-Adapter) for releasing the code base which our code is built upon.
