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

We provide our generated attributes for all datasets in PLACEHOLDER. You can also reproduce the process by running
```
python generate_gpt.py --api_key <your_api_key> --dataset StanfordCars --location --im_dir <path to directory containing images of StanfordCars> --json_file <path to json file of StanfordCars from VDT-Adapter> --gpt_version gpt4_0613_api
``` 

The above command will generate attributes for the StanfordCars dataset. The same command can be used to generate descriptions for all 14 datasets by changing the dataset, im_dir and json_file arguments. You do not need to provide json_file for CUB, NABirds and iNaturalist datasets.

This will save the attributes in a folders named `<gpt_version>_<dataset>` inside AdaptCLIPZS.
