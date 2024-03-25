import os
import openai
import re
import time
from vdt_utils import read_split, read_json
import json

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

def generate_descriptions(opt):
    version = opt.gpt_version

    out_dir = version + "_" + opt.dataset
    os.makedirs(out_dir, exist_ok=True)
    openai.api_key = opt.api_key

    im_dir = opt.im_dir

    _, _, test = read_split(opt.json_file, im_dir)
    classes = []
    labels = []
    prompt_dict = read_json("./assets/gpt_prompts.json")
    for ob in test:
        if ob.classname not in classes:
            classes.append(ob.classname)
            labels.append(ob.label)

    if opt.dataset == "INaturalist21":
        with open('./assets/categories_inat.json', 'r') as file:
            data_cat = json.load(file)

    if opt.dataset == "NABirds":
        with open("./assets/modified_species_names_and_ids_NABirds.txt", "r") as file:
            classes = file.readlines()
        classes = [line.rstrip('\n') for line in classes]
        classes = [line.split(' ', 1) for line in classes]

    if opt.dataset == "CUB":
        with open("./assets/class_names_cub.txt", "r") as file:
            classes = file.readlines()
        classes = [line.rstrip('\n') for line in classes]

    if opt.dataset == "INaturalist21":
        for item in data_cat:
            organism1 = item["common_name"]
            sn1 = item["name"]
            type1 = supercat_mapping[item["supercategory"]]
            type1 = type1.lower()
            if opt.location:
                base_prompt = prompt_dict[opt.dataset]["prompt_location"]
                system_content = prompt_dict[opt.dataset]["system_content_location"]
            else:
                base_prompt = prompt_dict[opt.dataset]["prompt_visual"]
                system_content = prompt_dict[opt.dataset]["system_content"]
                
            prompt = base_prompt.replace("<Item1>", organism1.lower()).replace("<type1>", type1).replace("<sn1>", sn1)
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4-0613",
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )

                if 'choices' in response and len(response['choices']) > 0:
                    result = response['choices'][0]['message']['content']
                    result = re.sub(r'^\d+\.\s', '', result, flags=re.M)
                    item1_sv = organism1.replace('/','SLASH')
                    item1_sv = item1_sv + "_" + sn1
                    with open(f'{out_dir}/{item1_sv}.txt', 'w') as f:
                        f.write(result)
                else:
                    print(f"Unexpected response for {item}: {response}")

            except Exception as e:
                print(f"Error generating details for {item}: {e}")
            print(item1_sv)
            time.sleep(0.02)

    else:
        for item in classes:
            if os.path.exists(f'{out_dir}/{item}.txt'):
                continue            

            if opt.location:
                base_prompt = prompt_dict[opt.dataset]["prompt_location"]
                system_content = prompt_dict[opt.dataset]["system_content_location"]
            else:
                base_prompt = prompt_dict[opt.dataset]["prompt_visual"]
                system_content = prompt_dict[opt.dataset]["system_content"]

            prompt = base_prompt.replace("<Item1>", item.lower())
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4-0613",
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )

                if 'choices' in response and len(response['choices']) > 0:
                    result = response['choices'][0]['message']['content']
                    result = re.sub(r'^\d+\.\s', '', result, flags=re.M)
                    item1_sv = item.replace('/','SLASH')
                    with open(f'{out_dir}/{item1_sv}.txt', 'w') as f:
                        f.write(result)
                else:
                    print(f"Unexpected response for {item}: {response}")

            except Exception as e:
                print(f"Error generating details for {item}: {e}")
            print(item1_sv)
            time.sleep(0.02) 

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', required=True, type=str)
    parser.add_argument('--dataset', type=str, default='CUB', choices=['CUB', 'StanfordCars', 'FGVCAircraft', 'INaturalist21', 'NABirds', 'Flowers102', 'Food101', 'ImageNet', 'EuroSAT', 'DTD', 'Sun397', 'UCF101', 'CalTech101', 'OxfordIIITPets'])
    parser.add_argument('--location', action='store_true', help="generate location texts and not visual")
    parser.add_argument('--im_dir', type=str, required=True, help="dataset image directory")
    parser.add_argument('--json_file', type=str, required=True, help="dataset split json")  
    parser.add_argument('--gpt_version', type=str, default="gpt4_0613_api")    
    opt = parser.parse_args()
    generate_descriptions(opt)




