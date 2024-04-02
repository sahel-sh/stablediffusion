import argparse
import cv2
from functools import partial
import numpy as np
import os
from typing import List

from tqdm import tqdm
import torch
from torchmetrics.functional.multimodal import clip_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-dir",
        type=str,
        required=True,
        help="The directory containing txt prompt (caption) files, each file contains a subset of prompts and each prompt is in a separate line"
    )
    parser.add_argument(
        "--generated-img-root-dir",
        type=str,
        required=True,
        help="root dir containing subdirs, each subdir contains a subset of generated images"
    )
    parser.add_argument(
        "--batch-count",
        type=int,
        default=4,
        help="comma separated image dirs, each dir contains a subset of generated images"
    )
    return parser.parse_args()

def load_prompts(prompt_dir:str, batch_count:int):
    prompts = []
    for i in range(batch_count):
        file_name = f"prompts_{i}.txt"
        file_path = os.path.join(prompt_dir, file_name)
        with open(file_path, 'r') as f:
            prompts.extend(f.readlines())
    return prompts

def load_images(root_img_dir:str, batch_count:int):
    images = []
    for i in range(batch_count):
        dir = os.path.join(root_img_dir, f"prompts_{i}/samples")
        img_files = os.listdir(dir)
        for img_name in tqdm(img_files):
            img_path = os.path.join(dir, img_name)
            img = cv2.imread(img_path)
            images.append(img)
    return images

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
def calculate_clip_score(images, prompts):
    assert len(images) == len(prompts), "number of images and prompts should be equal"
    images = np.asarray(images).astype("uint8")
    print(images.shape)
    clip_score = clip_score_fn(torch.from_numpy(images).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

def main(args):
    prompts = load_prompts(args.prompt_dir, args.batch_count)
    print(len(prompts))
    images = load_images(args.generated_img_root_dir, args.batch_count)
    print(len(images))
    sd_clip_score = calculate_clip_score(images, prompts)
    print(f"CLIP score: {sd_clip_score}")

if __name__ == "__main__":
    args = parse_args()
    main(args)