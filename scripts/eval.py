import argparse
import cv2
import numpy as np
import os
from typing import List
import gc

from tqdm import tqdm
import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance

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
    parser.add_argument(
        "--clip-model",
        type=str,
        default='openai/clip-vit-base-patch16',
        help="the clip model name to use for calculating clip metric"
    )
    parser.add_argument(
        "--ground-truth-img-dir",
        type=str,
        required=True,
        help="The directory where the ground truth images are stored"
    )
    return parser.parse_args()

def load_prompts(prompt_dir:str, batch_index:int):
    file_name = f"prompts_{batch_index}.txt"
    file_path = os.path.join(prompt_dir, file_name)
    with open(file_path, 'r') as f:
        prompts = f.readlines()
    return prompts

def load_images(dir:str):
    images = []
    img_files = os.listdir(dir)
    for img_name in tqdm(img_files):
        img_path = os.path.join(dir, img_name)
        img = cv2.imread(img_path)
        resize_img = cv2.resize(img, (224, 224))
        images.append(resize_img)
        del img
    return images

def calculate_clip_score(images, prompts, model_name):
    assert len(images) == len(prompts), "The number of images and prompts must be equal"
    metric = CLIPScore(model_name_or_path=model_name)
    clip_score = metric(images, prompts)
    return round(float(clip_score), 4)

def create_fid(real_imgs, gen_imgs):
    metric = FrechetInceptionDistance()
    assert len(real_imgs) == len(gen_imgs), "The number of generated and ground truth images must be equal"
    metric.update(real_imgs, real=True)
    metric.update(gen_imgs, real=False)
    return metric

def main(args):
    # total_score = 0
    # total_count = 0
    # for i in range(args.batch_count):
    #     prompts = load_prompts(args.prompt_dir, i)
    #     dir = os.path.join(args.generated_img_root_dir, f"prompts_{i}/samples")
    #     images = load_images(dir)
    #     batch_size = len(prompts)
    #     total_count += batch_size
    #     total_score += batch_size * calculate_clip_score(images, prompts, args.clip_model)
    #     del prompts
    #     del images
    #     gc.collect()
    # print(f"CLIP score: {total_score/total_count}")

    gen_images = []
    for i in range(args.batch_count):
        dir = os.path.join(args.generated_img_root_dir, f"prompts_{i}/samples")
        gen_images.extend(load_images(dir))
    real_images = load_images(args.ground_truth_img_dir)
    sh = real_images[0].shape
    print(sh)
    for real_image in real_images:
        if real_image.shape != sh:
            print(real_image.shape)
    print("loaded all images")

    metric = FrechetInceptionDistance()
    batch_size = 100
    for i in tqdm(range(0, len(real_images), batch_size)):
        real = np.asarray(real_images[i:i+batch_size])
        print(real.shape)
        real = torch.from_numpy(real)
        real = real.permute(0, 3, 1, 2)
        print(real.shape)
        metric.update(real, real=True)
        gen = np.asarray(gen_images[i:i+batch_size])
        print(gen.shape)
        gen = torch.from_numpy(gen)
        gen = gen.permute(0, 3, 1, 2)
        print(gen.shape)
        metric.update(gen, real=False)
    print(f"FID score: {metric.compute()}")

    

if __name__ == "__main__":
    args = parse_args()
    main(args)