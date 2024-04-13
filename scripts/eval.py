import argparse
import cv2
import numpy as np
import os
from typing import List
import json

from tqdm import tqdm
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore

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
        help="The number of batches of prompts and images"
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="The batch size for calculating metrics in batches",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Where to store the json metrics file",
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
    img_files = sorted(img_files)
    for img_name in tqdm(img_files):
        img_path = os.path.join(dir, img_name)
        img = cv2.imread(img_path)
        resize_img = cv2.resize(img, (224, 224))
        images.append(resize_img)
        del img
    return images


def calculate_clip_score(images, prompts, batch_size, model_name, device):
    assert len(images) == len(prompts), "The number of images and prompts must be equal"
    metric = CLIPScore(model_name_or_path=model_name)
    for i in tqdm(range(0, len(images), batch_size)):
        image_chunk = torch.from_numpy(
            np.asarray(images[i : i + batch_size]).astype("uint8")
        )
        prompt_chunk = prompts[i : i + batch_size]
        metric.update(image_chunk.permute(0, 3, 1, 2).to(device), prompt_chunk)
    return metric.compute()


def calculate_fid(real_images, gen_images, batch_size, device):
    assert len(real_images) == len(
        gen_images
    ), "The number of ground truth and generated images must be equal"
    metric = FrechetInceptionDistance()
    for i in tqdm(range(0, len(real_images), batch_size)):
        real = np.asarray(real_images[i : i + batch_size]).astype("uint8")
        real = torch.from_numpy(real)
        real = real.permute(0, 3, 1, 2).to(device)
        metric.update(real, real=True)
        gen = np.asarray(gen_images[i : i + batch_size]).astype("uint8")
        gen = torch.from_numpy(gen)
        gen = gen.permute(0, 3, 1, 2).to(device)
        metric.update(gen, real=False)
    return metric.compute()


def calculate_inception_score(images, batch_size, device):
    images = [cv2.resize(img, (299, 299)) for img in images]
    metric = InceptionScore()
    for i in tqdm(range(0, len(images), batch_size)):
        image_chunk = torch.from_numpy(
            np.asarray(images[i : i + batch_size]).astype("uint8")
        )
        metric.update(image_chunk.permute(0, 3, 1, 2).to(device))
    return metric.compute()


def main(args):
    device = "cpu"
    torch.set_default_device(device)
    print("Loading prompts")
    prompts = []
    for i in range(args.batch_count):
        prompts.extend(load_prompts(args.prompt_dir, i))
    print("Loading generated images")
    gen_images = []
    for i in range(args.batch_count):
        dir = os.path.join(args.generated_img_root_dir, f"prompts_{i}/samples")
        gen_images.extend(load_images(dir))
 
    print("Loading ground truth images")
    real_images = load_images(args.ground_truth_img_dir)
    assert len(real_images) == len(gen_images) == len(prompts)

    print("Calculating FID:")
    fid = calculate_fid(real_images, gen_images, args.batch_size, device)
    print("Calculating CLIPScore:")
    clip_score = calculate_clip_score(
        gen_images, prompts, args.batch_size, args.clip_model, device
    )
    # calculate inception score last since it resizes the generated images to 299x299
    print("Calculating IS:")
    IS = calculate_inception_score(gen_images, args.batch_size, device)
    metrics = {
        "FID": round(fid.detach().cpu().numpy().tolist(), 2),
        "CLIPScore": round(clip_score.detach().cpu().numpy().tolist(), 2),
        "IS": round(IS[0].detach().cpu().numpy().tolist(), 2),
        "IS_Standard_Deviation": round(IS[1].detach().cpu().numpy().tolist(), 2),
    }
    print(metrics.__repr__())
    with open(args.output_file, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
