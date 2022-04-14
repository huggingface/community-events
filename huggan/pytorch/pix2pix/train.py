#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2022 Erik Linder-Norén and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions.

import argparse
import os
from pathlib import Path
import numpy as np
import time
import datetime
import sys
import tempfile

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomVerticalFlip
from torchvision.utils import save_image

from PIL import Image

from torch.utils.data import DataLoader

from modeling_pix2pix import GeneratorUNet, Discriminator

from datasets import load_dataset

from accelerate import Accelerator

import torch.nn as nn
import torch

from huggan.utils.hub import get_full_repo_name
from huggingface_hub import create_repo


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="huggan/facades", help="Dataset to use")
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--image_size", type=int, default=256, help="size of images for training")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument(
        "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the HuggingFace hub after training.",
        )
    parser.add_argument(
        "--model_name",
        required="--push_to_hub" in sys.argv,
        type=str,
        help="Name of the model on the hub.",
    )
    parser.add_argument(
        "--organization_name",
        required=False,
        default="huggan",
        type=str,
        help="Organization name to push to, in case args.push_to_hub is specified.",
    )
    return parser.parse_args(args=args)

# Custom weights initialization called on Generator and Discriminator
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def training_function(config, args):
    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu, mixed_precision=args.mixed_precision)

    os.makedirs("images/%s" % args.dataset, exist_ok=True)
    os.makedirs("saved_models/%s" % args.dataset, exist_ok=True)
    
    repo_name = get_full_repo_name(args.model_name, args.organization_name)
    if args.push_to_hub:
        if accelerator.is_main_process:
            repo_url = create_repo(repo_name, exist_ok=True)
    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 100

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, args.image_size // 2 ** 4, args.image_size // 2 ** 4)

    # Initialize generator and discriminator
    generator = GeneratorUNet()
    discriminator = Discriminator()

    if args.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (args.dataset, args.epoch)))
        discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (args.dataset, args.epoch)))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Configure dataloaders
    transform = Compose(
            [
                Resize((args.image_size, args.image_size), Image.BICUBIC),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def transforms(examples):
        # random vertical flip
        imagesA = []
        imagesB = []
        for imageA, imageB in zip(examples['imageA'], examples['imageB']):
            if np.random.random() < 0.5:
                imageA = Image.fromarray(np.array(imageA)[:, ::-1, :], "RGB")
                imageB = Image.fromarray(np.array(imageB)[:, ::-1, :], "RGB")
            imagesA.append(imageA)
            imagesB.append(imageB)  
        
        # transforms
        examples["A"] = [transform(image.convert("RGB")) for image in imagesA]
        examples["B"] = [transform(image.convert("RGB")) for image in imagesB]

        del examples["imageA"]
        del examples["imageB"]

        return examples

    dataset = load_dataset(args.dataset)
    transformed_dataset = dataset.with_transform(transforms)

    splits = transformed_dataset['train'].train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']

    dataloader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, num_workers=args.n_cpu)
    val_dataloader = DataLoader(val_ds, batch_size=10, shuffle=True, num_workers=1)

    def sample_images(batches_done, accelerator):
        """Saves a generated sample from the validation set"""
        batch = next(iter(val_dataloader))
        real_A = batch["A"]
        real_B = batch["B"]
        fake_B = generator(real_A)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        if accelerator.is_main_process:
            save_image(img_sample, "images/%s/%s.png" % (args.dataset, batches_done), nrow=5, normalize=True)

    generator, discriminator, optimizer_G, optimizer_D, dataloader, val_dataloader = accelerator.prepare(generator, discriminator, optimizer_G, optimizer_D, dataloader, val_dataloader)

    # ----------
    #  Training
    # ----------

    prev_time = time.time()

    for epoch in range(args.epoch, args.n_epochs):
        print("Epoch:", epoch)
        for i, batch in enumerate(dataloader):

            # Model inputs
            real_A = batch["A"]
            real_B = batch["B"]
            
            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), *patch), device=accelerator.device)
            fake = torch.zeros((real_A.size(0), *patch), device=accelerator.device)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)

            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            accelerator.backward(loss_G)

            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            accelerator.backward(loss_D)
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = args.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                % (
                    epoch,
                    args.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % args.sample_interval == 0:
                sample_images(batches_done, accelerator)

        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
            if accelerator.is_main_process:
                unwrapped_generator = accelerator.unwrap_model(generator)
                unwrapped_discriminator = accelerator.unwrap_model(discriminator)
                # Save model checkpoints
                torch.save(unwrapped_generator.state_dict(), "saved_models/%s/generator_%d.pth" % (args.dataset, epoch))
                torch.save(unwrapped_discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (args.dataset, epoch))

        # Optionally push to hub
        if args.push_to_hub:
            if accelerator.is_main_process:
                with tempfile.TemporaryDirectory() as temp_dir:
                    unwrapped_generator = accelerator.unwrap_model(generator)
                    unwrapped_generator.push_to_hub(
                        repo_path_or_name=temp_dir,
                        repo_url=repo_url,
                        commit_message=f"Training in progress, epoch {epoch}",
                        skip_lfs_files=True
                    )

def main():
    args = parse_args()
    print(args)

    training_function({}, args)


if __name__ == "__main__":
    main()
