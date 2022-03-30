#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2022 PyTorch contributors and The HuggingFace Inc. team. All rights reserved.
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

""" Training a Deep Convolutional Generative Adversarial Network (DCGAN) leveraging the 🤗 ecosystem.
Paper: https://arxiv.org/abs/1511.06434.
Based on PyTorch's official tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html.
"""


import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor, ToPILImage)
from torchvision.utils import save_image

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from accelerate import Accelerator

from modeling_dcgan import Discriminator, Generator

from datasets import load_dataset

from huggan.pytorch.metrics.inception import InceptionV3
from huggan.pytorch.metrics.fid_score import calculate_fretchet

import wandb

logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to load from the HuggingFace hub.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers when loading data")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size to use during training")
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Spatial size to use when resizing images for training.",
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=3,
        help="Number of channels in the training images. For color images this is 3.",
    )
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimensionality of the latent space.")
    parser.add_argument(
        "--generator_hidden_size",
        type=int,
        default=64,
        help="Hidden size of the generator's feature maps.",
    )
    parser.add_argument(
        "--discriminator_hidden_size",
        type=int,
        default=64,
        help="Hidden size of the discriminator's feature maps.",
    )
    parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs of training")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.5,
        help="adam: decay of first order momentum of gradient",
    )
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
    parser.add_argument("--output_dir", type=Path, default=Path("./output"), help="Name of the directory to dump generated images during training.")
    parser.add_argument("--wandb", action="store_true", help="If passed, will log to Weights and Biases.")
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Number of steps between each logging",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the HuggingFace hub after training.",
        )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="Name of the model on the hub.",
    )
    parser.add_argument(
        "--organization_name",
        default="huggan",
        type=str,
        help="Organization name to push to, in case args.push_to_hub is specified.",
    )
    args = parser.parse_args()
    
    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."
        assert args.model_name is not None, "Need a `model_name` to create a repo when `--push_to_hub` is passed."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    return args


# Custom weights initialization called on Generator and Discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def training_function(config, args):

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu, mixed_precision=args.mixed_precision)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        # set up Weights and Biases if requested
        if args.wandb:
            import wandb

            wandb.init(project=str(args.output_dir).split("/")[-1])   
    
    # Loss function
    criterion = nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(
        num_channels=args.num_channels,
        latent_dim=args.latent_dim,
        hidden_size=args.generator_hidden_size,
    )
    discriminator = Discriminator(num_channels=args.num_channels, hidden_size=args.discriminator_hidden_size)

    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Initialize Inceptionv3 (for FID metric)
    model = InceptionV3()

    # Initialize Inceptionv3 (for FID metric)
    model = InceptionV3()

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=accelerator.device)

    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0

    # Setup Adam optimizers for both G and D
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Configure data loader
    dataset = load_dataset(args.dataset)

    transform = Compose(
        [
            Resize(args.image_size),
            CenterCrop(args.image_size),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    def transforms(examples):
        examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]

        del examples["image"]

        return examples

    transformed_dataset = dataset.with_transform(transforms)

    dataloader = DataLoader(
        transformed_dataset["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    generator, discriminator, generator_optimizer, discriminator_optimizer, dataloader = accelerator.prepare(generator, discriminator, generator_optimizer, discriminator_optimizer, dataloader)

    # ----------
    #  Training
    # ----------
    
    # Training Loop

    # Lists to keep track of progress
    img_list = []

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    # For each epoch
    for epoch in range(args.num_epochs):
        # For each batch in the dataloader
        for step, batch in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_cpu = batch["pixel_values"]
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=accelerator.device)
            # Forward pass real batch through D
            output = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            accelerator.backward(errD_real)
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=accelerator.device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            accelerator.backward(errD_fake)
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            discriminator_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            accelerator.backward(errG)
            D_G_z2 = output.mean().item()
            # Update G
            generator_optimizer.step()

            # Log all results
            if (step + 1) % args.logging_steps == 0:
                errD.detach()
                errG.detach()

                if accelerator.state.num_processes > 1:
                    errD = accelerator.gather(errD).sum() / accelerator.state.num_processes
                    errG = accelerator.gather(errG).sum() / accelerator.state.num_processes

                    train_logs = {
                        "epoch": epoch,
                        "discriminator_loss": errD,
                        "generator_loss": errG,
                        "D_x": D_x,
                        "D_G_z1": D_G_z1,
                        "D_G_z2": D_G_z2,
                    }
                    log_str = ""
                    for k, v in train_logs.items():
                        log_str += "| {}: {:.3e}".format(k, v)

                    if accelerator.is_local_main_process:
                        logger.info(log_str)
                        if args.wandb:
                            wandb.log(train_logs)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (step % 500 == 0) or ((epoch == args.num_epochs - 1) and (step == len(dataloader) - 1)):
                with torch.no_grad():
                    fake_images = generator(fixed_noise).detach().cpu()
                file_name = args.output_dir/f"iter_{step}.png"
                save_image(fake_images.data[:25], file_name, nrow=5, normalize=True)
                if accelerator.is_local_main_process and args.wandb:
                    wandb.log({'generated_examples': wandb.Image(str(file_name)) })

        # Calculate FID metric
        fid = calculate_fretchet(real_cpu, fake, model.to(accelerator.device))
        logger.info(f"FID: {fid}")
        if accelerator.is_local_main_process and args.wandb:
            wandb.log({"FID": fid})

    # Optionally push to hub
    if accelerator.is_main_process and args.push_to_hub:
        generator.module.push_to_hub(
            repo_path_or_name=args.output_dir / args.model_name,
            organization=args.organization_name,
        )


def main():
    args = parse_args()
    print(args)

    training_function({}, args)


if __name__ == "__main__":
    main()
