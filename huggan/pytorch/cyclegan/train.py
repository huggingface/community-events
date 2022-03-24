import argparse
import os
import numpy as np
import itertools
from pathlib import Path
import datetime
import time
import sys

from PIL import Image

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader

from modeling_cyclegan import GeneratorResNet, Discriminator

from utils import ReplayBuffer, LambdaLR

from datasets import load_dataset

import torch.nn as nn
import torch

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="huggan/facades", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU threads to use during batch generation")
    parser.add_argument("--image_size", type=int, default=256, help="Size of images for training")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
    parser.add_argument(
            "--push_to_hub",
            action="store_true",
            help="Whether to push the model to the HuggingFace hub after training.",
            )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        required="--push_to_hub" in sys.argv,
        type=Path,
        help="Path to save the model. Will be created if it doesn't exist already.",
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

def train(args):
    # Create sample and checkpoint directories
    os.makedirs("images/%s" % args.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % args.dataset_name, exist_ok=True)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    input_shape = (args.channels, args.image_size, args.image_size)

    # Initialize generator and discriminator
    G_AB = GeneratorResNet(input_shape, args.n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, args.n_residual_blocks)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    G_AB = G_AB.to(device)
    G_BA = G_BA.to(device)
    D_A = D_A.to(device)
    D_B = D_B.to(device)
    criterion_GAN.to(device)
    criterion_cycle.to(device)
    criterion_identity.to(device)

    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    if args.epoch != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (args.dataset_name, args.epoch)))
        G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (args.dataset_name, args.epoch)))
        D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (args.dataset_name, args.epoch)))
        D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (args.dataset_name, args.epoch)))
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=args.lr, betas=(args.beta1, args.beta2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(args.num_epochs, args.epoch, args.decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(args.num_epochs, args.epoch, args.decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(args.num_epochs, args.epoch, args.decay_epoch).step
    )

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Image transformations
    transform = Compose([
        Resize(int(args.image_size * 1.12), Image.BICUBIC),
        RandomCrop((args.image_size, args.image_size)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def transforms(examples):
        examples["A"] = [transform(image.convert("RGB")) for image in examples["imageA"]]
        examples["B"] = [transform(image.convert("RGB")) for image in examples["imageB"]]

        del examples["imageA"]
        del examples["imageB"]

        return examples

    dataset = load_dataset(args.dataset_name)
    transformed_dataset = dataset.with_transform(transforms)

    splits = transformed_dataset['train'].train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']

    dataloader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_ds, batch_size=5, shuffle=True, num_workers=1)

    def sample_images(batches_done):
        """Saves a generated sample from the test set"""
        batch = next(iter(val_dataloader))
        G_AB.eval()
        G_BA.eval()
        real_A = batch["A"].to(device)
        fake_B = G_AB(real_A)
        real_B = batch["B"].to(device)
        fake_A = G_BA(real_B)
        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=True)
        real_B = make_grid(real_B, nrow=5, normalize=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        save_image(image_grid, "images/%s/%s.png" % (args.dataset_name, batches_done), normalize=False)


    # ----------
    #  Training
    # ----------

    prev_time = time.time()
    for epoch in range(args.epoch, args.num_epochs):
        for i, batch in enumerate(dataloader):

            # Set model input
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), *D_A.output_shape), device=device)
            fake = torch.zeros((real_A.size(0), *D_A.output_shape), device=device)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + args.lambda_cyc * loss_cycle + args.lambda_id * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = args.num_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    args.num_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % args.sample_interval == 0:
                sample_images(batches_done)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (args.dataset_name, epoch))
            torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (args.dataset_name, epoch))
            torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (args.dataset_name, epoch))
            torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (args.dataset_name, epoch))

    # Optionally push to hub
    if args.push_to_hub:
        save_directory = args.pytorch_dump_folder_path
        if not save_directory.exists():
            save_directory.mkdir(parents=True)

        G_AB.push_to_hub(
            repo_path_or_name=save_directory / args.model_name,
            organization=args.organization_name,
        )

def main():
    args = parse_args()
    print(args)

    # Make directory for saving generated images
    os.makedirs("images", exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()