# Note - training loop and architectures are modified from https://github.com/pytorch/examples/blob/master/dcgan/main.py

import argparse
import csv
import logging
import os
import random
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data import DataLoader
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize, ToTensor)
from datasets import load_dataset
import tqdm
from accelerate import Accelerator
from huggan.pytorch.metrics.fid_score import calculate_fretchet
from huggan.pytorch.metrics.inception import InceptionV3
from huggan.pytorch.huggan_mixin import HugGANModelHubMixin

logger = logging.getLogger(__name__)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return str(self.avg)


# Custom weights initialization called on Generator and Discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module, HugGANModelHubMixin):
    def __init__(self, num_channels=4, latent_dim=100, generator_hidden_size=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, generator_hidden_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(generator_hidden_size * 8),
            nn.ReLU(True),
            # state size. (generator_hidden_size*8) x 4 x 4
            nn.ConvTranspose2d(generator_hidden_size * 8, generator_hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_hidden_size * 4),
            nn.ReLU(True),
            # state size. (generator_hidden_size*4) x 8 x 8
            nn.ConvTranspose2d(generator_hidden_size * 4, generator_hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_hidden_size * 2),
            nn.ReLU(True),
            # state size. (generator_hidden_size*2) x 16 x 16
            nn.ConvTranspose2d(generator_hidden_size * 2, generator_hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(generator_hidden_size),
            nn.ReLU(True),
            # state size. (generator_hidden_size) x 32 x 32
            nn.ConvTranspose2d(generator_hidden_size, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (num_channels) x 64 x 64
        )

    def forward(self, input):
        output = self.model(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, num_channels, discriminator_hidden_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # input is (num_channels) x 64 x 64
            spectral_norm(nn.Conv2d(num_channels, discriminator_hidden_size, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (discriminator_hidden_size) x 32 x 32
            spectral_norm(nn.Conv2d(discriminator_hidden_size, discriminator_hidden_size * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (discriminator_hidden_size*2) x 16 x 16
            spectral_norm(nn.Conv2d(discriminator_hidden_size * 2, discriminator_hidden_size * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (discriminator_hidden_size*4) x 8 x 8
            spectral_norm(nn.Conv2d(discriminator_hidden_size * 4, discriminator_hidden_size * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (discriminator_hidden_size*8) x 4 x 4
            spectral_norm(nn.Conv2d(discriminator_hidden_size * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.model(input)
        return output.view(-1, 1).squeeze(1)


def main(args):
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

    Path(args.output_dir).mkdir(exist_ok=True)

    # for reproducibility
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    cudnn.benchmark = True

    dataset = load_dataset(args.dataset)

    transform = Compose(
        [
            Resize(args.image_size),
            CenterCrop(args.image_size),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
        ]
    )

    def transforms(examples):
        examples["pixel_values"] = [transform(image.convert("RGBA")) for image in examples["image"]]

        del examples["image"]

        return examples

    transformed_dataset = dataset.with_transform(transforms)

    dataloader = DataLoader(
        transformed_dataset["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    generator = Generator(args.num_channels, args.latent_dim, args.generator_hidden_size).to(accelerator.device)
    generator.apply(weights_init)

    discriminator = Discriminator(args.num_channels, args.discriminator_hidden_size).to(accelerator.device)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(args.batch_size, args.latent_dim, 1, 1, device=accelerator.device)
    real_label = 1
    fake_label = 0

    # Initialize Inceptionv3 (for FID metric)
    model = InceptionV3()

    # setup optimizer
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    generator_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    generator, discriminator, generator_optimizer, discriminator_optimizer, dataloader = accelerator.prepare(generator,
                                                                                                             discriminator,
                                                                                                             generator_optimizer,
                                                                                                             discriminator_optimizer,
                                                                                                             dataloader)

    with open(f"{args.output_dir}/logs.csv", "w") as f:
        csv.writer(f).writerow(["epoch", "loss_g", "loss_d", "d_x", "d_g_z1", "d_g_z2"])

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    for epoch in tqdm.tqdm(range(args.num_epochs)):

        avg_loss_g = AverageMeter()
        avg_loss_d = AverageMeter()
        avg_d_x = AverageMeter()
        avg_d_g_z1 = AverageMeter()
        avg_d_g_z2 = AverageMeter()

        for data in dataloader:
            ############################
            # (1) Update D model: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            discriminator.zero_grad()
            real_cpu = data["pixel_values"].to(accelerator.device)

            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=accelerator.device)

            output = discriminator(real_cpu).view(-1)
            loss_d_real = criterion(output, label)
            accelerator.backward(loss_d_real)

            avg_d_x.update(output.mean().item(), batch_size)

            # train with fake
            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=accelerator.device)
            fake = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake.detach())
            loss_d_fake = criterion(output, label)
            accelerator.backward(loss_d_fake)
            discriminator_optimizer.step()

            avg_loss_d.update((loss_d_real + loss_d_fake).item(), batch_size)
            avg_d_g_z1.update(output.mean().item())

            ############################
            # (2) Update G model: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = discriminator(fake)
            # minimize loss but also maximize alpha channel
            loss_g = criterion(output, label) + fake[:, -1].mean()
            accelerator.backward(loss_g)
            generator_optimizer.step()

            avg_loss_g.update(loss_g.item(), batch_size)
            avg_d_g_z2.update(output.mean().item())

        # write logs
        with open(f"{args.output_dir}/logs.csv", "a") as f:
            csv.writer(f).writerow(
                [epoch, avg_loss_g, avg_loss_d, avg_d_x, avg_d_g_z1, avg_d_g_z2]
            )

        train_logs = {
            "epoch": epoch,
            "discriminator_loss": avg_loss_d.avg,
            "generator_loss": avg_loss_g.avg,
            "D_x": avg_d_x.avg,
            "D_G_z1": avg_d_g_z1.avg,
            "D_G_z2": avg_d_g_z2.avg,
        }
        if accelerator.is_local_main_process:
            if args.wandb:
                wandb.log(train_logs)

        if (epoch + 1) % args.logging_steps == 0:
            # save samples
            fake = generator(fixed_noise)
            file_name = f"{args.output_dir}/fake_samples_epoch_{epoch}.png"
            vutils.save_image(
                fake.detach(),
                file_name,
                normalize=True,
            )

            if accelerator.is_local_main_process and args.wandb:
                wandb.log({'generated_examples': wandb.Image(str(file_name))})

            # save_checkpoints
            torch.save(generator.state_dict(), f"{args.output_dir}/generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"{args.output_dir}/discriminator_epoch_{epoch}.pth")

            # Calculate FID metric
            fid = calculate_fretchet(real_cpu, fake, model.to(accelerator.device))
            logger.info(f"FID: {fid}")
            if accelerator.is_local_main_process and args.wandb:
                wandb.log({"FID": fid})

    # Optionally push to hub
    if accelerator.is_main_process and args.push_to_hub:
        # os.makedirs(f"{args.output_dir}/{args.model_name}", exist_ok=True)
        generator.push_to_hub(
            repo_path_or_name=f"{args.output_dir}/{args.model_name}",
            organization=args.organization_name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to load from the HuggingFace hub.")
    parser.add_argument(
        "--num_workers", type=int, help="Number of data loading workers", default=0
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Spatial size to use when resizing images for training.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Input batch size")
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
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="Beta1 for adam"
    )
    parser.add_argument(
        "--output_dir", default="./output", help="Folder to output images and model checkpoints"
    )
    parser.add_argument("--manual_seed", type=int, default=0, help="Manual seed")
    parser.add_argument("--wandb", action="store_true", help="If passed, will log to Weights and Biases.")
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
             "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
             "and an Nvidia Ampere GPU.",
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
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Number of steps between each logging",
    )
    args = parser.parse_args()
    args.num_channels = 4
    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."
        assert args.model_name is not None, "Need a `model_name` to create a repo when `--push_to_hub` is passed."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
