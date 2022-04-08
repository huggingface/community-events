# Note - training loop and architectures are modified from https://github.com/pytorch/examples/blob/master/dcgan/main.py

import argparse
import csv
import logging
import random
from pathlib import Path

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data import DataLoader
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor, ToPILImage)
from datasets import load_dataset
import tqdm
from accelerate import Accelerator

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


def get_attributes():
    """
    Read punk attributes file and form one-hot matrix
    """
    df = pd.concat(
        [
            pd.read_csv(f, sep=", ", engine="python")
            for f in Path("attributes").glob("*.csv")
        ]
    )
    accessories = df["accessories"].str.get_dummies(sep=" / ")
    type_ = df["type"].str.get_dummies()
    gender = df["gender"].str.get_dummies()

    return pd.concat([df["id"], accessories, type_, gender], axis=1).set_index("id")


# folder dataset
class Punks(torch.utils.data.Dataset):
    def __init__(self, path, size=10_000):
        self.path = Path(path)
        self.size = size
        # self.attributes = get_attributes()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # randomly select attribute
        # attribute = random.choice(self.attributes.columns)
        # randomly select punk with that attribute
        id_ = random.randint(0, self.size - 1)

        return self.transform(
            Image.open(self.path / f"punk{int(id_):03}.png").convert("RGBA")
        )


# Custom weights initialization called on Generator and Discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, nc=4, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        output = self.network(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 4, 1, 3, 1, 0, bias=False)),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.network(input)
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

    dataset = load_dataset("AlekseyKorshuk/dooggies")

    image_size = 24
    transform = Compose(
        [
            Resize(image_size),
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
        transformed_dataset["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net_g = Generator(args.nc, args.nz, args.ngf).to(device)
    net_g.apply(weights_init)

    net_d = Discriminator(args.nc, args.ndf).to(device)
    net_d.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizer_d = optim.Adam(net_d.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_g = optim.Adam(net_g.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    net_g, net_d, optimizer_g, optimizer_d, dataloader = accelerator.prepare(net_g, net_d, optimizer_g, optimizer_d, dataloader)

    with open(f"{args.output_dir}/logs.csv", "w") as f:
        csv.writer(f).writerow(["epoch", "loss_g", "loss_d", "d_x", "d_g_z1", "d_g_z2"])

    for epoch in tqdm.tqdm(range(args.niter)):

        avg_loss_g = AverageMeter()
        avg_loss_d = AverageMeter()
        avg_d_x = AverageMeter()
        avg_d_g_z1 = AverageMeter()
        avg_d_g_z2 = AverageMeter()

        for data in dataloader:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            net_d.zero_grad()
            real_cpu = data["pixel_values"].to(device)

            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=accelerator.device)

            output = net_d(real_cpu).view(-1)
            loss_d_real = criterion(output, label)
            accelerator.backward(loss_d_real)

            avg_d_x.update(output.mean().item(), batch_size)

            # train with fake
            noise = torch.randn(batch_size, args.nz, 1, 1, device=accelerator.device)
            fake = net_g(noise)
            label.fill_(fake_label)
            output = net_d(fake.detach())
            loss_d_fake = criterion(output, label)
            accelerator.backward(loss_d_fake)
            optimizer_d.step()

            avg_loss_d.update((loss_d_real + loss_d_fake).item(), batch_size)
            avg_d_g_z1.update(output.mean().item())

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            net_g.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = net_d(fake)
            # minimize loss but also maximize alpha channel
            loss_g = criterion(output, label) + fake[:, -1].mean()
            accelerator.backward(loss_g)
            optimizer_g.step()

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

        if (epoch + 1) % args.save_every == 0:
            # save samples
            fake = net_g(fixed_noise)
            file_name = f"{args.output_dir}/fake_samples_epoch_{epoch}.png"
            vutils.save_image(
                fake.detach(),
                file_name,
                normalize=True,
            )

            if accelerator.is_local_main_process and args.wandb:
                wandb.log({'generated_examples': wandb.Image(str(file_name))})

            # save_checkpoints
            torch.save(net_g.state_dict(), f"{args.output_dir}/net_g_epoch_{epoch}.pth")
            torch.save(net_d.state_dict(), f"{args.output_dir}/net_d_epoch_{epoch}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data", help="path to dataset")
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=0
    )
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size")
    parser.add_argument(
        "--nz", type=int, default=100, help="size of the latent z vector"
    )
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument(
        "--niter", type=int, default=1000, help="number of epochs to train for"
    )
    parser.add_argument("--save_every", type=int, default=10, help="how often to save")
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="learning rate, default=0.0002"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )
    parser.add_argument(
        "--output_dir", default="out-test", help="folder to output images and model checkpoints"
    )
    parser.add_argument("--manual_seed", type=int, default=0, help="manual seed")
    parser.add_argument("--wandb", action="store_true", help="if passed, will log to Weights and Biases.")
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
    args = parser.parse_args()
    args.cuda = True
    args.nc = 4
    print(args)
    main(args)
