import argparse
import logging
import os
import numpy as np
import itertools
from pathlib import Path
import datetime
import time
import sys

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader

from modeling_cyclegan import GeneratorResNet, Discriminator

from utils import ReplayBuffer, LambdaLR

from datasets import load_dataset

from accelerate import Accelerator

import torch.nn as nn
import torch


logger = logging.getLogger(__name__)

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
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
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
    parser.add_argument("--output_dir", type=Path, default=Path("./output-cyclegan"), help="Name of the directory to dump generated images during training.")
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
    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."
        assert args.model_name is not None, "Need a `model_name` to create a repo when `--push_to_hub` is passed."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)


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

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    input_shape = (args.channels, args.image_size, args.image_size)
    # Calculate output shape of image discriminator (PatchGAN)
    output_shape = (1, args.image_size // 2 ** 4, args.image_size // 2 ** 4)

    # Initialize generator and discriminator
    G_AB = GeneratorResNet(input_shape, args.n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, args.n_residual_blocks)
    D_A = Discriminator(args.channels)
    D_B = Discriminator(args.channels)

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
        examples["imageA"] = [transform(image.convert("RGB")) for image in examples["imageA"]]
        examples["imageB"] = [transform(image.convert("RGB")) for image in examples["imageB"]]

        return examples

    dataset = load_dataset(args.dataset_name)
    transformed_dataset = dataset.with_transform(transforms)

    splits = transformed_dataset['train'].train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']

    dataloader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_ds, batch_size=5, shuffle=True, num_workers=1)

    def sample_images():
        """Saves a generated sample from the test set"""
        batch = next(iter(val_dataloader))
        G_AB.eval()
        G_BA.eval()
        real_A = batch["imageA"]
        fake_B = G_AB(real_A)
        real_B = batch["imageB"]
        fake_A = G_BA(real_B)
        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=True)
        real_B = make_grid(real_B, nrow=5, normalize=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)

        return image_grid

    G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, dataloader, val_dataloader = accelerator.prepare(G_AB, G_BA, D_A, D_B, optimizer_G, optimizer_D_A, optimizer_D_B, dataloader, val_dataloader)
    
    # ----------
    #  Training
    # ----------

    prev_time = time.time()
    for epoch in range(args.epoch, args.num_epochs):
        for step, batch in enumerate(dataloader):

            # Set model input
            real_A = batch["imageA"]
            real_B = batch["imageB"]

            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), *output_shape), device=accelerator.device)
            fake = torch.zeros((real_A.size(0), *output_shape), device=accelerator.device)

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

            accelerator.backward(loss_G)
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

            accelerator.backward(loss_D_A)
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

            accelerator.backward(loss_D_B)
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + step
            batches_left = args.num_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Log results
            if (step + 1) % args.logging_steps == 0:
                loss_D.detach()
                loss_G.detach()
                loss_GAN.detach()
                loss_cycle.detach()
                loss_identity.detach()

                if accelerator.state.num_processes > 1:
                    loss_D = accelerator.gather(loss_D).sum() / accelerator.state.num_processes
                    loss_G = accelerator.gather(loss_G).sum() / accelerator.state.num_processes
                    loss_GAN = accelerator.gather(loss_GAN).sum() / accelerator.state.num_processes
                    loss_cycle = accelerator.gather(loss_cycle).sum() / accelerator.state.num_processes
                    loss_identity = accelerator.gather(loss_identity).sum() / accelerator.state.num_processes

                train_logs = {
                    "epoch": epoch,
                    "discriminator_loss": loss_D,
                    "generator_loss": loss_G,
                    "GAN_loss": loss_GAN,
                    "cycle_loss": loss_cycle,
                    "identity_loss": loss_identity,
                    # "time_left": time_left,
                }
                log_str = ""
                for k, v in train_logs.items():
                    log_str += "| {}: {:.3e}".format(k, v)

                if accelerator.is_local_main_process:
                    logger.info(log_str)
                    if args.wandb:
                        wandb.log(train_logs)

            # If at sample interval save image
            if batches_done % args.sample_interval == 0:
                image_grid = sample_images()
                file_name = args.output_dir / f"{batches_done}.png"
                save_image(image_grid, file_name, normalize=False)
                if accelerator.is_local_main_process and args.wandb:
                    wandb.log({'generated_examples': wandb.Image(str(file_name)) })

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

    # Optionally push to hub
    if accelerator.is_main_process and args.push_to_hub:
        model = accelerator.unwrap_model(G_AB)
        model.module.push_to_hub(
            repo_path_or_name=args.output_dir / args.model_name,
            organization=args.organization_name,
        )

def main():
    args = parse_args()
    print(args)

    training_function({}, args)


if __name__ == "__main__":
    main()