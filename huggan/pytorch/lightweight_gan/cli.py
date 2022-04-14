import fire
import random
from retry.api import retry_call
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from lightweight_gan import Trainer, NanException

import torch
import torch.multiprocessing as mp

import numpy as np

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_list(el):
    return el if isinstance(el, list) else [el]

def timestamped_filename(prefix = 'generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def run_training(model_args, data, load_from, new, num_train_steps, name, seed):
    
    if seed is not None:
        set_seed(seed)

    model = Trainer(**model_args)

    if not new:
        model.load(load_from)
    else:
        model.clear()

    progress_bar = tqdm(initial = model.steps, total = num_train_steps, mininterval=10., desc=f'{name}<{data}>')
    G, D, D_aug = model.init_accelerator()

    # model.set_data_src(data)

    while model.steps < num_train_steps:
        # retry_call(model.train, tries=3, exceptions=NanException)
        model.train(G, D, D_aug)
        progress_bar.n = model.steps
        progress_bar.refresh()
        if model.accelerator.is_local_main_process and model.steps % 50 == 0:
            model.print_log()

    model.save(model.checkpoint_num)

def train_from_folder(
    dataset_name = 'huggan/CelebA-faces',
    data = './data',
    results_dir = './results',
    models_dir = './models',
    name = 'default',
    new = False,
    load_from = -1,
    image_size = 256,
    optimizer = 'adam',
    fmap_max = 512,
    transparent = False,
    greyscale = False,
    batch_size = 10,
    gradient_accumulate_every = 4,
    num_train_steps = 150000,
    learning_rate = 2e-4,
    save_every = 10000,
    evaluate_every = 1000,
    generate = False,
    generate_types = ['default', 'ema'],
    generate_interpolation = False,
    aug_test = False,
    aug_prob=None,
    aug_types=['cutout', 'translation'],
    dataset_aug_prob=0.,
    attn_res_layers = [32],
    freq_chan_attn = False,
    disc_output_size = 1,
    dual_contrast_loss = False,
    antialias = False,
    interpolation_num_steps = 100,
    save_frames = False,
    num_image_tiles = None,
    calculate_fid_every = None,
    calculate_fid_num_images = 12800,
    clear_fid_cache = False,
    seed = 42,
    cpu = False,
    mixed_precision = "no",
    show_progress = False,
    wandb = False,
    push_to_hub = False,
    organization_name = None,
):
    if push_to_hub:
        if name == 'default':
            raise RuntimeError(
                "You've chosen to push to hub, but have left the --name flag as 'default'."
                " You should name your model something other than 'default'!"
            )

    num_image_tiles = default(num_image_tiles, 4 if image_size > 512 else 8)

    model_args = dict(
        dataset_name = dataset_name,
        name = name,
        results_dir = results_dir,
        models_dir = models_dir,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        attn_res_layers = cast_list(attn_res_layers),
        freq_chan_attn = freq_chan_attn,
        disc_output_size = disc_output_size,
        dual_contrast_loss = dual_contrast_loss,
        antialias = antialias,
        image_size = image_size,
        num_image_tiles = num_image_tiles,
        optimizer = optimizer,
        fmap_max = fmap_max,
        transparent = transparent,
        greyscale = greyscale,
        lr = learning_rate,
        save_every = save_every,
        evaluate_every = evaluate_every,
        aug_prob = aug_prob,
        aug_types = cast_list(aug_types),
        dataset_aug_prob = dataset_aug_prob,
        calculate_fid_every = calculate_fid_every,
        calculate_fid_num_images = calculate_fid_num_images,
        clear_fid_cache = clear_fid_cache,
        cpu = cpu,
        mixed_precision = mixed_precision,
        wandb = wandb,
        push_to_hub = push_to_hub,
        organization_name = organization_name
    )

    if generate:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        checkpoint = model.checkpoint_num
        dir_result = model.generate(samples_name, num_image_tiles, checkpoint, generate_types)
        print(f'sample images generated at {dir_result}')
        return

    if generate_interpolation:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        model.generate_interpolation(samples_name, num_image_tiles, num_steps = interpolation_num_steps, save_frames = save_frames)
        print(f'interpolation generated at {results_dir}/{name}/{samples_name}')
        return

    if show_progress:
        model = Trainer(**model_args)
        model.show_progress(num_images=num_image_tiles, types=generate_types)
        return

    run_training(model_args, data, load_from, new, num_train_steps, name, seed)

def main():
    fire.Fire(train_from_folder)

if __name__ == "__main__":
    main()