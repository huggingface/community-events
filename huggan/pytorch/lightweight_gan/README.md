# Train Lightweight GAN on your custom data

This folder contains a script to train ['Lightweight' GAN](https://openreview.net/forum?id=1Fqg133qRaI) for unconditional image generation, leveraging the [Hugging Face](https://huggingface.co/) ecosystem for processing your data and pushing the model to the Hub.

The script leverages 🤗 Datasets for loading and processing data, and 🤗 Accelerate for instantly running on CPU, single, multi-GPUs or TPU, also supporting mixed precision.

<p align="center">
    <img src="https://raw.githubusercontent.com/lucidrains/lightweight-gan/main/images/pizza-512.jpg" alt="drawing" width="300"/>
</p>

Pizza's that don't exist. Courtesy of Phil Wang.

## Launching the script

To train the model with the default parameters on [huggan/CelebA-faces](https://huggingface.co/datasets/huggan/CelebA-faces), first run:

```bash
accelerate config
```

and answer the questions asked about your environment. Next, launch the script as follows: 

```bash
accelerate launch cli.py
```

This will instantly run on multi-GPUs (if you asked for that). To train on another dataset available on the hub, simply do (for instance):

```bash
accelerate launch cli.py --dataset_name huggan/pokemon
```

In case you'd like to tweak the script to your liking, first fork the "community-events" [repo](https://github.com/huggingface/community-events) (see the button on the top right), then clone it locally:

```bash
git clone https://github.com/<your Github username>/community-events.git
```

and edit to your liking.

## Training on your own data

You can of course also train on your own images. For this, one can leverage Datasets' [ImageFolder](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder). Make sure to authenticate with the hub first, by running the `huggingface-cli login` command in a terminal, or the following in case you're working in a notebook:

```python
from huggingface_hub import notebook_login

notebook_login()
```

Next, run the following in a notebook/script:

```python
from datasets import load_dataset

# first: load dataset
# option 1: from local folder
dataset = load_dataset("imagefolder", data_dir="path_to_folder")
# option 2: from remote URL (e.g. a zip file)
dataset = load_dataset("imagefolder", data_files="URL to .zip file")

# next: push to the hub (assuming git-LFS is installed)
dataset.push_to_hub("huggan/my-awesome-dataset")
```

You can then simply pass the name of the dataset to the script:

```bash
accelerate launch cli.py --dataset huggan/my-awesome-dataset
```

## Weights and Biases integration

You can easily add logging to [Weights and Biases](https://wandb.ai/site) by passing the `--wandb` flag:

```bash
accelerate launch cli.py --wandb
````

You can then follow the progress of your GAN in a browser:

<p align="center">
    <img src="https://raw.githubusercontent.com/huggingface/community-events/main/huggan/assets/lightweight_gan_wandb.png" alt="drawing" width="700"/>
</p>


# Citation

This repo is entirely based on lucidrains' [Pytorch implementation](https://github.com/lucidrains/lightweight-gan), but with added HuggingFace goodies.
