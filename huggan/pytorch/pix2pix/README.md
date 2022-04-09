# Train Pix2pix on your custom data

This folder contains a script to train [pix2pix](https://arxiv.org/abs/1611.07004) for conditional image generation, leveraging the [Hugging Face](https://huggingface.co/) ecosystem for processing data and pushing the model to the Hub.

The script leverages 🤗 Datasets for loading and processing data, and 🤗 Accelerate for instantly running on CPU, single, multi-GPUs or TPU, also supporting fp16/mixed precision.

## Launching the script

To train the model with the default parameters (200 epochs, 256x256 images, etc.) on [huggan/facades](https://huggingface.co/datasets/huggan/facades) on your environment, first run:

```bash
accelerate config
```

and answer the questions asked about your environment. Next, launch the script as follows: 

```
accelerate launch train.py
```

This will create local "images" and "saved_models" directories, containing generated images and saved checkpoints over the course of the training.

To train on another dataset available on the hub, simply do (for instance):

```
accelerate launch train.py --dataset huggan/night2day
```

Make sure to pick a dataset which has "imageA" and "imageB" columns defined. One can always tweak the script in case the column names are different.

In case you'd like to tweak the script to your liking, first fork the "community-events" [repo](https://github.com/huggingface/community-events) (see the button on the top right), then clone it locally:

```bash
git clone https://github.com/<your Github username>/community-events.git
```

and edit to your liking.

## Training on your own data

You can of course also train on your own images. For this, one can leverage Datasets' [ImageFolder](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder). Make sure to authenticate with the hub first, either by running the `huggingface-cli login` command in a terminal, or the following in case you're working in a notebook:

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

# optional: remove "label" column, in case there are no subcategories
dataset['train'] = dataset['train'].remove_columns(column_names="label")

# next: push to the hub (assuming git-LFS is installed)
dataset.push_to_hub("huggan/my-awesome-dataset")
```

You can then simply pass the name of the dataset to the script:

```
accelerate launch train.py --dataset huggan/my-awesome-dataset
```

## Pushing model to the Hub

You can push your trained generator to the hub after training by specifying the `push_to_hub` flag, along with a `model_name` and `pytorch_dump_folder_path`. 

```bash
accelerate launch train.py --push_to_hub --model_name pix2pix-facades --pytorch_dump_folder_path output
```

This is made possible by making the generator inherit from `PyTorchModelHubMixin` available in the `huggingface_hub` library. 

# Citation

This repo is entirely based on Erik Linder-Norén's [PyTorch-GAN repo](https://github.com/eriklindernoren/PyTorch-GAN), but with added HuggingFace goodies.
