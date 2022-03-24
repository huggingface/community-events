# CycleGAN

An implementation of [CycleGAN](https://arxiv.org/abs/1703.10593), leveraging the [Hugging Face](https://huggingface.co/) ecosystem for processing data and pushing the model to the Hub.

To train the model with the default parameters (200 epochs, 256x256 images, etc.) on [huggan/facades](https://huggingface.co/datasets/huggan/facades), simply do:

```
python train.py
```

This will create local "images" and "saved_models" directories, containing generated images and saved checkpoints over the course of the training.

To train on another dataset available on the hub, simply do:

```
python train.py --dataset huggan/edges2shoes
```

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

```
python train.py --dataset huggan/my-awesome-dataset
```

## Pushing model to the Hub

You can push your trained generator to the hub after training by specifying the `push_to_hub` flag. 
Then, you can run the script as follows:

```
python train.py --push_to_hub --model_name dcgan-mnist
```

This is made possible by making the generator inherit from `PyTorchModelHubMixin`available in the `huggingface_hub` library. 

# Citation

This repo is entirely based on Erik Linder-Norén's [PyTorch-GAN repo](https://github.com/eriklindernoren/PyTorch-GAN), but with added HuggingFace goodies.