## Train DCGAN on your custom data
This folder contains a script to train DCGAN for unconditional image generation, leveraging the Hugging Face ecosystem for processing your data and pushing the model to the Hub.

The script leverages 🤗 [Datasets](https://huggingface.co/docs/datasets/index) for loading and processing data, and TensorFlow for training the model and 🤗 [Hub](https://huggingface.co/) for hosting it.

## Launching the script
You can simply run `python train.py --num_channels 1` with the default parameters. It will download the [MNIST](https://huggingface.co/datasets/mnist) dataset, preprocess it and train a model on it, will save results after each epoch in a local directory and push the model to the 🤗 Hub.

To train on another dataset available on the hub, simply do (for instance):

```bash
python train.py --dataset cifar10
```

## Training on your own data
You can of course also train on your own images. For this, one can leverage Datasets' [ImageFolder](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder). Make sure to authenticate with the hub first, by running the huggingface-cli login command in a terminal, or the following in case you're working in a notebook:

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
# You can then simply pass the name of the dataset to the script:

python train.py --dataset huggan/my-awesome-dataset
```

## Pushing model to the Hub

For this you can use `push_to_hub_keras` which generates a card for your model with training metrics, plot of the architecture and hyperparameters. For this, specify `--output_dir` and `--model_name` and use the `--push_to_hub` flag like so:
```bash
python train.py --push_to_hub --output_dir /output --model_name awesome_gan_model
```

## Citation
This repo is entirely based on [TensorFlow's official DCGAN tutorial](https://www.tensorflow.org/tutorials/generative/dcgan), but with added HuggingFace goodies.
