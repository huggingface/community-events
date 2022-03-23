# HugGAN Sprint

![Banner](assets/huggan_banner.png?raw=true "Banner")

_**Timeline**: April 4th, 2020 - April 17th, 2020_

---

Welcome to HugGAN Sprint! The goal of this sprint is to add more GANs and GAN-based demos to the Hugging Face Hub 🤗.

During the sprint, we’ll be bringing in some awesome speakers to talk about GANs and the future of generative models. Oh, and if you need access to compute for your project, we’ll help you there too! As an added bonus, if you choose to participate, we’ll send you a gift (specific details TBD). We encourage you to form teams of ~2-3 people! Make friends in the Discord :)

To join:

1. Fill out [this form](https://forms.gle/goq41UgzsvuKKTFFA), so we can keep track of who’s joining.
2. Send a reaction in the #join-sprint channel under the HugGAN category in Discord. This will add you to the rest of the related channels. If you haven't joined our discord yet, [click here](discord.gg/H3bUrDPTfS).
3. Once you’ve decided what you want to work on, add your project’s information to [this sheet](https://docs.google.com/spreadsheets/d/1aAHqOOk2SOw4j6mrJLkLT6ZyKyLDOvGF5D9tuUqnoG8/edit#gid=0), where you can describe your project and let us know if you need additional compute. Still brainstorming? Feel free to propose ideas in #sprint-discussions.

## Table of Contents

- [Important dates](##important-dates)
- [How to install relevant libraries](##how-to-install-relevant-libraries)
- [General workflow](##general-workflow)
- [Submissions](##submissions)
- [Links to check out](##links-to-check-out)

## Important dates

(to do)

## How to install relevant libraries

The following libraries are required to train a generative model for this sprint:

- [PyTorch](https://pytorch.org/) or [Keras](https://keras.io/) - depending on which framework you prefer ;)
- [🤗 Datasets](https://github.com/huggingface/datasets)

We recommend installing the above libraries in a [virtual environment](https://docs.python.org/3/library/venv.html). 
If you're unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). Create a virtual environment with the version of Python you're going to use and activate it.

You should be able to run the command:

```bash
python3 -m venv <your-venv-name>
```

You can activate your venv by running

```bash
source ~/<your-venv-name>/bin/activate
```

### Installing PyTorch or Keras

For installing PyTorch or Keras, we refer to the respective installing guides ([PyTorch](https://pytorch.org/get-started/locally/), [Keras](https://keras.io/getting_started/)). In case you're using PyTorch, please make sure you have both PyTorch and CUDA (for the GPUs) correctly installed. 
The following command should return ``True``:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If the above command doesn't print ``True``, in the first step, please follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch with CUDA.

### Installing 🤗 Datasets

To install the Datasets library, simply run:

```bash
pip install datasets
```

or, in case you're using Conda:

```bash
conda install -c huggingface -c conda-forge datasets
```

## General worfklow

The process to follow is outlined below. It consists of 3 steps:

1. Get a dataset and push to hub
2. Train a model and push to hub
3. Create a demo (🤗 Space)

These steps are explained in more detail below.

### 1. Get a dataset and push to hub

The first step is the most obvious one: to train a GAN (or any neural network), we need a dataset. This could be a standard one that is already available on the [hub](https://huggingface.co/datasets?task_categories=task_categories:image-classification) (such as MNIST, CIFAR-10, CIFAR-100, etc.) or it could be one that's not already on the hub, for instance one that you collected yourself.

In the former case, you can easily load a dataset as follows:

```python
from datasets import load_dataset

dataset = load_dataset("mnist")
```

In the latter case, it's required to upload the dataset to the hub, as part of the `huggan` [organization](https://huggingface.co/huggan). For this, we can leverage the [`ImageFolder`](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder) builder which was added recently to the Datasets library. 

First, load your image dataset as a `Dataset` object:

```python
from datasets import load_dataset

# option 1: local folder
dataset = load_dataset("imagefolder", data_dir="path_to_folder")
# option 2: local or remote file(s
dataset = load_dataset("imagefolder", data_files="path_to_zip")
```

Once you've loaded your dataset, you can check it out:

```python
dataset
```

Next, you can push it to the hub. To do this, make sure (1) git-LFS is installed and (2) that you're logged in. Installing git-LFS can be done as follows (in a terminal):

```bash
sudo apt-get install git-lfs
```

Logging in can be done as follows:

```python
from huggingface_hub import notebook_login

notebook_login()
```

in case you're working in a notebook, or by running the 

```bash
huggingface-cli login
``` 

command in a terminal. Finally, you can push your dataset to the hub as follows:

```python
dataset.push_to_hub("huggan/name-of-your-dataset")
```

### 2. Train a model and push to hub

Next, one can start training a model. This could be any model you'd like. However, we do provide some example scripts to help you get started, in both [PyTorch](pytorch) and [Keras](keras). An example is the [DCGAN](pytorch/dcgan) model. Simply follow the README that explains all the details of the relevant implementation, and run it in your environment.

IMPORTANT: If you're planning to train a custom PyTorch model, it's recommended to make it inherit from `PyTorchModelHubMixin`. This makes sure you can push it to the hub at the end of training, and reload it afterwards using `from_pretrained, as shown in the code example below:

```python
from huggingface_hub import PyTorchModelHubMixin

class MyModel(nn.Module, PyTorchModelHubMixin):
   def __init__(self, **kwargs):
      super().__init__()
      self.config = kwargs.pop("config", None)
      self.layer = ...
   def forward(self, ...):
      return ...

# Create model
model = MyModel()

# Push to HuggingFace Hub
model.push_to_hub("huggan/name-of-your-model").

# Reload from HuggingFace Hub
reloaded = MyModel.from_pretrained("huggan/name-of-your-model").
```

In Keras, one can leverage the `push_to_hub_keras` and `from_pretrained_keras` methods:

```python
import tensorflow as tf
from huggingface_hub import push_to_hub_keras, from_pretrained_keras

# Build a Keras model
inputs = tf.keras.layers.Input(shape=(2,))
x = tf.keras.layers.Dense(2, activation="relu")(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=x)
model.compile(optimizer="adam", loss="mse")

# Push to HuggingFace Hub
push_to_hub_keras(model, "huggan/my-cool-model")

# Reload from HuggingFace Hub
reloaded = from_pretrained_keras("huggan/my-cool-model")
```

Models can be either trained by you, or ones that are available you’d like to share with the community. If you didn’t train the model yourself, be sure to both credit the original authors and include the associated license in your model card! Here is an [example model repo](https://huggingface.co/merve/anime-faces-generator).

Model repositories are expected to have a full model card 🃏 that includes:
- license,
- task,
- dataset metadata,
- information related to the model,
- information on dataset, intended uses,
- a model output.

![Alt text](assets/example_model.png?raw=true "Title")

Don't know how to share a model? Check out [this guide](https://huggingface.co/docs/hub/adding-a-model#adding-your-model-to-the-hugging-face-hub)!

### 3. Create a demo

Once you share a model, you then should share a [Space](https://huggingface.co/spaces) based on your SDK of choice (Gradio or Streamlit) or as a static page. 🌌

![Alt text](assets/example_space.png?raw=true "Title")

Here is an [example Space](https://huggingface.co/spaces/merve/anime-face-generator) corresponding to the model example shared above. Don’t know how to create a space? Read more about how to add spaces [here](https://huggingface.co/docs/hub/spaces).

## Submissions

For each submission, you are expected to submit:

1. A model repository
2. A space made with the model repository you created

---

## Links to Check Out

- https://github.com/lucidrains/lightweight-gan
- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- https://keras.io/examples/generative/wgan_gp/
- https://keras.io/examples/generative/conditional_gan/
- https://github.com/lucidrains/stylegan2-pytorch
- https://github.com/NVlabs/stylegan3
- https://github.com/ajbrock/BigGAN-PyTorch
- https://github.com/huggingface/pytorch-pretrained-BigGAN
- https://paperswithcode.com/task/image-generation
- https://github.com/facebookresearch/ic_gan
