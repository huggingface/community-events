# HugGAN Sprint

![Banner](assets/huggan_banner.png?raw=true "Banner")

_**Timeline**: April 4th, 2020 - April 17th, 2020_

---

Welcome to HugGAN Sprint! The goal of this sprint is to add more GANs and GAN-based demos to the Hugging Face Hub 🤗.

During the sprint, we’ll be bringing in some awesome speakers to talk about GANs and the future of generative models. Oh, and if you need access to compute for your project, we’ll help you there too! As an added bonus, if you choose to participate, we’ll send you a gift (specific details TBD). We encourage you to form teams of ~2-3 people! Make friends in the Discord :)

To join:

1. Fill out [this form](https://forms.gle/goq41UgzsvuKKTFFA), so we can keep track of who’s joining.
2. Send a reaction in the [#join-sprint channel](https://discord.com/channels/879548962464493619/954070850645135462) under the HugGAN category in Discord. This will add you to the rest of the related channels. If you haven't joined our discord yet, [click here](discord.gg/H3bUrDPTfS).
3. Once you’ve decided what you want to work on, add your project’s information to [this sheet](https://docs.google.com/spreadsheets/d/1aAHqOOk2SOw4j6mrJLkLT6ZyKyLDOvGF5D9tuUqnoG8/edit#gid=0), where you can describe your project and let us know if you need additional compute. Still brainstorming? Feel free to propose ideas in #sprint-discussions.

## Table of Contents

- [Important dates](#important-dates)
- [How to install relevant libraries](#how-to-install-relevant-libraries)
- [General workflow](#general-workflow)
- [Links to check out](#links-to-check-out)
- [Evaluation](#evaluation)
- [Prizes](#prizes)
- [Communication and Problems](#communication-and-problems)
- [Talks](#talks)
- [General Tips & Tricks](#general-tips-and-tricks)

## Important dates

| Date      | Description |
| ----------- | ----------- |
| April 4th      | Sprint Kickoff 🚀      |
| April 15th   | Submission Deadline 🛑  |
| April 22nd | Prizes Announced for Participants 🎁 |

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

The first step is the most obvious one: to train a GAN (or any neural network), we need a dataset. This could be either a dataset that is already available on the hub, or one that isn't already. Below we'll explain how to load the data in both cases.

#### 1.1 Use a dataset already available on the hub

Most famous computer vision dataset are already available on the [hub](https://huggingface.co/datasets?task_categories=task_categories:image-classification) (such as MNIST, CIFAR-10, CIFAR-100, etc.).

Loading a dataset can be done as follows:

```python
from datasets import load_dataset

# a general one ...
dataset = load_dataset("mnist")

# ... or one that's part of the huggan organization
dataset = load_dataset("huggan/edges2shoes")
```
In a notebook, you can directly see the images by selecting a split and then the appropriate column:

```python
example = dataset['train'][0]
print(example['image'])
```

#### 1.2 Upload a new dataset to the hub

In case your dataset is not already on the hub, you can upload it to the `huggan` [organization](https://huggingface.co/huggan). If you've signed up for the event by filling in the [spreadsheet]((https://docs.google.com/spreadsheets/d/1aAHqOOk2SOw4j6mrJLkLT6ZyKyLDOvGF5D9tuUqnoG8/edit#gid=0)), your HuggingFace account should be part of it. 

To begin with, you should check that you are correctly logged in and that you have `git-lfs` installed so that your dataset can be uploaded.

Run:

```bash
huggingface-cli login
```

in a terminal, or case you're working in a notebook

```python
from huggingface_hub import notebook_login

notebook_login()
```

to login. It is recommended to login with your access token that can be found under your HuggingFace profile (icon in the top right corner on [hf.co](http://hf.co/), then Settings -> Access Tokens -> User Access Tokens -> New Token (if you haven't generated one already)

You can then copy-paste this token to log in locally.

Next, let's make sure that `git-lfs` is correctly installed. To so, simply run:

```bash
git-lfs -v
```

The output should show something like `git-lfs/2.13.2 (GitHub; linux amd64; go 1.15.4)`. If your console states that the `git-lfs` command was not found, please make sure to install it [here](https://git-lfs.github.com/) or simply via: 

```bash
sudo apt-get install git-lfs
```

Next, we can leverage the [`ImageFolder`](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder) builder to very easily upload an image dataset to the hub. First, load your image dataset as a `Dataset` object:

```python
from datasets import load_dataset

# option 1: local folder
dataset = load_dataset("imagefolder", data_dir="path_to_folder")
# option 2: local or remote file(s), e.g. the Edge2Shoes dataset of pix2pix
dataset = load_dataset("imagefolder", data_files="http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz")
```

Once you've loaded your dataset, you can push it to the hub, by simply typing:

```python
dataset.push_to_hub("huggan/name-of-your-dataset")
```

Et voila! Your dataset is now available on the hub :) If you wait a bit, the Dataset viewer should be able to preview images in the browser (check for instance [this edges2shoes dataset](https://huggingface.co/datasets/huggan/edges2shoes)). The cool thing is that anyone can now access this dataset from anywhere, using `load_dataset`. 

### 2. Train a model and push to hub

Next, one can start training a model. This could be any model you'd like, however, we do provide some example scripts to help you get started, in both [PyTorch](pytorch) and [Keras](keras). An example is the [DCGAN](pytorch/dcgan) model for unconditional image generation. Simply follow the README that explains all the details of the relevant implementation, and run it in your environment.

Alternatively, we also provide a [Links to Check Out](#links-to-check-out) section to give you some inspiration.

Below, we explain in more detail how to upload your model to the hub, depending on the framework you're using (sections 2.1 and 2.2). In section 2.3, we'll explain how to write a nice model card. In section 2.4, we'll illustrate alternative ways to upload (and re-use) a model to (and from) the hub.

#### 2.1 PyTorch

If you're planning to train a custom PyTorch model, it's recommended to make it inherit from `PyTorchModelHubMixin`. This makes sure you can push it to the hub at the end of training, and reload it afterwards using `from_pretrained`, as shown in the code example below:

```python
from huggingface_hub import PyTorchModelHubMixin

class MyGenerator(nn.Module, PyTorchModelHubMixin):
   def __init__(self, **kwargs):
      super().__init__()
      self.config = kwargs.pop("config", None)
      self.layer = ...
   def forward(self, ...):
      return ...

# Create model
model = MyGenerator()

# Push to HuggingFace Hub
model.push_to_hub("huggan/name-of-your-model").

# Reload from HuggingFace Hub
reloaded = MyGenerator.from_pretrained("huggan/name-of-your-model").
```

This `PyTorchModelHubMixin` class is available in the [`huggingface_hub` library](https://github.com/huggingface/huggingface_hub), which comes pre-installed if you install `datasets` (or `transformers`) in your environment.

#### 2.2 Keras

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

These methods are available in the [`huggingface_hub` library](https://github.com/huggingface/huggingface_hub), which comes pre-installed if you install `datasets` (or `transformers`) in your environment. Note that the `push_to_hub_keras` method supports pushing several models (such as a generator and discriminator) to the same repo, as illustrated [here](https://github.com/huggingface/huggingface_hub/issues/533#issuecomment-1058093158).

#### 2.3 Alternative ways to upload a model to the hub

Besides the methods explained in sections 2.1 and 2.2 above, you can also share model assets directly from git, which is explained in depth in [this guide](https://huggingface.co/docs/hub/adding-a-model#uploading-your-files).

#### 2.4 Model cards

When uploading a model to the hub, it's important to include a so-called [model card](https://huggingface.co/course/chapter4/4?fw=pt) with it. This is just a README (in Markdown) 🃏 that includes:
- license,
- task,
- `huggan` and `gan` tags,
- dataset metadata,
- information related to the model,
- information on dataset, intended uses,
- a model output.

If you trained one of the example models, this model card will be automatically generated for you. If you didn’t train the model yourself, be sure to both credit the original authors and include the associated license in your model card! Here is an [example model repo](https://huggingface.co/merve/anime-faces-generator).

You can also use this [template model card](model_card_template.md)
 as a guide to build your own.

![Alt text](assets/example_model.png?raw=true "Title")

### 3. Create a demo

Once you share a model, you then should share a [Space](https://huggingface.co/spaces) based on your SDK of choice (Gradio or Streamlit) or as a static page. 🌌

![Alt text](assets/example_space.png?raw=true "Title")

Here is an [example Space](https://huggingface.co/spaces/merve/anime-face-generator) corresponding to the model example shared above. Don’t know how to create a space? Read more about how to add spaces [here](https://huggingface.co/docs/hub/spaces).

## Example Scripts

In this repo, we have provided some example scripts you can use to train your own GANs. Below is a table of the available scripts:

| Name      | Paper |
| ----------- | ----------- |
| [CycleGAN](pytorch/cyclegan/README.md)  | [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
| [DCGAN](pytorch/dcgan/README.md)  | [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)  |
| [pix2pix](pytorch/pix2pix/README.md) | [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) |

## Links to Check Out

Possible models to port to the hub, PyTorch:
- Lightweight-GAN: https://github.com/lucidrains/lightweight-gan
- StyleGAN2: https://github.com/lucidrains/stylegan2-pytorch
- StyleGAN3: https://github.com/NVlabs/stylegan3
- BigGAN: https://github.com/ajbrock/BigGAN-PyTorch, https://github.com/huggingface/pytorch-pretrained-BigGAN
- ADGAN: https://github.com/menyifang/ADGAN
- ICGAN: https://github.com/facebookresearch/ic_gan
- StarGANv2: https://github.com/clovaai/stargan-v2
- Progressive Growing GAN: https://github.com/Maggiking/PGGAN-PyTorch

Possible models to port to the hub, Keras:
- WGAN-GP: https://keras.io/examples/generative/wgan_gp/
- Conditional GAN: https://keras.io/examples/generative/conditional_gan/

GAN metrics:
- https://github.com/yhlleo/GAN-Metrics

General links & tutorials:
- https://paperswithcode.com/task/image-generation
- https://github.com/facebookresearch/ic_gan

## Evaluation

For each submission, you are expected to submit:

1. A model repository
2. A space made with the model repository you created

## Prizes

TODO

## Communication and Problems

If you encounter any problems or have any questions, you should use one of the following platforms depending on your type of problem. Hugging Face is an "open-source-first" organization meaning  that we'll try to solve all problems in the most public and most transparent way possible so that everybody in the community profits.

The following table summarizes what platform to use for which problem.

- Problem/question/bug with the 🤗 Datasets library that you think is a general problem that also impacts other people, please open an [Issues on Datasets](https://github.com/huggingface/datasets/issues/new?assignees=&labels=bug&template=bug-report.md&title=) and ping @nielsrogge.
- Problem/question with a modified, customized training script that is less likely to impact other people, please post your problem/question [on the forum](https://discuss.huggingface.co/) and ping @nielsrogge.
- Questions regarding access to the OVHcloud GPU, please ask in the Discord channel **#ovh-support**.
- Other questions regarding the event, rules of the event, or if you are not sure where to post your question, please ask in the Discord channel [**#sprint-discussions**](https://discord.com/channels/879548962464493619/954111918895943720).

## Talks

TODO

## General Tips and Tricks

- Memory efficient training:

In case, you are getting out-of-memory errors on your GPU, we recommend to use  [bitsandbytes](https://github.com/facebookresearch/bitsandbytes) to replace the native memory-intensive Adam optimizer with the one of `bitsandbytes`. It can be used to both train the generator and the discriminator in case you're training a GAN.
