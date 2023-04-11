# JAX/Diffusers community sprint 🧨

Welcome to the JAX/Diffusers community sprint! The goal of this sprint is to work on fun and creative diffusion models using JAX and Diffusers.

In this event, we will create various applications with diffusion models in JAX/Flax and Diffusers using free TPU hours generously provided by Google Cloud.

This document is a walkthrough on all the important information to make a submission to the JAX/Diffusers community sprint.

Don't forget to fill out the [signup form]! 

> 💡 Note: This document is still WIP and it only contains initial details of the event. We will keep updating this document as we make other relevant information available throughout the community sprint.

## Table of Contents

- [Organization](#organization)
- [Important dates](#important-dates)
- [Communication](#communication)
- [Talks](#talks)
- [Data and Pre-processing](#data-and-pre-processing)
    - [Prepare a large local dataset](#prepare-a-large-local-dataset)
    - [Prepare a dataset with MediaPipe and Hugging Face](#prepare-a-dataset-with-mediapipe-and-hugging-face)    
- [Training ControlNet](#training-controlnet)
    - [Setting up your TPU VM](#setting-up-your-tpu-vm)
    - [Installing JAX](#installing-jax)
    - [Running the training script](#running-the-training-script)

## Organization 

Participants can propose ideas for an interesting project involving Diffusion models. Teams of 3 to 5 will then be formed around the most promising and interesting projects. Make sure to read through the [Projects] (TODO) section on how to propose projects, comment on other participants' project ideas, and create a team.

To help each team successfully finish their project, we will organize talks by leading scientists and engineers from Google, Hugging Face, and the open-source diffusion community. The talks will take place on 17th of April. Make sure to attend the talks to get the most out of your participation! Check out the [Talks](#talks) section to get an overview of the talks, including the speaker and the time of the talk.

Each team is then given **free access to a TPU v4-8 VM** from April 14 to May 1st. In addition, we will provide a training example in JAX/Flax and Diffusers to train [ControlNets](https://huggingface.co/blog/controlnet) to kick-start your project. We will also provide examples of how to prepare datasets for ControlNet training. During the sprint, we'll make sure to answer any questions you might have about JAX/Flax and Diffusers and help each team as much as possible to complete their projects!

At the end of the community sprint, each submission will be evaluated by a jury and the top-3 demos will be awarded a prize. Check out the [How to submit a demo] (TODO) section for more information and suggestions on how to submit your project.

> 💡 Note: Even though we provide an example for performing ControlNet training, participants can propose ideas that do not involve ControlNets at all. But the ideas need to be centered around Diffusion models.

## Important dates

- **29.03.** Official announcement of the community week. Make sure to fill out the [signup form].
- **31.03.** Start forming groups in #jax-diffusers-ideas channel in Discord. 
- **10.04.** Receiving acceptance e-mails, starting to collect data.
- **13.04. - 14.04. - [17.04.](https://www.youtube.com/watch?v=SOj2sxgvFe0)** Kick-off event with talks on Youtube. 
- **14.04. - 17.04.** Start providing access to TPUs. 
- **01.05.** Shutdown access to TPUs. 
- **08.05.**: Project presentations and determining the prize winners.

## Communication

All important communication will take place on our Discord server. Join the server using [this link](https://hf.co/join/discord). After you join the server, take the Diffusers role in `#role-assignment` channel and head to `#jax-diffusers-ideas` channel to share your idea as a forum post. To sign up for participation, fill out the [signup form] and we will give you access to two more Discord channels on discussions and technical support, and access to TPUs.
Important announcements of the Hugging Face, Flax/JAX, and Google Cloud team will be posted in the server.
The Discord server will be the central place for participants to post about their results, share their learning experiences, ask questions and get technical support in various obstacles they encounter.

For issues with Flax/JAX, Diffusers, Datasets or for questions that are specific to your project we will be interacting through public repositories and forums:

- Flax: [Issues](https://github.com/google/flax/issues), [Questions](https://github.com/google/flax/discussions)
- JAX: [Issues](https://github.com/google/jax/issues), [Questions](https://github.com/google/jax/discussions)
- 🤗 Diffusers: [Issues](https://github.com/huggingface/diffusers/issues), [Questions](https://discuss.huggingface.co/c/discussion-related-to-httpsgithubcomhuggingfacediffusers/63)
- 🤗 Datasets: [Issues](https://github.com/huggingface/datasets/issues), [Questions](https://discuss.huggingface.co/c/datasets/10)
- Project specific questions: Can be asked from each project's own post on #jax-diffusers-ideas channel on Discord.
- TPU related questions: `#jax-diffusers-tpu-support` channel on Discord. 
- General discussion: `#jax-diffusers-sprint channel` on Discord.
You will get access to `#jax-diffusers-tpu-support` and `#jax-diffusers-sprint` once you are accepted to attend the sprint.

When asking for help, we encourage you to post the link to [forum](https://discuss.huggingface.co) post to the Discord server, instead of directly posting issues or questions. 
This way, we make sure that the everybody in the community can benefit from your questions, even after the community sprint.

> 💡 Note: After 10th of April, if you have signed up on the google form, but you are not in the Discord channel, please leave a message on [the official forum announcement](https://discuss.huggingface.co/t/controlling-stable-diffusion-with-jax-and-diffusers-using-v4-tpus/35187/2) and ping `@mervenoyan`, `@sayakpaul`, and `@patrickvonplaten`. We might take a day to process these requests.

## Talks

We have invited prominent researchers and engineers from Google, Hugging Face, and the open-source community who are working in the Generative AI space. We will update this section with links to the talks, so keep an eye here or on Discord in diffusion models core-announcements channel and set your reminders!

### **April 13, 2023**

| Speaker	| Topic	| Time	| Video |
|---|---|---|---|
[Emiel Hoogeboom, Google Brain](https://twitter.com/emiel_hoogeboom?lang=en)	| Pixel-Space Diffusion models for High Resolution Images | 4.00pm-4.40pm CEST / 7.00am-7.40am PST| [![Youtube](https://www.youtube.com/s/desktop/f506bd45/img/favicon_32.png)](https://www.youtube.com/watch?v=iw2WCAGxdQ4) |
| [Suraj Patil, Hugging Face](https://twitter.com/psuraj28?lang=en)	| Introduction to Diffusers library	|  4.40pm-5.20pm CEST / 7.40am-08.20am PST	| [![Youtube](https://www.youtube.com/s/desktop/f506bd45/img/favicon_32.png)](https://www.youtube.com/watch?v=iw2WCAGxdQ4)
| [Ting Chen, Google Brain](https://twitter.com/tingchenai?lang=en)	| Diffusion++: discrete data and high-dimensional generation |	 5.45pm-6.25pm CEST / 08.45am-09.25am PST	| [![Youtube](https://www.youtube.com/s/desktop/f506bd45/img/favicon_32.png)](https://www.youtube.com/watch?v=iw2WCAGxdQ4) |
### **April 14, 2023**

| Speaker	| Topic	| Time	| Video |
|---|---|---|---|
| [Tim Salimans, Google Brain](https://twitter.com/timsalimans?lang=en)	| Efficient image and video generation with distilled diffusion models |   4.00pm-4.40pm CEST / 7.00am-7.40am PST| [![Youtube](https://www.youtube.com/s/desktop/f506bd45/img/favicon_32.png)](https://www.youtube.com/watch?v=6f5chgbKjSg&ab_channel=HuggingFace) |
| [Huiwen Chang, Google Research](https://scholar.google.com/citations?user=eZQNcvcAAAAJ&hl=en)	| Masked Generative Models: MaskGIT/Muse	|  4.40pm-5.20pm CEST / 7.40am-08.20am PST	| [![Youtube](https://www.youtube.com/s/desktop/f506bd45/img/favicon_32.png)](https://www.youtube.com/watch?v=6f5chgbKjSg&ab_channel=HuggingFace) |
| [Sabrina Mielke, John Hopkins University](https://twitter.com/sjmielke?lang=en)	| From stateful code to purified JAX: how to build your neural net framework |	 5.20pm-6.00pm CEST / 08.20am-09.00am PST	| [![Youtube](https://www.youtube.com/s/desktop/f506bd45/img/favicon_32.png)](https://www.youtube.com/watch?v=6f5chgbKjSg&ab_channel=HuggingFace) |

### **April 17, 2023**

| Speaker	| Topic	| Time	| Video |
|---|---|---|---|
| [Andreas Steiner, Google Brain](https://twitter.com/AndreasPSteiner)	| JAX & ControlNet |  4.00pm-4.40pm CEST / 7.00am-7.40am PST| [![Youtube](https://www.youtube.com/s/desktop/f506bd45/img/favicon_32.png)](https://www.youtube.com/watch?v=SOj2sxgvFe0) |
| [Boris Dayma, craiyon](https://twitter.com/borisdayma?lang=en)	| DALL-E Mini	|  4.40pm-5.20pm CEST / 7.40am-08.20am PST	| [![Youtube](https://www.youtube.com/s/desktop/f506bd45/img/favicon_32.png)](https://www.youtube.com/watch?v=SOj2sxgvFe0) |
| [Margaret Mitchell, Hugging Face](https://twitter.com/mmitchell_ai?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)	| Ethics of Text-to-Image |	 5.20pm-6.00pm CEST / 08.20am-09.00am PST	| [![Youtube](https://www.youtube.com/s/desktop/f506bd45/img/favicon_32.png)](https://www.youtube.com/watch?v=SOj2sxgvFe0) |

[signup form]: https://forms.gle/t3M7aNPuLL9V1sfa9

## Data and Pre-Processing

In this section, we will cover how to build your own dataset for ControlNet training.

### Prepare a large local dataset

#### Mount a disk 

If you need extra space, you can follow [this guide](https://cloud.google.com/tpu/docs/setup-persistent-disk#prerequisites) to create a persistent disk, attach it to your TPU VM, and create a directory to mount the disk. You can then use this directory to store your dataset.

#### Data preprocessing 

Here we demonstrate how to prepare a large dataset to train a ControlNet model with canny edge detection. More specifically, we provide an [example script](./dataset_tools/coyo_1m_dataset_preprocess.py) that:
* Selects 1 million image-text pairs from an existing dataset [COYO-700M](https://huggingface.co/datasets/kakaobrain/coyo-700m).
* Downloads each image and use Canny edge detector to generate the conditioning image. 
* Create a metafile that links all the images and processed images to their text captions. 

Use the following command to run the example data preprocessing script. If you've mounted a disk to your TPU, you should place your `train_data_dir` and `cache_dir` on the mounted disk

```bash
python3 coyo_1m_dataset_preprocess.py \
 --train_data_dir="/mnt/disks/persist/data" \
 --cache_dir="/mnt/disks/persist" \
 --max_train_samples=1000000 \
 --num_proc=16
```

Once the script finishes running, you can find a data folder at the specified `train_data_dir` with the below folder structure:

```
data
├── images
│   ├── image_1.png
│   ├── .......
│   └── image_1000000.jpeg
├── processed_images
│   ├── image_1.png
│   ├── .......
│   └── image_1000000.jpeg
└── meta.jsonl
```

#### Load dataset

To load a dataset from the data folder you just created, you should add a dataset loading script to your data folder. The dataset loading script should have the same name as the folder. For example, if your data folder is `data`, you should add a data loading script named `data.py`. We provide an [example data loading script](./dataset_tools/data.py) for you to use. All you need to do is to update the `DATA_DIR` with the correct path to your data folder. For more details about how to write a dataset loading script, refer to the [documentation](https://huggingface.co/docs/datasets/dataset_script).

Once the dataset loading script is added to your data folder, you can load it with: 

```python
dataset = load_dataset("/mnt/disks/persist/data", cache_dir="/mnt/disks/persist" )
```

Note that you can use the `--train_data_dir` flag to pass your data folder directory to the training script and generate your dataset automatically during the training.
 
For large datasets, we recommend generating the dataset once and saving it on the disk with

```python
dataset.save_to_disk("/mnt/disks/persist/dataset")
```

You can then reuse the saved dataset for your training by passing the `--load_from_disk` flag.

Here is an example to run a training script that will load the dataset from the disk 

```python
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/mnt/disks/persist/canny_model"
export DATASET_DIR="/mnt/disks/persist/dataset"
export DISK_DIR="/mnt/disks/persist"

python3 train_controlnet_flax.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=$DATASET_DIR \
 --load_from_disk \
 --cache_dir=$DISK_DIR \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=2 \
 --revision="non-ema" \
 --from_pt \
 --max_train_steps=500000 \
 --checkpointing_steps=10000 \
 --dataloader_num_workers=16 
 ```

### Prepare a dataset with MediaPipe and Hugging Face 

We provide a notebook ([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/community-events/blob/main/jax-controlnet-sprint/dataset_tools/create_pose_dataset.ipynb)) that shows you how to prepare a dataset for ControlNet training using [MediaPipe](https://developers.google.com/mediapipe) and Hugging Face. Specifically, in the notebook, we show:

* How to leverage MediaPipe solutions to extract pose body joints from the input images.
* Predict captions using BLIP-2 from the input images using 🤗 Transformers.
* Build and push the final dataset to the Hugging Face Hub using 🤗 Datasets. 

You can refer to the notebook to create your own datasets using other MediaPipe solutions as well. Below, we list all the relevant ones:

* [Pose Landmark Detection](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
* [Face Landmark Detection](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
* [Selfie Segmentation](https://developers.google.com/mediapipe/solutions/vision/image_segmenter)


## Training ControlNet

This is perhaps the most fun and interesting part of this document as here we show you how to train a custom ControlNet model. 

> 💡 Note: For this sprint, you are NOT restricted to just training ControlNets. We provide this training script as a reference for you to get started. 

For faster training on TPUs and GPUs you can leverage the flax training example. Follow the instructions above to get the model and dataset before running the script.

### Setting up your TPU VM

_Before proceeding with the rest of this section, you must ensure that the
email address you're using has been added to the `diffusers-jax` project on
Google Cloud Platform. If it's not the case, please let us know in the Discord server (you can tag `@sayakpaul`, `@merve`, and `@patrickvonplaten`)._

In the following, we will describe how to do so using a standard console, but you should also be able to connect to the TPU VM via IDEs, like Visual Studio Code, etc.

1. You need to install the Google Cloud SDK. Please follow the instructions on cloud.google.com/sdk.

2. Once you've installed the google cloud sdk, you should set your account by running the following command. Make sure that <your-email-address> corresponds to the gmail address you used to sign up for this event.
  
    ```bash
    $ gcloud config set account <your-email-adress>
    ```

3. Let's also make sure the correct project is set in case your email is used for multiple gcloud projects:

    ```bash
    $ gcloud config set project diffusers-jax
    ```

4. Next, you will need to authenticate yourself. You can do so by running:

    ```bash
    $ gcloud auth login
    ```

    This should give you a link to a website, where you can authenticate your gmail account.

5. Finally, you can ssh into the TPU VM! Please run the following command by setting`--zone` to `us-central2-b` and to the TPU name also sent to you in the second email.

    ```bash
    $ gcloud alpha compute tpus tpu-vm ssh <tpu-name> --zone <zone> --project hf-flax
    ```

This should ssh you into the TPU VM!

### Installing JAX

Let's first create a Python virtual environment:

```bash
$ python3 -m venv <your-venv-name>
```

We can activate the environment by running:

```bash
source ~/<your-venv-name>/bin/activate
```

Now, we can install JAX `0.4.5`:

```bash
$ pip install "jax[tpu]==0.4.5" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

To verify that JAX was correctly installed, you can run the following command:

```python
import jax
jax.device_count()
```

This should display the number of TPU cores, which should be 4 on a TPUv4-8 VM.

Then install Diffusers and the library's training dependencies:

```bash
$ pip install git+https://github.com/huggingface/diffusers.git
```

Then clone this repository and install the other dependencies:

```bash
$ git clone https://github.com/huggingface/community-events
$ cd community-events/training_scripts
$ pip install -U -r requirements_flax.txt
```

If you want to use Weights and Biases logging, you should also install `wandb` now:

```bash
pip install wandb
```
### Running the training script

Now let's download two conditioning images that we will use to run validation during the training in order to track our progress

```bash
$ wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
$ wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

We encourage you to store or share your model with the community. To use huggingface hub, please login to your Hugging Face account, or ([create one](https://huggingface.co/docs/diffusers/main/en/training/hf.co/join) if you don’t have one already):

```bash
$ huggingface-cli login
```

Make sure you have the `MODEL_DIR`,`OUTPUT_DIR` and `HUB_MODEL_ID` environment variables set. The `OUTPUT_DIR` and `HUB_MODEL_ID` variables specify where to save the model to on the Hub:

```bash
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="control_out"
export HUB_MODEL_ID="fill-circle-controlnet"
```

And finally start the training (make sure you're in the `training_scripts` directory)!

```bash
python3 train_controlnet_flax.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --validation_steps=1000 \
 --train_batch_size=2 \
 --revision="non-ema" \
 --from_pt \
 --report_to="wandb" \
 --max_train_steps=10000 \
 --push_to_hub \
 --hub_model_id=$HUB_MODEL_ID
 ```

Since we passed the `--push_to_hub` flag, it will automatically create a model repo under your huggingface account based on `$HUB_MODEL_ID`. By the end of training, the final checkpoint will be automatically stored on the hub. You can find an example model repo [here](https://huggingface.co/YiYiXu/fill-circle-controlnet).

Our training script also provides limited support for streaming large datasets from the Hugging Face Hub. In order to enable streaming, one must also set `--max_train_samples`.  Here is an example command:

```bash
python3 train_controlnet_flax.py \
  --pretrained_model_name_or_path=$MODEL_DIR \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=multimodalart/facesyntheticsspigacaptioned \
  --streaming \
  --conditioning_image_column=spiga_seg \
  --image_column=image \
  --caption_column=image_caption \
  --resolution=512 \
  --max_train_samples 50 \
  --max_train_steps 5 \
  --learning_rate=1e-5 \
  --validation_steps=2 \
  --train_batch_size=1 \
  --revision="flax" \
  --report_to="wandb"
```

Note, however, that the performance of the TPUs might get bottlenecked as streaming with `datasets` is not optimized for images. For ensuring maximum throughput, we encourage you to explore the following options:

* [Webdataset](https://webdataset.github.io/webdataset/)
* [TorchData](https://github.com/pytorch/data)
* [TensorFlow Datasets](https://www.tensorflow.org/datasets/tfless_tfds)
