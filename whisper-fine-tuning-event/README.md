# Whisper Fine-Tuning Event 🤗

Welcome to the Whisper fine-tuning event 🎙️!

For two weeks, we will endeavour to fine-tune the Whisper model to build state-of-the-art speech recognition systems in 
the languages of our choice 🗣. We will work together as a community to achieve this, helping others and learning where 
we can 🤗. If necessary and available, free access to A100 40 GB GPUs will kindly be provided by our cloud compute 
partners, [Lambda](https://lambdalabs.com) 🚀.

This document summarises all the relevant information required for the event 📋. Please read it thoroughly 
and make sure to:
- Sign-up using the [Google form](https://forms.gle/F2bpouvhDpKKisM39)
- Join the [Hugging Face Discord server](https://hf.co/join/discord) and make sure to assign yourself **@ml-4-audio** role in #role-assignment so that you can access #events channel. 

## Table of Contents

- [Introduction](#introduction)
- [Important Dates](#important-dates)
- [Launch a Lambda Cloud GPU](#launch-a-lambda-cloud-gpu)
- [Set Up an Environment](#set-up-an-environment)
- [Data and Pre-Processing](#data-and-pre-processing)
- [Fine-Tune a Whisper Model](#fine-tune-whisper)
- [Evaluation](#evaluation)
- [Building a Demo](#building-a-demo)
- [Communication and Problems](#communication-and-problems)
- [Talks](#talks)
- [Tips and Tricks](#tips-and-tricks)
- [Feedback](#feedback)

## Introduction
Whisper is a pre-trained model for automatic speech recognition (ASR) published in [September 2022](https://openai.com/blog/whisper/) 
by the authors Radford et al. from OpenAI. Pre-trained on 680,000 hours of labelled data, it demonstrates a strong ability 
to generalise to different datasets and domains. Through fine-tuning, the performance of this model can be significantly 
boosted for a given language.

In this event, we're bringing the community together to fine-tune Whisper in as many languages as possible. Our aim is 
to achieve state-of-the-art on the languages spoken by the community. Together, we can democratise speech recognition 
for all.

We are providing training scripts, notebooks, blog posts, talks and compute (where available), so you have all the 
resources you need to participate! You are free to chose your level of participation, from using the template script and setting 
it to your language, right the way through to exploring advanced training methods. We encourage you to participate to 
level that suits you best. We'll be on hand to facilitate this!

Participants are allowed to fine-tune their systems on the training data of their choice, including datasets from the 
Hugging Face Hub, web-scraped data from the internet, or private datasets. Whisper models will be evaluated 
on the "test" split of the [Common Voice 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) 
dataset for the participant's chosen language.

We believe that framing the event as a competition is fun! But at the core, the event is about
fine-tuning Whisper in as many languages as possible as a community. We want to foster an environment where we 
work together, help each other solve bugs, share important findings and ultimately learn something new.

This README contains all the information you need for the event. It is structured such that you can read it sequentially, 
section-by-section. **We recommend that you read the document once from start to finish before running any code.** This will 
give you an idea of where to look for the relevant information and an idea of how the event is going to run.

## Important Dates

- *Introduction Talk*: 2nd December 2022
- *Sprint start*: 5th December 2022
- *Speaker Events*: 5th December 2022
- *Sprint end*: 19th December 2022
- *Results*: 23rd December 2022

## Launch a Lambda Cloud GPU
Where possible, we encourage you to fine-tune Whisper on a local GPU machine. This will mean a faster set-up and more 
familiarity with your device. If you are running on a local GPU machine, you can skip ahead to the next section: [Set Up an Environment](#set-up-an-environment). 

The training scripts can also be run as a notebook through Google Colab. We recommend you train on Google Colab if you 
have a "Colab Pro" or "Pro+" subscription. This is to ensure that you receive a sufficiently powerful GPU on your Colab for 
fine-tuning Whisper. If you wish to fine-tune Whisper through Google Colab, you can skip ahead to the section: [Data and Pre-Processing](#data-and-pre-processing). 

If you do not have access to a local GPU or Colab Pro/Pro+, we'll endeavour to provide you with a cloud GPU instance.
We've partnered up with Lambda to provide cloud compute for this event. They'll be providing the latest NVIDIA A100 
40 GB GPUs, so you'll be loaded with some serious firepower! The Lambda API makes it easy to spin-up and launch 
a GPU instance. In this section, we'll go through the steps for spinning up an instance one-by-one.

<p align="center" width="100%">
    <img width="50%" src="https://raw.githubusercontent.com/sanchit-gandhi/codesnippets/main/hf_x_lambda.jpg">
</p>

This section is split into three parts:

1. [Signing-Up with Lambda](#signing-up-with-lambda)
2. [Creating a Cloud Instance](#creating-a-cloud-instance)
3. [Deleting a Cloud Instance](#deleting-a-cloud-instance)

### Signing-Up with Lambda

1. Create an account with Lambda using your email address of choice: https://cloud.lambdalabs.com/sign-up. If you already have an account, skip to step 2.
2. Using this same email address, email `cloud@lambdal.com` with the Subject line: `Lambda cloud account for HuggingFace Whisper event - payment authentication and credit request`.
3. Each user who emails as above will receive $110 in credits (amounting to 100 hours of 1x A100 usage).
4. Register a valid payment method with Lambda in order to redeem the credits (see instructions below).

100 hours of 1x A100 usage should enable you to complete 5-10 fine-tuning runs. To redeem these credits, you will need to 
authorise a valid payment method with Lambda. Provided that you remain within $110 of compute spending, your card **will not** 
be charged 💸. Registering your card with Lambda is a mandatory sign-up step that we unfortunately cannot bypass. But we 
reiterate: you will not be charged provided you remain within $110 of compute spending!

Follow steps 1-4 in the next section [Creating a Cloud Instance](#creating-a-cloud-instance) to register your
card. If you experience issues with registering your card, contact the Lambda team on Discord (see [Communications and Problems](#communication-and-problems)).

In order to maximise the free GPU hours you have available for training, we advise that you shut down GPUs when you are 
not using them and closely monitor your GPU usage. We've detailed the steps you can follow to achieve this in [Deleting a Cloud Instance](#deleting-a-cloud-instance).

### Creating a Cloud Instance
Estimated time to complete: 5 mins

*You can also follow our video tutorial to set up a cloud instance on Lambda* 👉️ [YouTube Video](https://www.youtube.com/watch?v=Ndm9CROuk5g&list=PLo2EIpI_JMQtncHQHdHq2cinRVk_VZdGW)

1. Click the link: https://cloud.lambdalabs.com/instances
2. You'll be asked to sign in to your Lambda account (if you haven't done so already).
3. Once on the GPU instance page, click the purple button "Launch instance" in the top right.
4. Verify a payment method if you haven't done so already. IMPORTANT: if you have followed the instructions in the previous section, you will have received $110 in GPU credits. Exceeding 100 hours of 1x A100 usage may incur charges on your credit card. Contact the Lambda team on Discord if you have issues authenticating your payment method (see [Communications and Problems](#communication-and-problems))
5. Launching an instance:
   1. In "Instance type", select the instance type "1x A100 (40 GB SXM4)"
   2. In "Select region", select the region with availability closest to you.
   3. In "Select filesystem", select "Don't attach a filesystem".
6. You will be asked to provide your public SSH key. This will allow you to SSH into the GPU device from your local machine.
   1. If you’ve not already created an SSH key pair, you can do so with the following command from your local device: 
      ```bash
      ssh-keygen
      ```
   2. You can find your public SSH key using the command: 
      ```bash
      cat ~/.ssh/id_rsa.pub
      ```
      (Windows: `type C:UsersUSERNAME.sshid_rsa.pub` where `USERNAME` is the name of your user)
   4. Copy and paste the output of this command into the first text box
   5. Give your SSH key a memorable name (e.g. `sanchits-mbp`)
   6. Click "Add SSH Key"
7. Select the SSH key from the drop-down menu and click "Launch instance"
8. Read the terms of use and agree
9. We can now see on the "GPU instances" page that our device is booting up!
10. Once the device status changes to "✅ Running", click on the SSH login ("ssh ubuntu@..."). This will copy the SSH login to your clipboard.
11. Now open a new command line window, paste the SSH login, and hit Enter.
12. If asked "Are you sure you want to continue connecting?", type "yes" and press Enter.
13. Great! You're now SSH'd into your A100 device! We're now ready to set up our Python environment!

You can see your total GPU usage from the Lambda cloud interface: https://cloud.lambdalabs.com/usage

Here, you can see the total charges that you have incurred since the start of the event. We advise that you check your 
total on a daily basis to make sure that it remains below the credit allocation of $110. This ensures that you are 
not inadvertently charged for GPU hours.

If you are unable to SSH into your Lambda GPU in step 11, there is a workaround that you can try. On the [GPU instances page](https://cloud.lambdalabs.com/instances), 
under the column "Cloud IDE", click the button "Launch". This will launch a Jupyter Lab on your GPU which will be displayed in your browser. In the 
top left-hand corner, click "File" -> "New" -> "Terminal". This will open up a new terminal window. You can use this 
terminal window to set up your Python environment in the next section [Set Up an Environment](#set-up-an-environment).

### Deleting a Cloud Instance

100 1x A100 hours should provide you with enough time for 5-10 fine-tuning runs (depending on how long you train for 
and which size models). To maximise the GPU time you have for training, we advise that you shut down GPUs over prolonged 
periods of time when they are not in use. Leaving a GPU running accidentally over the weekend will incur 48 hours of 
wasted GPU hours. That's nearly half of your compute allocation! So be smart and shut down your GPU when you're not training.

Creating an instance and setting it up for the first time may take up to 20 minutes. Subsequently, this process will 
be much faster as you gain familiarity with the steps, so you shouldn't worry about having to delete a GPU and spinning one 
up the next time you need one. You can expect to spin-up and delete 2-3 GPUs over the course of the fine-tuning event.


We'll quickly run through the steps for deleting a Lambda GPU. You can come back to these steps after you've 
performed your first training run and you want to shut down the GPU:

1. Go to the instances page: https://cloud.lambdalabs.com/instances
2. Click the checkbox on the left next to the GPU device you want to delete
3. Click the button "Terminate" in the top right-hand side of your screen (under the purple button "Launch instance")
4. Type "erase data on instance" in the text box and press "ok"

Your GPU device is now deleted and will stop consuming GPU credits.

## Set Up an Environment
Estimated time to complete: 5 mins

*Follow along our video tutorial detailing the set up* 👉️ [YouTube Video](https://www.youtube.com/playlist?list=PLo2EIpI_JMQtzC5feNpqQL7eToYKcOxYf)

The Whisper model should be fine-tuned using **PyTorch**, **🤗 Transformers**, and, **🤗 Datasets**. In this 
section, we'll cover how to set up an environment with the required libraries. This section assumes that you are SSH'd 
into your GPU device. This section does not apply if you are fine-tuning the Whisper model in a Google Colab.

If you are returning to this section having read through it previously and want to quickly set up an environment, you 
can do so in one call by executing the following code cell. If this is your first time setting up an environment, we 
recommend you read this section to understand the steps involved.

```bash
sudo add-apt-repository -y ppa:jonathonf/ffmpeg-4
sudo apt update
sudo apt install -y ffmpeg

sudo apt-get install git-lfs

python3 -m venv hf_env
source hf_env/bin/activate
echo "source ~/hf_env/bin/activate" >> ~/.bashrc

git clone https://github.com/huggingface/community-events.git
pip install -r community-events/whisper-fine-tuning-event/requirements.txt

git config --global credential.helper store
huggingface-cli login
 ```

### Unix Libraries

First, we need to make sure we have the required NVIDIA drivers installed. We can check that we have these drivers 
through the following command:

```bash
nvidia-smi
```

This should print a table with our NVIDIA driver version and CUDA version, and should work out of the box for Lambda GPUs!
If you get an error running this command, refer to your device manual for installing the required NVIDIA driver.

Before installing the required libraries, we'd need to install and update the Unix package `ffmpeg` to version 4:

 ```bash
sudo add-apt-repository -y ppa:jonathonf/ffmpeg-4
sudo apt update
sudo apt install -y ffmpeg
 ```

We'll also need the package `git-lfs` to push large model weights to the Hugging Face Hub. To check whether 
`git-lfs` is installed, simply run:

```bash
git-lfs -v
```

The output should show something like `git-lfs/2.13.2 (GitHub; linux amd64; go 1.15.4)`. If your console states that 
the `git-lfs` command was not found, you can install it via: 

```bash
sudo apt-get install git-lfs
```

### Python Libraries

We recommend installing the required libraries in a Python virtual environment. If you're unfamiliar with Python virtual 
environments, check out the [official user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Let's define a variable that denotes the name of the environment we're going to create:

```bash
env_name=<your-venv-name>
```

We can create a virtual environment (venv) with this name using the following command:

```bash
python3 -m venv $env_name
```

We'll instruct our bash shell to activate the venv by default by placing the venv source command in `.bashrc`:

```bash
echo "source ~/$env_name/bin/activate" >> ~/.bashrc
```

Re-launching the bash shell will activate the venv:

```bash
bash
```

Great! We can see that our venv name is at the start of our command line - this means that we're operating from 
within the venv. We can now go ahead and start installing the required Python packages to our venv.

The [`requirements.txt`](https://github.com/huggingface/community-events/blob/main/whisper-fine-tuning-event/requirements.txt) 
file in this directory has all the necessary Python packages we need to fine-tune Whisper, including PyTorch, Transformers 
and Datasets. We'll install all the packages in this file through one `pip install` command.

First, let's clone the `community-events` repository to our device:

```bash
git clone https://github.com/huggingface/community-events.git
```

Now we can install the packages from the `requirements.txt` file using the following command:

```bash
pip install -r community-events/whisper-fine-tuning-event/requirements.txt
```

Note: when installing packages, you might see warnings such as:

```bash
  error: invalid command 'bdist_wheel'
  ----------------------------------------
  ERROR: Failed building wheel for audioread
```

This is perfectly ok! It does not affect our installation.

We can check that above steps installed the correct version of PyTorch to match our CUDA version. The following command should return True:

```python
python -c "import torch; print(torch.cuda.is_available())"
```

If the above command does not return True, refer to the [official instructions](https://pytorch.org/get-started/locally/) for installing PyTorch.

We can now verify that `transformers` and `datasets` have been correctly installed. First, launch a Python shell:

```bash
python
```

Running the following code cell will load one sample of the [Common Voice](https://huggingface.co/datasets/common_voice) 
dataset from the Hugging Face Hub and perform a forward pass of the "tiny" Whisper model:

```python
import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration
from datasets import load_dataset

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

common_voice = load_dataset("common_voice", "en", split="validation", streaming=True)

inputs = feature_extractor(next(iter(common_voice))["audio"]["array"], sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features

decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
logits = model(input_features, decoder_input_ids=decoder_input_ids).logits

print("Environment set up successful?", logits.shape[-1] == 51865)

```

If the final check returns True, the libraries have been installed correctly. Finally, exit the Python shell:

```python
quit()
```

The last thing we need to do is link our Hugging Face account. Run the command:

```bash
git config --global credential.helper store
huggingface-cli login
```

And then enter an authentication token from https://huggingface.co/settings/tokens. Create a new token if you do not have 
one already. You should make sure that this token has "write" privileges.

## Data and Pre-Processing

In this section, we will cover how to find suitable training data and the necessary steps to pre-process it. 
If you are new to the 🤗 Datasets library, we highly recommend reading the comprehensive blog post: [A Complete Guide To Audio Datasets](https://huggingface.co/blog/audio-datasets). 
This blog post will tell you everything you need to know about 🤗 Datasets and its one-line API.

### Data

Whisper models will be evaluated on the `"test"` split of the [Common Voice 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) 
dataset. Any data can be used to fine-tune the Whisper model **except Common Voice's `"test"` split**. This exception 
extends to all Common Voice versions, as the test split of legacy Common Voice releases often overlaps with the 
latest one. For instance, the test split of Common Voice 10 is largely the same as that of Common Voice 11.

So, the test data:

```python
load_dataset("mozilla-foundation/common_voice_11_0", "en", split="test", use_auth_token=True)
```

More or less includes the same data as:

```python
load_dataset("mozilla-foundation/common_voice_10_0", "en", split="test", use_auth_token=True)
```

And **neither** are allowed for training purposes. However, we strongly encourage participants to make use of the other 
Common Voice splits for training data, such as the `"train"` and `"validation"` splits:

```python
load_dataset("mozilla-foundation/common_voice_10_0", "en", split="train", use_auth_token=True)
```

For most languages, the `"train"` split of Common Voice 11 dataset offers a reasonable amount of training data. 
For low-resource languages, it is normal procedure to combine the `"train"` and `"validation"` splits to give a larger 
training corpus:

```python
load_dataset("mozilla-foundation/common_voice_10_0", "en", split="train+validation", use_auth_token=True)
```

This notation for combining splits (`"split_a+split_b"`) is consistent for all resources in the event. You can combine 
splits in this same way using the fine-tuning scripts in the following section [Fine-Tune Whisper](#fine-tune-whisper).

If combining the `"train"` and `"validation"` splits of the Common Voice 11 dataset still gives insufficient training 
data for your language, you can explore using other datasets on the Hub to train your model and try 
[Mixing Datasets](#mixing-datasets-optional) to give larger training splits.

### Streaming Mode

Audio datasets are very large. This causes two issues:
1. They require a significant amount of **storage space** to download.
2. They take a significant amount of **time** to download and process.

The storage and time requirements present limitations to most speech researchers. For example, downloading the English 
subset of the Common Voice 11 dataset (2,300 hours) requires upwards of 200GB of disk space and up to several hours 
of download time. For these reasons, we **do not** recommend that you run the following code cell! 
```python
from datasets import load_dataset

common_voice = load_dataset("mozilla-foundation/common_voice_11_0", "en", use_auth_token=True)

# we have to wait several hours until the entire dataset is downloaded before we can access the first sample... 
print(next(iter(common_voice["train"])))
```

However, both these issues can be solved with 🤗 Datasets. Rather than downloading the whole dataset at once, we 
load individual samples as we cycle over the dataset, in a process called _streaming_. Since the data is loaded 
progressively as we iterate over the dataset, we can get started with a dataset as soon as the first sample is ready. 
This way, we don't have to wait for the entire dataset to download before we can run our code! We are also free of any 
disk space contraints: once we're done with a sample, we discard it and load the next one to memory. This way, we only 
have the data when we need it, and not when we don't!

Streaming is enabled by passing the argument `streaming=True` to the `load_dataset` function. We can then use our 
audio datasets in much the same way as before! For these reasons, **we highly recommend** that you try out the following 
code cell! Just make sure you've accepted the Common Voice 11 [terms of use](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) on the Hugging Face Hub.

```python
from datasets import load_dataset

common_voice = load_dataset("mozilla-foundation/common_voice_11_0", "en", use_auth_token=True, streaming=True)

# get the first sample of the dataset straight away!
print(next(iter(common_voice["train"])))
```

The examples for this event rely heavily on streaming mode to fine-tune Whisper. With streaming mode, we can use **any 
speech recognition dataset on the Hub with just 20GB of disk space**. As a speech recognition practitioner, this is 
game changing! The largest speech recognition datasets are available to us regardless of our device disk space. We are 
extremely excited to be showcasing streaming mode in this event and hope that you will enjoy using it.

There is one caveat to streaming mode. When downloading a dataset to disk, the processed data is saved to our cache. If 
we want to re-use this data, we can directly load the processed data from cache, skipping the download and processing 
steps. Consequently, we only have to perform the downloading and processing operations once. With streaming mode, the 
data is not downloaded to disk. Thus, neither the download nor pre-processing are cached. If we want to re-use the data, 
the streaming steps must be repeated, with the audio files loaded and pre-processed again. Therefore, we recommend not 
using streaming mode if your dataset is small (< 10 hours). In this case, it is faster to download and pre-process the 
dataset in the conventional way once at the start, and then re-use it at each epoch. We provide pointers for disabling 
streaming mode in the section [Fine-Tune Whisper](#fine-tune-whisper).

If you want to read more about streaming mode, we 
recommend you check out the aforementioned blog post: [A Complete Guide To Audio Datasets](https://huggingface.co/blog/audio-datasets).

### Pre-Processing

Data pre-processing is a very grey area when it comes to speech recognition. In this section, we'll try to make the 
situation as clear as possible for you as participants.

The Common Voice dataset is both cased and punctuated:

```python
print(next(iter(common_voice["train"]))["sentence"])
```
**Print Output:**
```
Why does Melissandre look like she wants to consume Jon Snow on the ride up the wall?
```

If we train the Whisper model on the raw Common Voice dataset, it will learn to predict casing and punctuation. This is 
great when we want to use out model for actual speech recognition applications, such as transcribing meetings or 
dictation, as the predicted transcriptions will be formatted with casing and punctuation.

However, we also have the option of 'normalising' the dataset to remove any casing and punctuation. Normalising the 
dataset makes the speech recognition task easier: the model no longer needs to distinguish between upper and lower case 
characters, or have to predict punctuation from the audio data alone. Because of this, the word error rates are 
naturally lower (meaning the results are better). The Whisper paper demonstrates the drastic effect that normalising 
transcriptions can have on WER results (_c.f._ Section 4.4 of the [Whisper paper](https://cdn.openai.com/papers/whisper.pdf)). 
But while we get lower WERs, we can't necessarily use our model in production. The lack of casing and punctuation makes 
the predicted text from the model much harder to read. We would need additional post-processing models to restore casing and 
punctuation in our predictions if we wanted to use it for downstream applications.

There is a happy medium between the two: we can train our systems on cased and normalised transcriptions, and then 
evaluate them on normalised text. This way, we train our systems to predict fully formatted text, but also benefit from 
the WER improvements we get by normalising the transcriptions.

The choice of whether you normalise the transcriptions is ultimately down to you. We recommend training on un-normalised 
text and evaluating on normalised text to get the best of both worlds. Since those choices are not always obvious, feel 
free to ask on Discord or (even better) post your question on the [forum](https://discuss.huggingface.co).

| Train         | Eval          | Pros                                                           | Cons                                     |
|---------------|---------------|----------------------------------------------------------------|------------------------------------------|
| Un-normalised | Un-normalised | * Predict casing + punctuation<br>* One logic for train / eval | * WERs are higher                        |
| Un-normalised | Normalised    | * Predict casing + punctuation<br>* WERs are lower             | * Different logic for train / eval       |
| Normalised    | Normalised    | * One logic for train / eval<br>* WERs are lower               | * No casing / punctuation in predictions |

With the provided training scripts, it is trivial to toggle between removing or retaining punctuation and casing, 
requiring at most three lines of code change. Switching between the different modes is explained in more detail in the 
following section [Fine-Tune Whisper](#fine-tune-whisper).

If you want to find out more about pre- and post-processing for speech recognition, we refer you in the direction of 
the paper: [ESB: A Benchmark For Multi-Domain End-to-End Speech Recognition](https://arxiv.org/abs/2210.13352).

The following two subsections are optional. They cover how you can mix datasets to form larger training splits and how 
you can use custom data to fine-tune your model. If the Common Voice 11 dataset has sufficient data in your language to 
fine-tune your model, you can skip to the next section [Fine-Tune Whisper](#fine-tune-whisper).

### Mixing Datasets (optional)

If the Common Voice 11 dataset contains insufficient training data to fine-tune Whisper in your language, you can explore mixing 
different datasets to create a larger combined training set. Incorporating supplementary training data is almost always beneficial for training. 
The Whisper paper demonstrates the significant effect that increasing the amount of training data can have on downstream 
performance (_c.f._ Section 4.2 of the [paper](https://cdn.openai.com/papers/whisper.pdf)). There are a number of datasets 
that are available on the Hugging Face Hub that can be downloaded via the 🤗 Datasets library in much the same way as 
Common Voice 11.

We recommend selecting from the following four datasets on the Hugging Face Hub for multilingual speech recognition:

| Dataset                                                                                       | Languages | Casing | Punctuation |
|-----------------------------------------------------------------------------------------------|-----------|--------|-------------|
| [Common Voice 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)       | 100+      | ✅      | ✅           |
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli)                               | 15        | ❌      | ✅           |
| [Multilingual LibriSpeech](https://huggingface.co/datasets/facebook/multilingual_librispeech) | 6         | ❌      | ❌           |
| [FLEURS](https://huggingface.co/datasets/google/fleurs)                                       | 100+      | ✅      | ✅           |


<!---
<details>
<summary>

#### Common Voice 11

</summary>

[Common Voice 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) is a crowd-sourced 
open-licensed speech dataset where speakers record text from Wikipedia in various languages. Since anyone can contribute 
recordings, there is significant variation in both audio quality and speakers. The audio conditions are challenging, with 
recording artefacts, accented speech, hesitations, and the presence of foreign words. The transcriptions are both cased 
and punctuated. As of version 11, there are over 100 languages available, both low and high-resource.
</details>
<details>
<summary>

#### VoxPopuli

</summary>

[VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) is a large-scale multilingual speech corpus consisting 
of data sourced from 2009-2020 European Parliament event recordings. Consequently, it occupies the unique domain of 
oratory, political speech, largely sourced from non-native speakers. It contains labelled audio-transcription data for 
15 European languages. The transcriptions are punctuated but not cased.
</details>
<details>
<summary>

#### Multilingual LibriSpeech

</summary>

[Multilingual LibriSpeech](https://huggingface.co/datasets/facebook/multilingual_librispeech) is the multilingual 
equivalent of the [LibriSpeech ASR](https://huggingface.co/datasets/librispeech_asr) corpus. It comprises a large corpus 
of read audiobooks taken from the [LibriVox](https://librivox.org/) project, making it a suitable dataset for academic 
research. It contains data split into eight high-resource languages - English, German, Dutch, Spanish, French, Italian, 
Portuguese and Polish. The transcriptions are neither punctuated nor cased.
</details>
<details>
<summary>

#### FLEURS

</summary>

[FLEURS](https://huggingface.co/datasets/google/fleurs) (Few-shot Learning Evaluation of Universal Representations of 
Speech) is a dataset for evaluating speech recognition systems in 102 languages, including many that are classified as 
'low-resource'. The data is derived from the [FLoRes-101](https://arxiv.org/abs/2106.03193) dataset, a machine 
translation corpus with 3001 sentence translations from English to 101 other languages. Native speakers are recorded 
narrating the sentence transcriptions in their native language. The recorded audio data is paired with the sentence 
transcriptions to yield a multilingual speech recognition over all 101 languages. The training sets contain 
approximately 10 hours of supervised audio-transcription data per language. Transcriptions come in two formats: un-normalised 
(`"raw_transcription"`) and normalised (`"transcription"`).
</details>

The previously mentioned blog post provides a more in-depth explanation of the main English speech recognition, 
multilingual speech recognition and speech translation datasets on the Hub: [A Complete Guide To Audio Datasets](https://huggingface.co/blog/audio-datasets#a-tour-of-audio-datasets-on-the-hub)  

You can also explore all speech recognition datasets on the Hub to find one suited for your language and needs: [ASR datasets on the Hub](https://huggingface.co/datasets?task_categories=task_categories:automatic-speech-recognition&sort=downloads).
--->

You can try training on these datasets individually, or mix them to form larger train sets.

When mixing datasets, you should ensure the transcription format is consistent across datasets. For example, if you mix 
Common Voice 11 (cased + punctuated) with VoxPopuli (un-cased + punctuated), you will need to lower-case **all the text** 
for both training and evaluation, such that the transcriptions are consistent across training samples (un-cased + punctuated). 

Likewise, if mixing Common Voice 11 (cased + punctuated) with Multilingual LibriSpeech (un-cased + un-punctuated), you 
should make sure to remove all casing and punctuation in **all the text** for both training and evaluation, such that 
all transcriptions are un-cased and un-punctuated for all training samples.

Having a mismatch in formatting for different training samples can reduce the final performance of your fine-tuned Whisper 
model.

If you want to combine multiple datasets for training, you can refer to the code-snippet provided for interleaving 
datasets with streaming mode: [interleave_streaming_datasets.ipynb](https://github.com/huggingface/community-events/blob/main/whisper-fine-tuning-event/interleave_streaming_datasets.ipynb).

### Custom Data (optional)

In addition to publicly available data on the Hugging Face Hub, participants can also make use of their own audio data 
for training. When using your own audio data, please make sure that you **are allowed to use the audio data**. For 
instance, if the audio data is taken from media platforms, such as YouTube, please verify that the media platform and 
the owner of the data have given their approval to use the audio data in the context of machine learning research. If 
you are not sure whether the data you want to use has the appropriate licensing, please contact the Hugging Face team 
on Discord.

<!--- TODO: VB - tutorial for adding own data via audio folder --->

## Fine-Tune Whisper

Throughout the event, participants are encouraged to leverage the official pre-trained [Whisper checkpoints](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=downloads&search=whisper).
The Whisper checkpoints come in five configurations of varying model sizes.
The smallest four are trained on either English-only or multilingual data.
The largest checkpoint is multilingual only. The checkpoints are summarised in the following table with links to the 
models on the Hugging Face Hub:

| Size   | Layers | Width | Heads | Parameters | English-only                                         | Multilingual                                      |
|--------|--------|-------|-------|------------|------------------------------------------------------|---------------------------------------------------|
| tiny   | 4      | 384   | 6     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny)  |
| base   | 6      | 512   | 8     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)   |
| small  | 12     | 768   | 12    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)  |
| medium | 24     | 1024  | 16    | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium) |
| large  | 32     | 1280  | 20    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)  |

The English-only checkpoints should be used for English speech recognition. For all other languages, one should use the 
multilingual checkpoints.

We recommend using the tiny model for rapid prototyping. **We advise that the small or medium checkpoints are used for 
fine-tuning**. These checkpoints achieve comparable performance to the large checkpoint, but can be trained much faster 
(and hence for much longer!).

A complete guide to Whisper fine-tuning can be found in the blog post: [Fine-Tune Whisper with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper).
While it is not necessary to have read this blog post before fine-tuning Whisper, it is strongly advised to gain 
familiarity with the fine-tuning code.

There are three ways in which you can execute the fine-tuning code:
1. [Python Script](#python-script)
2. [Jupyter Notebook](#jupyter-notebook)
3. [Google Colab](#google-colab)

1 and 2 are applicable when running on a local GPU or cloud GPU instance (such as on Lambda). 3 applies if you have 
a Google Colab Pro/Pro+ subscription and want to run training in a Google Colab. The proceeding instructions for running 
each of these methods are quite lengthy. Feel free to read through each of them to get a better idea for which one you 
want to use for training. Once you've read through, we advise you pick one method and stick to it!

For the walk-through, we'll assume that we're fine-tuning the Whisper model on Spanish ("es") on the Common Voice 11 
dataset. We'll point out where you'll need to change variables to run the script for your language of choice.

Before jumping into any training, make sure you've accepted the Common Voice 11 [terms of use](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) 
on the Hugging Face Hub.

### Python Script
*Checkout the video tutorial detailing how to fine-tune your whisper model via the CLI* 👉️ [YouTube Video](https://www.youtube.com/playlist?list=PLo2EIpI_JMQuKpnFm1ntcLKP6gq0l0f1Q)

1. **Create a model repository**

The steps for running training with a Python script assume that you are SSH'd into your GPU device and have set up 
your environment according to the previous section [Set Up an Environment](#set-up-an-environment).

First, we need to create a model repository on the Hugging Face Hub. This repository will contain all the required files 
to reproduce the training run, alongside model weights, training logs and a README.md card. You can either create a model 
repository directly on the Hugging Face Hub using the link: https://huggingface.co/new Or, via the CLI. Here, we'll show 
how to use the CLI. 

Let's pick a name for our fine-tuned Whisper model: *whisper-small-es*. We can run the following command to create a 
repository under this name.

```bash
huggingface-cli repo create whisper-small-es
```
(change "es" to your language code)

We can now see the model on the Hub, *e.g.* under https://huggingface.co/sanchit-gandhi/whisper-small-es

Let's clone the repository so that we can place our training script and model weights inside:

```bash
git lfs install
git clone https://huggingface.co/sanchit-gandhi/whisper-small-es
```

(be sure to change the repo address to `https://huggingface.co/<your-user-name>/<your-repo-name>`)

We can then enter the repository using the `cd` command:

```bash
cd whisper-small-es
```

2. **Add training script and `run` command**

We encourage participants to add all the relevant files for training directly to the model repository. This way, 
training runs are fully reproducible.

We provide a Python training script for fine-tuning Whisper with 🤗 Datasets' streaming mode: [`run_speech_recognition_seq2seq_streaming.py`](https://github.com/huggingface/community-events/blob/main/whisper-fine-tuning-event/run_speech_recognition_streaming.py)
This script can be copied to your model repository with the following command:

```bash
cp ~/community-events/whisper-fine-tuning-event/run_speech_recognition_seq2seq_streaming.py .
```

This will download a copy of the training script to your model repository.

We can then define the model, training and data arguments for fine-tuning:

```bash
echo 'python run_speech_recognition_seq2seq_streaming.py \
	--model_name_or_path="openai/whisper-small" \
	--dataset_name="mozilla-foundation/common_voice_11_0" \
	--dataset_config_name="es" \
	--language="spanish" \
	--train_split_name="train+validation" \
	--eval_split_name="test" \
	--model_index_name="Whisper Small Spanish" \
	--max_steps="5000" \
	--output_dir="./" \
	--per_device_train_batch_size="64" \
	--per_device_eval_batch_size="32" \
	--logging_steps="25" \
	--learning_rate="1e-5" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--eval_steps="1000" \
	--save_strategy="steps" \
	--save_steps="1000" \
	--generation_max_length="225" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="sentence" \
	--freeze_feature_encoder="False" \
	--report_to="tensorboard" \
	--metric_for_best_model="wer" \
	--greater_is_better="False" \
	--load_best_model_at_end \
	--gradient_checkpointing \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--do_normalize_eval \
	--streaming \
	--use_auth_token \
	--push_to_hub' >> run.sh
```

Make sure to change the `--dataset_config_name` and `--language` to the correct values for your language! See also how 
we combine the train and validation splits as `--train_split_name="train+validation"`. This is recommended for low-resource 
languages (it probably isn't strictly necessary for Spanish, where the `"train"` split for Common Voice 11 contains 
ample training data). We also assign a `"model_index_name"` - a pretty name that will go on the model card. If you are 
training on a very small dataset (< 10 hours), it is advisable to disable streaming mode: `--streaming="False"`.

We provide the train/eval batch sizes for the "small" checkpoint fine-tuned on a 1x A100 device. Depending on your device and checkpoint, 
you might need to lower these values. Refer to the subsection [Recommended Training Configurations](#recommended-training-configurations) 
for suggested batch-sizes for other devices and checkpoints.

3. **Launch training 🚀**

We recommend running training through a `tmux` session. This means that training won't be interrupted when you close 
your SSH connection. To start a `tmux` session named `mysession`: 

```bash
tmux new -s mysession
```
(if `tmux` is not installed, you can install it through: `sudo apt-get install tmux`)

Once in the `tmux` session, we can launch training:

```bash
bash run.sh
```

Training should take approximately 8 hours, with a final cross-entropy loss of **1e-4** and word error rate of **32.6%**.

Since we're in a `tmux` session, we're free to close our SSH window without stopping training!

If you close your SSH connection and want to rejoin the `tmux` window, you can SSH into your GPU and then connect to 
your session with the following command:

```bash
tmux a -t mysession
```

It will be like you never left! 

`tmux` guide: https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/

### Jupyter Notebook
*We've detailed these steps in a video tutorial to help you get up to speed faster* 👉️ [YouTube Video](https://www.youtube.com/playlist?list=PLo2EIpI_JMQs9z-N4v8L_Jb4KF6kAkylX)

1. **SSH port forwarding**

The steps for running training with a Python script assume that you have set up your environment according to the 
previous section [Set Up an Environment](#set-up-an-environment) and are **not** SSH'd into your GPU device. If you are 
SSH'd into your GPU device, you can close this SSH window and start from your local machine.

The command to SSH into our GPU looked something as follows:

```bash
ssh ubuntu@104.171.202.236
```

When running a Jupyter Notebook, we need to "forward" the SSH port from the remote port to the local one. This amounts 
to adding `-L 8888:localhost:8888` to the end of our SSH command. We can SSH into our remote machine using this modified 
SSH command:

```bash
ssh ubuntu@104.171.202.236 -L 8888:localhost:8888
```

Be sure to change the `ssh ubuntu@...` part to your corresponding SSH command, it's simply the `-L 8888:localhost:8888` 
part added onto the end that is new. If you want to find out more about SSH port forwarding, we recommend you read the guide: 
[SSH/OpenSSH/PortForwarding](https://help.ubuntu.com/community/SSH/OpenSSH/PortForwarding).

2. **Create a model repository (copied from previous subsection [Python Script](#python-script))**

First, we need to create a model repository on the Hugging Face Hub. This repository will contain all the required files 
to reproduce the training run, alongside model weights, training logs and a README.md card. 

You can either create a model repository directly on the Hugging Face Hub using the link: https://huggingface.co/new
Or, via the CLI. Here, we'll show how to use the CLI. 

Let's pick a name for our fine-tuned Whisper model: *whisper-small-es*. We can run the following command to create a 
repository under this name.

```bash
huggingface-cli repo create whisper-small-es
```
(change "es" to your language code)

We can now see the model on the Hub, *e.g.* under https://huggingface.co/sanchit-gandhi/whisper-small-es

Let's clone the repository so that we can place our training script and model weights inside:

```bash
git lfs install
git clone https://huggingface.co/sanchit-gandhi/whisper-small-es
```

(be sure to change the repo address to `https://huggingface.co/<your-user-name>/<your-repo-name>`)

We can then enter the repository using the `cd` command:

```bash
cd whisper-small-es
```

3. **Add notebook**

We encourage participants to add all the training notebook directly to the model repository. This way, 
training runs are fully reproducible.

We provide an iPython notebook for fine-tuning Whisper with 🤗 Datasets' streaming mode: [`fine-tune-whisper-streaming.ipynb`](https://github.com/huggingface/community-events/blob/main/whisper-fine-tuning-event/fine-tune-whisper-streaming.ipynb)
This notebook can be copied to your model repository with the following command:

```bash
cp ~/community-events/whisper-fine-tuning-event/fine-tune-whisper-streaming.ipynb .
```

If you are fine-tuning Whisper on a very small dataset (< 10 hours), it is advised that you use the non-streaming notebook 
[`fine-tune-whisper-non-streaming.ipynb`](https://github.com/huggingface/community-events/blob/main/whisper-fine-tuning-event/fine-tune-whisper-non-streaming.ipynb) 
(see section [Streaming Mode](#streaming-mode)). This notebook can be copied to your model repository with the following 
command:

```bash
cp ~/community-events/whisper-fine-tuning-event/fine-tune-whisper-non-streaming.ipynb .
```

4. **Launch Jupyter**

First, we need to make sure `jupyterlab` is installed:

```bash
pip install jupyterlab
```

We can then link `jupyter lab` to our venv:
```bash
python -m ipykernel install --user --name=<your-venv-name>
```

We recommend running training through a `tmux` session. This means that training won't be interrupted when you close 
your SSH connection. To start a `tmux` session named `mysession`:

```bash
tmux new -s mysession
```
(if `tmux` is not installed, you can install it through: `sudo apt-get install tmux`)

Once in the `tmux` session, we can launch `jupyter lab`:

```bash
jupyter lab --port 8888
```

5. **Open Jupyter in browser**

Now, this is the hardest step of running training from a Jupyter Notebook! Open a second terminal window on your local 
machine and SSH into your GPU again. This time, it doesn't matter whether we include the `-L 8888:localhost:8888` part, 
the important thing is that you re-enter your GPU device in a new SSH window.

Once SSH'd into your GPU, view all running `jupyter lab` sessions:

```bash
jupyter lab list
```

Copy the URL for the lab corresponding to port 8888 your clipboard, it will take the form `http://localhost:8888/?token=...`. 
On your local desktop, open a web browser window (Safari, Firefox, Chrome, etc.). Paste the URL into the browser web 
address bar and press Enter.

Voilà! We're now running a Jupyter Notebook on our GPU machine through the web browser on our local device!

6. **Open fine-tuning notebook**

We can use the file explorer on the left to go to our model repository and open the Jupyter notebook `fine_tune_whisper_streaming.ipynb`. 
In the top right of the notebook, you'll see a small window that says "Python 3". Clicking on this window will open a 
dropdown menu, from which we can select a Python kernel. Select your venv from this dropdown menu. This will ensure that 
you run the notebook in the venv we previously set up.

You can now run this notebook from start to finish and fine-tune the Whisper model as you desire 🤗 The notebook 
contains pointers for where you need to change variables for your language.

Since we're operating within a `tmux` session, we're free to close our SSH connection and browser window when we desire. 
Training won't be interrupted by closing this window. However, the notebook will cease to update, so you should make 
sure that training is working before closing the notebook. You can monitor training progress through your model repo 
on the Hugging Face Hub under the "Training Metrics" tab.

### Google Colab
The Google Colab for fine-tuning Whisper is entirely self-contained. No need to set up an environment or sping up a GPU.
You can access it through the following link:

<a target="_blank" href="https://colab.research.google.com/github/huggingface/community-events/blob/main/whisper-fine-tuning-event/fine_tune_whisper_streaming_colab.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Recommended Training Configurations

In this section, we provide guidance for appropriate training and evaluation batch sizes depending on your GPU device. 
Since the Whisper model expects log-Mel input features of a fixed dimension, the GPU memory required by the models is 
the same for audio samples of any length. Thus, these recommendations should stand for all 16/40GB GPU devices. However, 
if you experience out-of-memory errors, we recommend reducing the `per_device_train_batch_size` by factors of 2 and 
increasing the `gradient_accumulation_steps` to compensate.

If you want to explore methods for reducing the memory of the Whisper model, check out the section [Tips and Tricks](#tips-and-tricks).

#### V100 / 16 GB GPU

| Model  | Train Batch Size | Gradient Acc Steps | Eval Batch size |
|--------|------------------|--------------------|-----------------|
| small  | 16               | 2                  | 8               |
| medium | 2                | 16                 | 1               |

It is advised to run the "small" checkpoint if training on a V100 device. Running the medium checkpoint will take 
upwards of 12 hours for 5k training steps. We reckon you're better off training the "small" checkpoint for longer!

#### A100 / 40GB GPU

| Model  | Train Batch Size | Gradient Acc Steps | Eval Batch size |
|--------|------------------|--------------------|-----------------|
| small  | 64               | 1                  | 32              |
| medium | 32               | 1                  | 16              |

### Punctuation, Casing and Normalisation

When using the Python training script, removing casing for the training data is enabled by passing the flag `--do_lower_case`. 
Removing punctuation in the training data is achieved by passing the flag `--do_remove_punctuation`. Both of these flags 
default to False, and we **do not** recommend setting either of them to True. This will ensure your fine-tuned model 
learns to predict casing and punctuation. Normalisation is only applied during evaluation by setting the flag 
`--do_normalize_eval` (which defaults to True and recommend setting). Normalisation is performed according to the 
'official' Whisper normaliser. This normaliser applies the following basic standardisation for non-English text:
1. Remove any phrases between matching brackets ([, ]).
2. Remove any phrases between matching parentheses ((, )).
3. Replace any markers, symbols, and punctuation characters with a space, i.e. when the Unicode category of each character in the NFKC-normalized string starts with M, S, or P.
4. Make the text lowercase.
5. Replace any successive whitespace characters with a space.

Similarly, in the notebooks, removing casing in the training data is enabled by setting the variable `do_lower_case = True`, 
and punctuation by `do_remove_punctuation = True`. We do not recommend setting either of these to True to ensure that 
your model learns to predict casing and punctuation. Thus, they are set to False by default. Normalisation is only 
applied during evaluation by setting the variable `do_normalize_eval=True` (which we do recommend setting). 

## Evaluation

We'll be running a live leaderboard throughout the event to track the best performing models across all languages. The leaderboard will track your models performance across *all* the speech recognition models available on the hub for your chosen language and dataset.

You can find the leaderboard [here](https://huggingface.co/spaces/autoevaluate/leaderboards?dataset=common_voice_11_0&only_verified=0&task=automatic-speech-recognition&config=th&split=train%2Bvalidation&metric=wer) 📈.

Each participant should evaluate their fine-tuned Whisper checkpoint on the `"test"` split of the Common Voice 11 
dataset for their respective language. For languages that are not part of the Common Voice 11 dataset, please contact 
the organisers on Discord so that we can work together to find suitable evaluation data.

We recommend running evaluation during training by setting your eval dataset to the `"test"` split of Common Voice 11.
We'll also provide you with a standalone evaluation script so that you can test your model after training on Common Voice 
or other datasets of your choice.

In addition to running evaluation while training, you can noe use your Whisper checkpoints to run evaluation on *any* speech recognition dataset on the hub. The [run_eval_whisper_streaming.py](https://github.com/huggingface/community-events/blob/main/whisper-fine-tuning-event/run_eval_whisper_streaming.py) script loads your whisper checkpoints, runs batch inference on your specified dataset and returns the WER.

You can use the script as follows:
```bash
python run_eval_whisper_streaming.py --model_id="openai/whisper-tiny" --dataset="google/fleurs" --config="ar_eg" --device=0 --language="ar"
```

The evaluation script can be customised with the following parameters:
1. `model_id` - Whisper model identifier e.g. `openai/whisper-tiny`
2. `dataset` - Dataset name to evaluate the `model_id` on. Default value: `mozilla-foundation/common_voice_11_0`
3. `config` - Config of the dataset. e.g. `'en'` for the English split of Common Voice
4. `split` - Split of the dataset. Default value: `test`
5. `batch_size` - Number of samples to go through each streamed batch for inference. Default value: `16`
6. `max_eval_samples` - Max number of samples to be evaluated from the dataset. Put a lower number e.g. `64` for testing this script. **Only use this for testing the script**
7. `streaming` - Whether you'd like to download the entire dataset or stream it during the evaluation. Default value: `True`
8. `language` - Language you want the `model_id` to transcribe the audio in.
9. `device` - The device to run the pipeline on. e.g. `0` for running on GPU 0. Default value: -1 for CPU.

## Building a Demo

Finally, on to the fun part! Time to sit back and watch the model transcribe audio. We've created a [template Gradio demo](https://huggingface.co/spaces/whisper-event/whisper-demo) 
that you can use to showcase your fine-tuned Whisper model 📢

Click the link to duplicate the template demo to your account: https://huggingface.co/spaces/whisper-event/whisper-demo?duplicate=true

We recommend giving your space a similar name to your fine-tuned model (e.g. `whisper-demo-es`) and setting the visibility 
to "Public". 

Once you've duplicated the Space to your account, click "Files and versions" -> "app.py" -> "edit". Change the model 
identifier to your fine-tuned model (line 9). Scroll to the bottom of the page and click "Commit changes to `main`". 
The demo will reboot, this time using your fine-tuned model. You can share this demo with your friends and family so 
that they can use the model that you've trained!

*Checkout our video tutorial to get a better understanding 👉️ [YouTube Video](https://www.youtube.com/watch?v=VQYuvl6-9VE)*

## Communication and Problems

If you encounter any problems or have any questions, you should use one of the following platforms
depending on your type of problem. Hugging Face is an "open-source-first" organisation, meaning 
that we'll try to solve all problems in the most public and transparent way possible so that everybody
in the community benefits.

The following paragraph summarises the platform to use for each kind of problem:

- Problem/question/bug with the 🤗 Datasets library that you think is a general problem that also impacts other people, please open an [Issue on Datasets](https://github.com/huggingface/datasets/issues/new?assignees=&labels=bug&template=bug-report.md&title=) and ping @sanchit-gandhi and @vaibhavs10.
- Problem/question/bug with the 🤗 Transformers library that you think is a general problem that also impacts other people, please open an [Issue on Transformers](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title=) and ping @sanchit-gandhi and @vaibhavs10.
- Problem/question with a modified, customised training script that is less likely to impact other people, please post your problem/question [on the forum](https://discuss.huggingface.co/) and ping @sanchit-gandhi and @vaibhavs10.
- Problem/question regarding access or set up of a Lambda GPU, please ask in the Discord channel **#lambdalabs-infra-support**.
- Other questions regarding the event, rules of the event, or if you are unsure where to post your question, please ask in the Discord channel **#events**.

## Talks

We are very excited to be hosting talks from Open AI, Meta AI and Hugging Face to help you get a better understanding of the Whisper model, the VoxPopuli dataset and details about the fine-tuning event itself!

| **Speaker**                  | **Topic**                                                  | **Time**                      | **Video**                                                                                                                 |
|------------------------------|------------------------------------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| Sanchit Gandhi, Hugging Face | Introduction to Whisper Fine-Tuning Event                  | 15:00 UTC, 2nd December, 2022 | [![Youtube](https://www.youtube.com/s/desktop/f506bd45/img/favicon_32.png)](https://www.youtube.com/watch?v=1cVBLOMlv3w)  |
| Jong Wook Kim, OpenAI        | [Whisper Model](https://cdn.openai.com/papers/whisper.pdf) | 16:30 UTC, 5th December, 2022 | [![Youtube](https://www.youtube.com/s/desktop/f506bd45/img/favicon_32.png)](https://www.youtube.com/watch?v=fZMiD8sDzzg ) |
| Changhan Wang, MetaAI        | [VoxPopuli Dataset](https://arxiv.org/abs/2101.00390)      | 17:30 UTC, 5th December, 2022 | [![Youtube](https://www.youtube.com/s/desktop/f506bd45/img/favicon_32.png)](https://www.youtube.com/watch?v=fZMiD8sDzzg ) |

## Tips and Tricks

We include three memory saving tricks that you can explore to run the fine-tuning scripts with larger batch-sizes and 
potentially larger checkpoints.

### Adam 8bit
The [Adam optimiser](https://arxiv.org/abs/1412.6980a) requires two params (betas) for every model parameter. So the memory requirement of the optimiser is 
**two times** that of the model. You can switch to using an 8bit version of the Adam optimiser from [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes#bitsandbytes). 
This will cast the optimiser parameters into 8bit precision, saving you a lot of memory and potentially allowing you to run bigger batch sizes.
To use Adam 8bit, you first need to pip install `bitsandbytes`:

```bash
pip install bitsandbytes
```

Then, set `optim="adamw_bnb_8bit"`, either in your `run.sh` file if running from a Python script, or when you 
instantiate the Seq2SeqTrainingArguments from a Jupyter Notebook or Google Colab:

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./",
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    optim="adamw_bnb_8bit"
)
```

### Adafactor

Rather than using Adam, you can use a different optimiser all together. Adam requires two optimiser params per one model 
param, but [Adafactor](https://arxiv.org/abs/1804.04235) uses only one. To enable Adafactor, set `optim="adafactor"` in the 
`Seq2SeqTrainingArguments`. You can expect to double your training batch size when using Adafactor compared to Adam. 

A word of caution: Adafactor is untested for fine-tuning Whisper, so we are unsure sure how 
Adafactor performance compares to Adam! Typically, using Adafactor results in **slower convergence** than using Adam or 
Adam 8bit. For this reason, we recommend Adafactor as an **experimental feature** only.

### DeepSpeed

DeepSpeed is a framework for training larger deep learning models with limited GPU resources by optimising GPU utilisation. 
We provide implementation details for DeepSpeed ZeRo Stage 2, which partitions the optimiser states (ZeRO stage 1) and gradients 
(ZeRO stage 2). With DeepSpeed, it is more than possible to train the medium Whisper checkpoint on a V100, or the large 
checkpoint on an A100. For more details, we refer you to the blog post by the original authors: [DeepSpeed ZeRO](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/).

Using DeepSpeed with 🤗 Transformers is straightforward. First, we need to install the packages 🤗 Accelerate and DeepSpeed: 

```bash
pip install -U accelerate deepspeed
```

The DeepSpeed configuration file specifies precisely what form of optimiser/gradient offloading we are going to perform.
The key to getting a huge improvement on a single GPU with DeepSpeed is to have at least the provided DeepSpeed configuration 
in the configuration file [`ds_config.json`](https://github.com/huggingface/community-events/blob/main/whisper-fine-tuning-event/ds_config.json). 

You can copy the DeepSpeed configuration file to your model repository as follows:

```bash
cp ~/community-events/whisper-fine-tuning-event/ds_config.json .
```

### Python Script

Using DeepSpeed with the Python training script requires two changes to the `run.sh` file. Firstly, we launch the script using `deepspeed` 
instead of Python. Secondly, we pass the DeepSpeed config `ds_config.json` as a training argument. The remainder of the `run.sh` 
file takes the same format as using the native Trainer configuration:

```bash
deepspeed run_speech_recognition_seq2seq_streaming.py \
	--deepspeed="ds_config.json" \
	--model_name_or_path="openai/whisper-small" \
	--dataset_name="mozilla-foundation/common_voice_11_0" \
	--dataset_config_name="es" \
	--language="spanish" \
	--train_split_name="train+validation" \
	--eval_split_name="test" \
	--model_index_name="Whisper Small Spanish" \
	--max_steps="5000" \
	--output_dir="./" \
	--per_device_train_batch_size="64" \
	--per_device_eval_batch_size="32" \
	--logging_steps="25" \
	--learning_rate="1e-5" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--eval_steps="1000" \
	--save_strategy="steps" \
	--save_steps="1000" \
	--generation_max_length="225" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="sentence" \
	--freeze_feature_encoder="False" \
	--report_to="tensorboard" \
	--metric_for_best_model="wer" \
	--greater_is_better="False" \
	--load_best_model_at_end \
	--gradient_checkpointing \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--do_normalize_eval \
	--streaming \
	--use_auth_token \
	--push_to_hub
```

### Jupyter Notebook

Using DeepSpeed with the template Jupyter Notebooks requires two changes. Firstly, we add the following code cell at the 
start of the notebook to configure the DeepSpeed environment:

```python
# DeepSpeed requires a distributed environment even when only one process is used.
# This emulates a launcher in the notebook
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
```

Secondly, we pass the DeepSpeed config file to the training args:

```python
training_args = Seq2SeqTrainingArguments(..., deepspeed="ds_config.json")
```

### Recommended Batch Sizes with DeepSpeed

Using DeepSpeed, it is possible to fit larger batch sizes and even larger checkpoints on your device, be it a V100 or 
A100. We provide recommended batch sizes for the three checkpoint sizes of interest for 16GB GPUs and 40GB GPUs. As before,
these batch sizes are only indicative: you should tune the batch size depending on your device, checkpoint and language.

#### V100 / 16 GB GPU

| Model  | Train Batch Size | Gradient Acc Steps | Eval Batch size | Speed   |
|--------|------------------|--------------------|-----------------|---------|
| small  | 32               | 1                  | 16              | 1.3s/it |
| medium | 16               | 1 or 2             | 8               | 2.0s/it |
| large  | 8                | 2 or 4             | 4               | 3.8s/it |

#### A100 / 40GB GPU

| Model  | Train Batch Size | Gradient Acc Steps | Eval Batch size | Speed   |
|--------|------------------|--------------------|-----------------|---------|
| small  | 64               | 1                  | 32              | 2.3s/it |
| medium | 64               | 1                  | 32              | 5.8s/it |
| large  | 32               | 1 or 2             | 16              | 5.9s/it |


## Scripts & Colabs

1. [Whirlwind tour of Whispering with 🤗Transformers](https://colab.research.google.com/drive/1l290cRv4RdvuLNlSeo9WexByHaNWs3s3?usp=sharing)
2. [8bit inference for Whisper large model (6.5 gig VRAM) 🤯](https://colab.research.google.com/drive/1EMOwwfm1V1fHxH7eT1LLg7yBjhTooB6j?usp=sharing)

<!--- TODO: VB - Move these colabs to a GitHub repo --->

## Feedback

We would love to get your feedback on the event! If you have a spare ten minutes, we'd appreciate you filling out the 
feedback form at: https://forms.gle/7hvrTE8NaSdQwwU68
