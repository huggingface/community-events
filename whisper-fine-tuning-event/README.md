# Whisper Fine-Tuning Event 🤗

Welcome to the Whisper fine-tuning event 🎙️!

For two weeks, we will endeavour to fine-tune the Whisper model to build state-of-the-art speech recognition systems in 
the languages of our choice 🗣. We will work together as a community to achieve this, helping others and learning where 
we can 🤗. If necessary and available, free access to A100 40 GB GPUs will kindly be provided by our cloud compute 
partners, [Lambda Labs](https://lambdalabs.com) 🚀.

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
- *Speaker Events* 5th December 2022
- *Sprint end*: 19th December 2022
- *Results*: 23rd December 2022

## Launch a Lambda Cloud GPU
Where possible, we encourage you to fine-tune Whisper on a local GPU machine. This will mean a faster set-up and more 
familiarity with your device. If you are running on a local GPU machine, you can skip ahead to the next Section: [Set Up an Environment](#set-up-an-environment). 

The training scripts can also be run as a notebook through Google Colab. We recommend you train on Google Colab if you 
have a "Colab Pro" or "Pro+" subscription. This is to ensure that you receive a sufficiently powerful GPU on your Colab for 
fine-tuning Whisper. If you wish to fine-tune Whisper through Google Colab, you can skip ahead to the Section: [Data and Pre-Processing](#data-and-pre-processing). 

If you do not have access to a local GPU or Colab Pro/Pro+, we'll endeavour to provide you with a cloud GPU instance.
We've partnered up with Lambda Labs to provide cloud compute for this event. They'll be providing the latest NVIDIA A100 
40 GB GPUs, so you'll be loaded with some serious firepower! The Lambda Labs Cloud API makes it easy to spin-up and launch 
a GPU instance. In this Section, we'll go through the steps for spinning up an instance one-by-one.

This Section is split into two halves:

1. [Signing-Up with Lambda Labs](#signing-up-with-lambda-labs)
2. [Creating a Cloud Instance](#creating-a-cloud-instance)

### Signing-Up with Lambda Labs
TODO: SG - add Section once we've figured out how the 'teams' function is going to work with Lambda

### Creating a Cloud Instance
Estimated time to complete: 5 mins

1. Click the link: https://cloud.lambdalabs.com/instance
2. You'll be asked to sign in to your Lambda Labs account (if you haven't done so already).
3. Once on the GPU instance page, click the purple button "Launch instance" in the top right.
4. Launching an instance:
   1. In "Instance type", select the instance type "1x A100 (40 GB SXM4)" TODO: SG - SXM4 or PCle?
   2. In "Select region", select the region with availability closest to you.
   3. In "Select filesystem", select "Don't attach a filesystem".
5. You will be asked to provide your public SSH key. This will allow you to SSH into the GPU device from your local machine.
   1. If you’ve not already created an SSH key pair, you can do so with the following command from your local device: 
      ```bash
      ssh-keygen
      ```
   2. You can find your public SSH key using the command: 
      ```bash
      cat ~/.ssh/id_rsa.pub
      ```
   4. Copy and paste the output of this command into the first text box
   5. Give your SSH key a memorable name (e.g. `sanchits-mbp`)
   6. Click "Add SSH Key"
6. Select the SSH key from the drop-down menu and click "Launch instance"
7. Read the terms of use and agree
8. We can now see on the "GPU instances" page that our device is booting up!
9. Once the device status changes to "✅ Running", click on the SSH login ("ssh ubuntu@..."). This will copy the SSH login to your clipboard.
10. Now open a new command line window, paste the SSH login, and hit Enter.
11. If asked "Are you sure you want to continue connecting?", type "yes" and press Enter.
12. Great! You're now SSH'd into your A100 device! We can now move on to setting up an environment.

TODO: SG - video for launching an instance

## Set Up an Environment
Estimated time to complete: 5 mins

The Whisper model should be fine-tuned using **PyTorch**, **🤗 Transformers**, and, **🤗 Datasets**. In this 
Section, we'll cover how to set up an environment with the required libraries.

First, we need to make sure we have the required NVIDIA drivers installed. We can check that we have these drivers 
through the following command:

```bash
nvidia-smi
```

This should print a table with our NVIDIA driver version and CUDA version, and should work out of the box for Lambda Labs GPUs!
If you get an error running this command, refer to your device manual for installing the required NVIDIA driver.

Before installing the required libraries, we'd need to install and update `ffmpeg` to version 4:

 ```bash
add-apt-repository -y ppa:jonathonf/ffmpeg-4
apt update
apt install -y ffmpeg
 ```

We recommend installing the required libraries in a Python virtual environment. If you're unfamiliar with Python virtual 
environments, check out the official user guide: [installing-using-pip-and-virtual-environments](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

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

First, let's copy the `requirements.txt` file to our GPU device:

```bash
wget https://github.com/huggingface/community-events/blob/main/whisper-fine-tuning-event/requirements.txt
```

Now we can install the packages in this file using the following command:

```bash
pip install -r requirements.txt
```

We can check that above steps installed the correct version of PyTorch to match our CUDA version. The following command should return True:

```python
python -c "import torch; print(torch.cuda.is_available())"
```

We can now verify that `transformers` and `datasets` have been correctly installed. First, launch a Python shell:

```bash
python
```

Running the following code cell will load one sample of the [common voice](https://huggingface.co/datasets/common_voice) 
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
huggingface-cli login
```

And then enter an authentication token from https://huggingface.co/settings/tokens.

<!--- TODO: SG - do we need to set git.config? --->

<!--- TODO: SG - do we need to install tensorboard? Add to requirements.txt if so --->

## Data and Pre-Processing

In this Section, we will cover how to find suitable training data and the necessary steps to pre-process it. 
If you are new to the 🤗 Datasets library, we recommend reading the comprehensive blog post: [A Complete Guide To Audio Datasets](https://huggingface.co/blog/audio-datasets). 
This will tell you everything you need to know about 🤗 Datasets and its one-line API.

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
Common Voice splits as training data, such as `"train"` and `"validation"`:

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
splits in this same way using the fine-tuning scripts in the following Section [Fine-Tune Whisper](#fine-tune-whisper).

In addition to the Common Voice corpus, incorporating supplementary training data is usually beneficial. The Whisper 
project demonstrates the significant effect that increasing the amount of training data can have on downstream 
performance. There are a number of datasets that are available on the Hugging Face Hub that can be downloaded via 
the 🤗 Datasets library in much the same way as Common Voice 11.

We recommend the following four datasets on the Hugging Face Hub for multilingual speech recognition:

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
You can also explore all speech recognition datasets on the Hub to find one suited for your language and needs: https://huggingface.co/datasets?task_categories=task_categories:automatic-speech-recognition&sort=downloads.

If one wants to combine multiple datasets for training, it might make sense to take a look at 
the [`interleave_datasets`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=interleave#datasets.interleave_datasets) function.

<!--- TODO: SG - example script for doing this --->

In addition to publicly available data on the Hugging Face Hub, participants can also make use of their own audio data 
for training. When using your own audio data, please make sure that you **are allowed to use the audio data**. For 
instance, if the audio data is taken from media platforms, such as YouTube, please verify that the media platform and 
the owner of the data have given their approval to use the audio data in the context of machine learning research. If 
you are not sure whether the data you want to use has the appropriate licensing, please contact the Hugging Face team 
on Discord.

<!--- TODO: VB - tutorial for adding own data via audio folder --->

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
download small chunks of the dataset at a time, in a process called _streaming_. Since the data is downloaded 
progressively as we iterate over the dataset, we can get started with a dataset without waiting for the entire dataset 
to download. Once we're done with a chunk, it's automatically deleted. This way, we only have the data when we need it, 
and not when we don't!

Streaming is enabled by passing the argument `streaming=True` to the `load_dataset` function. We can then use our 
audio datasets in much the same way as before! For these reasons, **we highly recommend** that you try out the following 
code cell! Just make sure you've accepted the Common Voice 11 terms of use on the Hugging Face Hub: https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0.

```python
from datasets import load_dataset

common_voice = load_dataset("mozilla-foundation/common_voice_11_0", "en", use_auth_token=True, streaming=True)

# get the first sample of the dataset straight away!
print(next(iter(common_voice["train"])))
```

The examples in this README heavily rely on streaming mode to fine-tune Whisper. With streaming mode, we can use **any 
speech recognition dataset with just 20GB of disk space**. If you want to read more about streaming mode, we 
recommend you check out the aforementioned blog post: [A Complete Guide To Audio Datasets](https://huggingface.co/blog/audio-datasets). 

### Pre-Processing

Data pre-processing is a very grey area when it comes to speech recognition. In this Section, we'll try to make the 
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
characters, or try and predict punctuation from the audio data alone. Because of this, the word error rates are 
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
following Section [Fine-Tune Whisper](#fine-tune-whisper).

When mixing datasets, you should ensure the transcription format is consistent across datasets. For example, if you mix 
Common Voice 11 (cased + punctuated) with VoxPopuli (un-cased + punctuated), you will need to lower-case **all the text** 
for both training and evaluation, such that the transcriptions are consistent across training samples (un-cased + punctuated). 

Likewise, if mixing Common Voice 11 (cased + punctuated) with Multilingual LibriSpeech (un-cased + un-punctuated), you 
should make sure to remove all casing and punctuation in **all the text** for both training and evaluation, such that 
all transcriptions are un-cased and un-punctuated for all training samples.

Having a mismatch in formatting for different training samples can reduce the final performance of your fine-tuned Whisper 
model.

| Dataset                                                                                       | Casing | Punctuation |
|-----------------------------------------------------------------------------------------------|--------|-------------|
| [Common Voice 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)       | ✅      | ✅           |
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli)                               | ❌      | ✅           |
| [Multilingual LibriSpeech](https://huggingface.co/datasets/facebook/multilingual_librispeech) | ❌      | ❌           |
| [FLEURS](https://huggingface.co/datasets/google/fleurs)                                       | ✅      | ✅           |

If you want to find out more about pre- and post-processing for speech recognition, we point you in the direction of 
the paper: [ESB: A Benchmark For Multi-Domain End-to-End Speech Recognition](https://arxiv.org/abs/2210.13352).

<!--- TODO: SG What pre-processing steps do we deem appropriate? --->

## Fine-Tune Whisper

Throughout the event, participants are encouraged to leverage the official pre-trained [Whisper checkpoints](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=downloads&search=whisper).
The Whisper checkpoints come in five configurations of varying model sizes.
The smallest four are trained on either English-only or multilingual data.
The largest checkpoint is multilingual only. The checkpoints are summarised in the following table with links to the 
models on the Hugging Face Hub:

| Size   | Layers | Width | Heads | Parameters | English-only                                         | Multilingual                                      |
|--------|--------|-------|-------|------------|------------------------------------------------------|---------------------------------------------------|
| tiny   | 4      | 384   | 6     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny.)  |
| base   | 6      | 512   | 8     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)   |
| small  | 12     | 768   | 12    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)  |
| medium | 24     | 1024  | 16    | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium) |
| large  | 32     | 1280  | 20    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)  |

We recommend using the tiny model for rapid prototyping. We advise that the small or medium checkpoints are used for 
fine-tuning. These checkpoints achieve comparable performance to the large checkpoint with very little fine-tuning, but 
can be trained much faster (and hence for much longer!).
<!--- TODO: SG - review this after lambda testing --->

When using the training scripts, removing casing is enabled by passing the flag `--do_lower_case`. Remove 
punctuation is achieved by passing the flag `--do_remove_punctuation`. The punctuation characters removed are defined 
in TODO. Normalisation is only applied during evaluation by setting the flag `--do_normalize_eval_only`.

Similarly, in the notebooks, removing casing is enabled by setting the variable `do_lower_case = True` and punctuation 
by `do_remove_punctuation = True`. Normalisation is only applied during evaluation by setting the variable 
`do_normalize_eval_only=True`.

## Evaluation

<!--- TODO: Live leaderboard at XYZ --->

<!--- TODO: VB - To add after we have decided on the final evaluation criteria --->

## Building a Demo

## Communication and Problems

If you encounter any problems or have any questions, you should use one of the following platforms
depending on your type of problem. Hugging Face is an "open-source-first" organization meaning 
that we'll try to solve all problems in the most public and transparent way possible so that everybody
in the community profits.

The following table summarizes what platform to use for which problem.

- Problem/question/bug with the 🤗 Datasets library that you think is a general problem that also impacts other people, please open an [Issues on Datasets](https://github.com/huggingface/datasets/issues/new?assignees=&labels=bug&template=bug-report.md&title=) and ping @sanchit-gandhi and @vaibhavs10.
- Problem/question/bug with the 🤗 Transformers library that you think is a general problem that also impacts other people, please open an [Issues on Transformers](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title=) and ping @sanchit-gandhi and @vaibhavs10.
- Problem/question with a modified, customized training script that is less likely to impact other people, please post your problem/question [on the forum](https://discuss.huggingface.co/) and ping @sanchit-gandhi and @vaibhavs10.
- Other questions regarding the event, rules of the event, or if you are not sure where to post your question, please ask in the Discord channel **#events**.

<!-- TODO: VB - Add a note about cloud issues when we have the cloud provider identified -->

## Talks

<!-- TODO: VB - Add Talk schedule when up. -->

## Tips and Tricks

<!-- TODO: VB - Add tips for faster convergence/ memory efficient training. -->

## Feedback

We would love to get your feedback on the event! If you have a spare two minutes, we'd appreciate you filling out the 
feedback form at: TODO

<!--- TODO: VB - topics for feedback form (Google form):
* Discord comms
* README info
* Streaming mode
* Examples scripts
* Satisfaction with the event
--->
