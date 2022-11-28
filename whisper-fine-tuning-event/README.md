# Whisper Fine-Tuning Event 🤗

Welcome to the Whisper fine-tuning event 🎙️!

For two weeks, we will endeavour to fine-tune the Whisper model to build state-of-the-art speech recognition systems in 
the languages of our choice 🗣. We will work together as a community to achieve this, helping others and learning where 
we can 🤗. If necessary and available, free access to A100 40 GB GPUs will kindly be provided by our cloud compute 
partners, [Lambda Labs](https://lambdalabs.com) 🚀.

This document summarises all the relevant information required for the event 📋. Please read it thoroughly 
and make sure to:
- Sign-up using the [Google form](https://forms.gle/F2bpouvhDpKKisM39)
- Join the [Hugging Face Discord server](https://hf.co/join/discord) and make sure you have access to the #events channel. TODO: VB - add specific instructions for going to the role-assignments channel and accept audio

## Table of Contents

- [Introduction](#introduction)
- [Important Dates](#important-dates)
- [Launch a Lambda Cloud GPU](#launch-a-lambda-cloud-gpu)
- [Set Up an Environment](#set-up-an-environment)
- [Data and Pre-Processing](#data-and-pre-processing)
- [Fine-Tune a Whisper Model](#fine-tune-whisper)
- [Evaluation](#evaluation)
- [Communication and Problems](#communication-and-problems)
- [Talks](#talks)
- [Tips and Tricks](#tips-and-tricks)

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
Hugging Face Hub, web-scraped data from the internet, or private datasets. Speech recognition systems will be evaluated 
on the "test" split of the [Common Voice 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) 
dataset for the participant's chosen language.

We believe that framing the event as a competition is fun! But at the core, the event is about
fine-tuning Whisper in as many languages as possible as a community. We want to foster an environment where we 
work together, help each other solve bugs, share important findings and ultimately learn something new.

This README contains all the information you need for the event. It is structured such that you can read it sequentially, 
section-by-section. **We recommend that you read the document once from start to finish before running any code.** This will 
give you an idea of where to look for the relevant information and an idea of how the event is going to run.

## Important Dates

- *Introduction Talk*: 1st December 2022
- *Sprint start*: 5th December 2022
- *Speaker Events* 5th December 2022
- *Sprint end*: 19th December 2022
- *Results*: 23rd December 2022

## Launch a Lambda Cloud GPU
Where possible, we encourage you to fine-tune Whisper on a local GPU machine. This will mean a faster set-up and more 
familiarity with your device. If you are running on a local GPU machine, you can skip ahead to the next Section: [Set Up an Environment](#set-up-an-environment). 
However, if you do not have access to a GPU, we'll endeavour to provide you with a cloud GPU instance.

We've partnered up with Lambda Labs to provide cloud compute for this event. They'll be providing the latest NVIDIA A100 
40 GB GPUs, so you'll be loaded with some serious firepower! The Lambda Labs Cloud API makes it easy to spin-up and launch 
a GPU instance. In this Section, we'll go through the steps for spinning up an instance one-by-one.

This Section is split into two halves:

1. [Signing-Up with Lambda Labs](#signing-up-with-lambda-labs)
2. [Creating a Cloud Instance](#creating-a-cloud-instance)

### Signing-Up with Lambda Labs
TODO: SG - add Section once we've figured out how the 'teams' function is going to work with Mitesh

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



We can now verify that `transformers` and `datasets` have been correctly installed. First, launch a Python shell:

```bash
python
```

Running the following code cell will load a "dummy" dataset from the Hub and perform a forward pass of the 
"tiny" Whisper model:

```python
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from evaluate import load

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

inputs = processor(ds[0]["audio"]["array"], sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features

with torch.no_grad():
    tokens = model.generate(input_features, max_length=40)
predictions = processor.batch_decode(tokens, skip_special_tokens=True)

wer_metric = load("wer")

wer = wer_metric.compute(references=ds[0]["text"], predictions=predictions)

assert round(wer) == 
```

If the final `assert` statement passes, the libraries have been installed correctly. Finally, exit the Python shell:
```python
quit()
```

TODO: SG - do we need to install:
* tensorboard

## Data and Pre-Processing

In this Section, we will quickly go over how to find suitable training data and 
how to preprocess it.

To begin with, **all data except Common Voice's `"test"` data can be used as training data.**
The exception includes all Common Voice versions as the test data split of later Common Voice versions often
overlaps with the one of previous versions, *e.g.* the test data of Common Voice 7 in English is 
to a big part identical to the test data of Common Voice 6 in English:

```python
load_dataset("mozilla-foundation/common_voice_11_0", "en", split="test") 
```

includes more or less the same data as

```python
load_dataset("mozilla-foundation/common_voice_10_0", "en", split="test") 
```

However, we strongly encourage participants to make use of Common Voice's other splits, *e.g.* `"train"` and `"validation"`.
For most languages, the Common Voice dataset offers already a decent amount of training data. It is usually 
always advantageous to collect additional data. To do so, the participants are in first step encouraged to search the
Hugging Face Hub for additional audio data, for example by selecting the category 
["speech-processing"](https://huggingface.co/datasets?task_categories=task_categories:speech-processing&sort=downloads).
All datasets that are available on the Hub can be downloaded via the 🤗 Datasets library in the same way Common Voice is downloaded.
If one wants to combine multiple datasets for training, it might make sense to take a look at 
the [`interleave_datasets`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=interleave#datasets.interleave_datasets) function.

In addition, participants can also make use of their audio data. Here, please make sure that you **are allowed to use the audio data**. E.g., if audio data 
is taken from media platforms, such as YouTube, it should be verified that the media platform and the owner of the data have given their approval to use the audio 
data in the context of machine learning research. If you are not sure whether the data you want to use has the appropriate licensing, please contact the Hugging Face 
team on discord.

Next, let's talk about preprocessing. Audio data and transcriptions have to be brought into the correct format when 
training the acoustic model (example shown in [How to fine-tune a Whisper model](#how-to-finetune-a-whisper-model)).
It is recommended that this is done by using 🤗 Datasets `.map()` function as shown below.

```python
def remove_special_characters(batch):
    if chars_to_ignore_regex is not None:
        batch["target_text"] = re.sub(chars_to_ignore_regex, "", batch[text_column_name]).lower() + " "
    else:
        batch["target_text"] = batch[text_column_name].lower() + " "
    return batch


raw_datasets = raw_datasets.map(
    remove_special_characters,
    remove_columns=[text_column_name],
    desc="remove special characters from datasets",
    )
```

The participants are free to modify this preprocessing by removing more characters or even replacing characters as 
it is done in the [official script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_seq2seq.py).
**However**, there are some rules regarding what characters are allowed to be removed/replaced and which are not.
These rules are not this straightforward and therefore often have to be evaluated case-by-case.
It is allowed (and recommended) to normalize the data to only have lower-case characters. It is also allowed (and recommended) to remove typographical 
symbols and punctuation marks. A list of such symbols can *e.g.* be found [here](https://en.wikipedia.org/wiki/List_of_typographical_symbols_and_punctuation_marks) - however here we already must be careful. We should **not** remove a symbol that would change the meaning of the words, *e.g.* in English, 
we should not remove the single quotation mark `'` since it would change the meaning of the word `"it's"` to `"its"` which would then be incorrect. 
So the golden rule here is to not remove any characters that could change the meaning of a word into another word. This is not always obvious and should 
be given some consideration. As another example, it is fine to remove the "Hyphen-minus" sign "`-`" since it doesn't change the 
meaning of a word to another one. *E.g.* "`fine-tuning`" would be changed to "`finetuning`" which has still the same meaning.

Since those choices are not always obvious when in doubt feel free to ask on Discord or even better post your question on the forum.

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

<!-- TODO: VB - Add a fine-tuning guide here after testing the script on lambda labs GPU -->

## Evaluation

<!-- TODO: VB - To add after we have decided on the final evaluation criteria -->

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
