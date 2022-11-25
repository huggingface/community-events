# Whisper Fine-Tuning Event 🤗

Welcome to the Whisper fine-tuning event 🎙️!

The goal of this event is to fine-tune Whisper and build state-of-the-art speech recognition systems in as many languages as possible 🌏🌍🌎. 
We will work together as a community to achieve this, helping others and learning where we can 🤗. If necessary and 
available, free access to A100 40 GB GPUs will kindly be provided by our cloud compute partners, [Lambda Labs](https://lambdalabs.com) 🚀.

This document summarises all the relevant information required for the event 📋. Please read it thoroughly 
and make sure to:
- Sign-up using the [Google form](https://forms.gle/F2bpouvhDpKKisM39)
- Join the [Hugging Face Discord server](https://hf.co/join/discord) and make sure you have access to the #events channel. TODO: VB - add specific instructions for going to the role-assignments channel and accept audio

## Table of Contents

- [TLDR](#tldr)
- [Important Dates](#important-dates)
- [Introduction](#introduction)
- [Launch a Lambda Cloud GPU](#launch-a-lambda-cloud-gpu)
- [Set Up an Environment](#set-up-an-environment)
- [Data and Pre-Processing](#data-and-pre-processing)
- [Fine-Tune a Whisper Model](#fine-tune-whisper)
- [Evaluation](#evaluation)
- [Prizes](#prizes)
- [Communication and Problems](#communication-and-problems)
- [Talks](#talks)
- [Tips and Tricks](#tips-and-tricks)

## TLDR

Whisper achieves strong performance on many datasets, domains and languages. Through fine-tuning, the 
performance of this model can be boosted further for specific languages.

In this event, we're bringing the community together to fine-tune Whisper in as many languages as possible. Our aim is 
to achieve state-of-the-art on the languages spoken by the community. Together, we can democratise speech recognition for all.

We are providing scripts, notebooks, blog posts, talks and compute (where available), so you have all the resources you 
need to participate.

During the event, the speech recognition system will be evaluated on both the Common Voice `"test"` split 
of the participants' chosen language as well as the *real-world* `"dev"` data provided by 
the Hugging Face team. 
At the end of the whisper fine-tuning sprint, the speech recognition system will also be evaluated on the
*real-world* `"test"` data provided by the Hugging Face team. Each participant should add an 
`eval.py` script to her/his model repository in a specific format that lets one easily 
evaluate the speech recognition system on both Common Voice's `"test"` data as well as the *real-world* audio 
data. Please read through the [Evaluation](#evaluation) section to make sure your evaluation script is in the correct format. Models
with evaluation scripts in an incorrect format can sadly not be considered for the Challenge.

At the end of the event, the best-performing speech recognition system 
will receive a prize 🏆 - more information regarding the prizes can be found under [Prizes](#prizes).

We believe that framing the event as a competition is more fun, but at the core, the event is about
fine-tuning Whisper in as many languages as possible as a community.
This can be achieved by working together, helping each other to solve bugs, sharing important findings, etc...🤗

**Note**:
Please, read through the section on [Communication & Problems](#communication-and-problems) to make sure you 
know how to ask for help, etc...
All important announcements will be made on discord. Please make sure that 
you've joined [#events channel](https://hf.co/join/discord)

## Important Dates
<!--- TODO: SG - this section can probably be collapsed into a subsection under TLDR or intro, otherwise think about where it fits as a section --->

- *Talks*: 1st & 2nd December 2022
- *Sprint start*: 5th December 2022
- *Sprint end*: 19th December 2022
- *Whisper benchmark & results*: 26th December 2022 

## Introduction

## Data and Pre-Processing

In this section, we will quickly go over how to find suitable training data and 
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

## Launch a Lambda Cloud GPU
Placeholder section for Cloud GPU

## Set Up an Environment

Before installing the required libraries, we'd need to install and update `ffmpeg` to version 4:

```bash
add-apt-repository -y ppa:jonathonf/ffmpeg-4
apt update
apt install -y ffmpeg
```

Now, on to installing the relevant libraries for our fine-tuning runs. The following libraries are required to fine-tune Whisper with 🤗 Transformers and 🤗 Datasets in PyTorch.

- [PyTorch](https://pytorch.org/)
- [Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)

We recommend installing the above libraries in a [virtual environment](https://docs.python.org/3/library/venv.html). 
If you're unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). Create a virtual environment with the version of Python you're going
to use and activate it.

You should be able to run the command:

```bash
python3 -m venv <your-venv-name>
```

You can activate your venv by running

```bash
source ~/<your-venv-name>/bin/activate
```

To begin with please make sure you have PyTorch and CUDA correctly installed. 
The following command should return ``True``:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If the above command doesn't print ``True``, in the first step, please follow the
instructions [here](https://pytorch.org/) to install PyTorch with CUDA.

We strongly recommend making use of the provided PyTorch Seq2Seq Speech Recognition script in [transformers/examples/pytorch/speech-recognition](https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_seq2seq.py) to fine-tune your Whisper model.

Alright, onto the home stretch, let's install all the required packages into our virtual environment.

```bash
pip install datasets>=2.6.1
pip install git+https://github.com/huggingface/transformers
pip install librosa
pip install evaluate>=0.30
pip install jiwer
```
<!-- TODO: VB - these are based on a colab env install, double check this if it works on a fresh VM too.-->

To verify that all libraries are correctly installed, you can run the following command in a Python shell.
It verifies that both `transformers` and `datasets` have been correctly installed.

```python
import torch
from transformers import WhisperFeatureExtractor, WhisperModel
from datasets import load_dataset

model = WhisperModel.from_pretrained("openai/whisper-base")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
input_features = inputs.input_features

decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state

assert last_hidden_state.shape[-1] == 512
```

Note: If you plan on contributing a specific dataset during 
the community week, please fork the datasets repository and follow the instructions 
[here](https://github.com/huggingface/datasets/blob/master/CONTRIBUTING.md#how-to-create-a-pull-request).

## Fine-Tune Whisper

<!-- TODO: VB - Add a fine-tuning guide here after testing the script on lambda labs GPU -->

## Evaluation

<!-- TODO: VB - To add after we have decided on the final evaluation criteria -->

## Prizes

<!-- TODO: Sanchit/ Omar/ VB - Put prizes here when decided. -->

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
