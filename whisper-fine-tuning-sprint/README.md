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

- [Introduction](#introduction)
- [Important Dates](#important-dates)
- [Launch a Lambda Cloud GPU](#launch-a-lambda-cloud-gpu)
- [Set Up an Environment](#set-up-an-environment)
- [Data and Pre-Processing](#data-and-pre-processing)
- [Fine-Tune a Whisper Model](#fine-tune-whisper)
- [Evaluation](#evaluation)
- [Prizes](#prizes)
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

Speech recognition systems will be evaluated on the "test" split of the [Common Voice 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) 
dataset for the participant's chosen language. At the end of the event, we'll verify the results, and the 
best-performing speech recognition system in each language will receive a prize 🏆.

We believe that framing the event as a competition is fun! But at the core, the event is about
fine-tuning Whisper in as many languages as possible as a community. We want to foster an environment where we 
work together, help each other solve bugs, share important findings and ultimately learn something new.

This README contains all the information you need for the event. It is structured such that you can read it sequentially, 
section-by-section. We recommend that you read the document once from start to finish before running any code. This will 
give you an idea of where to look for the relevant information and an idea of how the event is going to run.

## Important Dates

- *Sprint start*: 5th December 2022
- *Sprint end*: 19th December 2022
- *Whisper benchmark & results*: 26th December 2022 (tentative) TODO: VB, SG - decide a timeline for evaluation

## Launch a Lambda Cloud GPU
Where possible, we encourage you to fine-tune Whisper on a local GPU machine. If you are running on a local GPU machine, 
you can skip ahead to the next section: [Set Up an Environment](#set-up-an-environment). However, if you do not have 
access to one, we'll endeavour to provide you with a cloud GPU instance.

We've partnered up with Lambda Labs to provide cloud compute for this event. They'll be providing the latest NVIDIA A100 
40 GB GPUs, so you'll be loaded with some serious firepower! This section is split into two halves:

1. [Signing-Up with Lambda Labs](#signing-up-with-lambda-labs)
2. [Creating a Cloud Instance](#creating-a-cloud-instance)

### Signing-Up with Lambda Labs
TODO: SG - add section once we've figured out how the 'teams' function is going to work with Mitesh

### Creating a Cloud Instance
TODO: SG

## Set Up an Environment

Speech recognition systems should be trained using **PyTorch**, **🤗 Transformers**, and, **🤗 Datasets**. In this 
section, we'll cover how to set up an environment with the required libraries.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If the above command doesn't print ``True``, in the first step, please follow the
instructions [here](https://pytorch.org/) to install PyTorch with CUDA.

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
