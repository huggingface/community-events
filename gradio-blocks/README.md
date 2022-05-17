# Welcome to the [Gradio](https://gradio.app/) Blocks Party 🥳

![image (1)](https://user-images.githubusercontent.com/81195143/167954125-9854bf6b-4ae5-4735-8fdd-830fec41efa1.png)


_**Timeline**: May 17th, 2022 - May 31st, 2022_

---

We are happy to invite you to the Gradio Blocks Party - a community event in which we will create **interactive demos** for state-of-the-art machine learning models. Demos are powerful because they allow anyone — not just ML engineers — to try out models in the browser, give feedback on predictions, identify trustworthy models. The event will take place from **May 17th to 31st**. We will be organizing this event on [Github](https://github.com/huggingface/community-events) and the [Hugging Face discord channel](https://discord.com/invite/feTf9x3ZSB). Prizes will be given at the end of the event, see: [Prizes](#prizes)

<img src="https://user-images.githubusercontent.com/81195143/168656398-ace7acc9-ef7a-4e90-a9cd-c7d15dd800e1.gif" width="1160" height="600"/>

## What is Gradio?

Gradio is a Python library that allows you to quickly build web-based machine learning demos, data science dashboards, or other kinds of web apps, entirely in Python. These web apps can be launched from wherever you use Python (jupyter notebooks, colab notebooks, Python terminal, etc.) and shared with anyone instantly using Gradio's auto-generated share links. To learn more about Gradio see the Getting Started Guide: https://gradio.app/getting_started/ and the new Course on Huggingface about Gradio: [Gradio Course](https://huggingface.co/course/chapter9/1?fw=pt).

Gradio can be installed via pip and comes preinstalled in Hugging Face Spaces, the latest version of Gradio can be set in the README in spaces by setting the sdk_version for example `sdk_version: 3.0b8`

`pip install gradio` to install gradio locally


## What is Blocks?

`gradio.Blocks` is a low-level API that allows you to have full control over the data flows and layout of your application. You can build very complex, multi-step applications using Blocks. If you have already used `gradio.Interface`, you know that you can easily create fully-fledged machine learning demos with just a few lines of code. The Interface API is very convenient but in some cases may not be sufficiently flexible for your needs. For example, you might want to:

* Group together related demos as multiple tabs in one web app.
* Change the layout of your demo instead of just having all of the inputs on the left and outputs on the right.
* Have multi-step interfaces, in which the output of one model becomes the input to the next model, or have more flexible data flows in general.
* Change a component's properties (for example, the choices in a Dropdown) or its visibility based on user input.

To learn more about Blocks, see the [official guide](https://www.gradio.app/introduction_to_blocks/) and the [docs](https://gradio.app/docs/).

## What is Hugging Face Spaces?

Spaces are a simple way to host ML demo apps directly on your profile or your organization’s profile on Hugging Face. This allows you to create your ML portfolio, showcase your projects at conferences or to stakeholders, and work collaboratively with other people in the ML ecosystem. Learn more about Spaces in the [docs](https://huggingface.co/docs/hub/spaces).

## How Do Gradio and Hugging Face work together?

Hugging Face Spaces is a free hosting option for Gradio demos. Spaces comes with 3 SDK options: Gradio, Streamlit and Static HTML demos. Spaces can be public or private and the workflow is similar to github repos. There are over 2000+ Gradio spaces currently on Hugging Face. Learn more about spaces and gradio: https://huggingface.co/docs/hub/spaces

## Event Plan

main components of the event consist of:

1. Learning about Gradio and the new Blocks Feature
2. Building your own Blocks demo using Gradio and Hugging Face Spaces
3. Submitting your demo on Spaces to the Gradio Blocks Party Organization
4. Share your blocks demo with a permanent shareable link 
5. Win Prizes


## Example spaces using Blocks

<img width="1180" alt="mindseye-lite" src="https://user-images.githubusercontent.com/81195143/168619604-cf1ac733-c10e-487f-add4-8da48002dcff.png">

- [dalle-mini](https://huggingface.co/spaces/dalle-mini/dalle-mini)([Code](https://huggingface.co/spaces/dalle-mini/dalle-mini/blob/main/app/gradio/app.py))
- [mindseye-lite](https://huggingface.co/spaces/multimodalart/mindseye-lite)([Code](https://huggingface.co/spaces/multimodalart/mindseye-lite/blob/main/app.py))
- [ArcaneGAN-blocks](https://huggingface.co/spaces/akhaliq/ArcaneGAN-blocks)([Code](https://huggingface.co/spaces/akhaliq/ArcaneGAN-blocks/blob/main/app.py))
- [gr-blocks](https://huggingface.co/spaces/merve/gr-blocks)([Code](https://huggingface.co/spaces/merve/gr-blocks/blob/main/app.py))
- [tortoisse-tts](https://huggingface.co/spaces/osanseviero/tortoisse-tts)([Code](https://huggingface.co/spaces/osanseviero/tortoisse-tts/blob/main/app.py))
- [CaptchaCracker](https://huggingface.co/spaces/osanseviero/tortoisse-tts)([Code](https://huggingface.co/spaces/akhaliq/CaptchaCracker/blob/main/app.py))


## To participate in the event

- Join the organization for Blocks event
    - [https://huggingface.co/Gradio-Blocks](https://huggingface.co/Gradio-Blocks)
- Join the discord
    - [discord](https://discord.com/invite/feTf9x3ZSB)


Participants will be building and sharing Gradio demos using the Blocks feature. We will share a list of ideas of spaces that can be created using blocks or participants are free to try out their own ideas. At the end of the event, spaces will be evaluated and prizes will be given.


## Potential ideas for creating spaces:


- Trending papers from https://paperswithcode.com/
- Models from huggingface model hub: https://huggingface.co/models
- Models from other model hubs
    - Tensorflow Hub: see example Gradio demos at https://huggingface.co/tensorflow
    - Pytorch Hub: see example Gradio demos at https://huggingface.co/pytorch
    - ONNX model Hub: see example Gradio demos at https://huggingface.co/onnx
    - PaddlePaddle Model Hub: see example Gradio demos at https://huggingface.co/PaddlePaddle
- participant ideas, try out your own ideas


## Prizes
- 1st place winner based on likes
    - [Hugging Face PRO subscription](https://huggingface.co/pricing) for 1 year
    - Embedding your Gradio Blocks demo in the Gradio Blog
- top 10 winners based on likes
    - Swag from [Hugging Face merch shop](https://huggingface.myshopify.com/): t-shirts, hoodies, mugs of your choice
- top 25 winners based on likes
    - [Hugging Face PRO subscription](https://huggingface.co/pricing) for 1 month
- Blocks event badge on HF for all participants!

## Prizes Criteria

- Staff Picks
- Most liked Spaces
- Community Pick (voting)
- Most Creative Space (voting)
- Most Educational Space (voting)
- CEO's pick (one prize for a particularly impactful demo), picked by @clem
- CTO's pick (one prize for a particularly technically impressive demo), picked by @julien


## Creating a Gradio demo on Hugging Face Spaces

Once a model has been picked from the choices above or feel free to try your own idea, you can share a model in a Space using Gradio

Read more about how to add [Gradio spaces](https://huggingface.co/blog/gradio-spaces).
 
Steps to add Gradio Spaces to the Gradio Blocks Party org
1. Create an account on Hugging Face
2. Join the Gradio Blocks Party Organization by clicking "Join Organization" button in the organization page or using the shared link above
3. Once your request is approved, add your space using the Gradio SDK and share the link with the community!

## LeaderBoard for Most Popular Blocks Event Spaces based on Likes

- See Leaderboard: https://huggingface.co/spaces/Gradio-Blocks/Leaderboard
