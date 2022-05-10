# Gradio Blocks Party  🥳


_**Timeline**: May 11th, 2022 - May 31st, 2022_

---

Happy to invite you to the Gradio Blocks Party - a community event in which we will setup spaces for models using the new Gradio Blocks feature. Blocks allows you to build web-based demos in a flexible way using the Gradio library. The event will take place from May 11th to 31st. We will be organizing this event on github and the huggingface discord channel.

**Introduction**

The goal of this event if for people to become more familiar with the blocks feature of Gradio. Some example spaces that already use blocks can be found at 

- [https://huggingface.co/spaces/akhaliq/ArcaneGAN-blocks](https://huggingface.co/spaces/akhaliq/ArcaneGAN-blocks)
- [https://huggingface.co/spaces/merve/gr-blocks](https://huggingface.co/spaces/merve/gr-blocks)
- [https://huggingface.co/spaces/osanseviero/tortoisse-tts](https://huggingface.co/spaces/osanseviero/tortoisse-tts)
- [https://huggingface.co/spaces/akhaliq/CaptchaCracker](https://huggingface.co/spaces/akhaliq/CaptchaCracker)

main components of the event consist of:

- Learning about Gradio and the new Blocks Feature
- How Gradio and Hugging Face Spaces work together
- Understanding the Hugging Face ecosystem
- Share Demos in a Hugging Face organization for the Blocks event

**What is Gradio?**

Gradio is a Python library that allows you to quickly build web-based machine learning demos, data science dashboards, or other kinds of web apps, entirely in Python. These web apps can be launched from wherever you use Python (jupyter notebooks, colab notebooks, Python terminal, etc.) and shared with anyone instantly using Gradio's auto-generated share links. To learn more about Gradio see the Get Started Guide: https://gradio.app/getting_started/

Gradio can be installed via pip

```pip install gradio```


**What is Blocks?**

gradio.Blocks is a low-level API that allows you to have full control over the data flows and layout of your application. You can build very complex, multi-step applications using Blocks. If you have already used gradio.Interface, you know that you can easily create fully-fledged machine learning demos with just a few lines of code. The Interface API is very convenient but in some cases may not be sufficiently flexible for your needs. For example, you might want to:

Group together related demos as multiple tabs in one web app
Change the layout of your demo instead of just having all of the inputs on the left and outputs on the right
Have multi-step interfaces, in which the output of one model becomes the input to the next model, or have more flexible data flows in general
Change a component's properties (for example, the choices in a Dropdown) or its visibilty based on user input

to learn more about Blocks see the guide


**How Does Gradio and Hugging Face work together?**

Hugging Face Spaces is a free hosting option for Gradio demos. Spaces comes with 3 SDK options: Gradio, Streamlit and Static HTML demos. Spaces can be public or private and the workflow is similar to github repos. There are over 2000+ spaces currently on Hugging Face. Learn more about spaces [here](https://huggingface.co/docs/hub/spaces).

**Event Plan**

To participate in the event

- Join the organization for Blocks event
    - [https://huggingface.co/Gradio-Blocks](https://huggingface.co/Gradio-Blocks)
- Join the discord

Participants will have a month to build and share Gradio demos using the Blocks feature. We will share a list of ideas of spaces that can be created using blocks or participants are free to try out their own ideas. At the end of the event, spaces will be evaluated for creativity and prizes will be given 

potential ideas for creating spaces:


- papers from https://paperswithcode.com/
- themed spaces: Vision, NLP, Audio based spaces
- Models from huggingface model hub
- Models from other model hubs
    - Tensorflow Hub
    - Pytorch Hub
    - ONNX model Hub
    - PaddlePaddle Model Hub
- participant ideas


**Prizes**

- HF PRO 1 month or 1 year!
- 1st place 12 months, 2nd 6m, 3rd 3m, 4th 1m HF PRO.
- Legendary prize, Lifetime HF PRO!!!
- HF PRO badge on HF
- Gradio badge on HF
- Special event badge on HF
- A lot more swag: t-shirts, hoodies, mugs, cluth
- Embed the demo or highlight it
- Time with someone on the Gradio team (unique ?)


**Creating a Gradio demo on Hugging Face Spaces**

Once a model has been picked from the choices above, you can share a model in a Space using Gradio


 Read more about how to add Gradio spaces [here](https://huggingface.co/blog/gradio-spaces).
 
 
Steps to add Gradio Spaces to the Gradio Blocks Party org
- Create a account on Hugging Face
- Join the Gradio Blocks Party Org
- Once approved add your space using the Gradio SDK
