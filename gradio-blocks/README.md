# Gradio Blocks Party  🥳

![image](https://user-images.githubusercontent.com/81195143/167940868-082bdbed-fd35-4aa4-8188-189cd451008e.png)


_**Timeline**: May 16th, 2022 - May 31st, 2022_

---

Happy to invite you to the Gradio Blocks Party - a community event in which we will build cool machine learning demos using the new Gradio Blocks feature. Blocks allows you to build web-based demos in a flexible way using the Gradio library. The event will take place from May 16th to 31st. We will be organizing this event on github and the huggingface discord channel.


**What is Gradio?**

Gradio is a Python library that allows you to quickly build web-based machine learning demos, data science dashboards, or other kinds of web apps, entirely in Python. These web apps can be launched from wherever you use Python (jupyter notebooks, colab notebooks, Python terminal, etc.) and shared with anyone instantly using Gradio's auto-generated share links. To learn more about Gradio see the Get Started Guide: https://gradio.app/getting_started/

Gradio can be installed via pip

```pip install gradio```


**What is Blocks?**

`gradio.Blocks` is a low-level API that allows you to have full control over the data flows and layout of your application. You can build very complex, multi-step applications using Blocks. If you have already used gradio.Interface, you know that you can easily create fully-fledged machine learning demos with just a few lines of code. The Interface API is very convenient but in some cases may not be sufficiently flexible for your needs. For example, you might want to:

* Group together related demos as multiple tabs in one web app.
* Change the layout of your demo instead of just having all of the inputs on the left and outputs on the right.
* Have multi-step interfaces, in which the output of one model becomes the input to the next model, or have more flexible data flows in general.
* Change a component's properties (for example, the choices in a Dropdown) or its visibility based on user input.

To learn more about Blocks, see the [official guide](https://www.gradio.app/introduction_to_blocks/).

**What is Hugging Face Spaces?**

Spaces are a simple way to host ML demo apps directly on your profile or your organization’s profile on Hugging Face. This allows you to create your ML portfolio, showcase your projects at conferences or to stakeholders, and work collaboratively with other people in the ML ecosystem. Learn more about spaces [here](https://huggingface.co/docs/hub/spaces).

**How Do Gradio and Hugging Face work together?**

Hugging Face Spaces is a free hosting option for Gradio demos. Spaces comes with 3 SDK options: Gradio, Streamlit and Static HTML demos. Spaces can be public or private and the workflow is similar to github repos. There are over 2000+ Gradio spaces currently on Hugging Face. 


**Event Plan**

In this event, you will:

1. Learning about Gradio and the new Blocks Feature
2. Building your own Blocks demo using Gradio and Hugging Face Spaces
3. Submitting your demo on Spaces to the Gradio Blocks Party Organization
4. Share your blocks demo with a permanent shareable link 
5. Win Prizes


**Example spaces using Blocks**
<img width="1302" alt="Screen Shot 2022-05-11 at 3 03 29 PM" src="https://user-images.githubusercontent.com/81195143/167926383-0faf0561-ce3f-4415-a36e-9516b814157e.png">
- [https://huggingface.co/spaces/multimodalart/mindseye-lite](https://huggingface.co/spaces/multimodalart/mindseye-lite)
- [https://huggingface.co/spaces/akhaliq/ArcaneGAN-blocks](https://huggingface.co/spaces/akhaliq/ArcaneGAN-blocks)
- [https://huggingface.co/spaces/merve/gr-blocks](https://huggingface.co/spaces/merve/gr-blocks)
- [https://huggingface.co/spaces/osanseviero/tortoisse-tts](https://huggingface.co/spaces/osanseviero/tortoisse-tts)
- [https://huggingface.co/spaces/akhaliq/CaptchaCracker](https://huggingface.co/spaces/akhaliq/CaptchaCracker)


To participate in the event

- Join the organization for Blocks event
    - [https://huggingface.co/Gradio-Blocks](https://huggingface.co/Gradio-Blocks)
- Join the discord
    - [discord](https://discord.com/invite/feTf9x3ZSB)


Participants will be building and sharing Gradio demos using the Blocks feature. We will share a list of ideas of spaces that can be created using blocks or participants are free to try out their own ideas. At the end of the event, spaces will be evaluated for creativity and prizes will be given. 


potential ideas for creating spaces:


- papers from https://paperswithcode.com/
- themed spaces: Vision, NLP, Audio based spaces
- Models from huggingface model hub
- Models from other model hubs
    - Tensorflow Hub: see example Gradio demos at https://huggingface.co/tensorflow
    - Pytorch Hub: see example Gradio demos at https://huggingface.co/pytorch
    - ONNX model Hub: see example Gradio demos at https://huggingface.co/onnx
    - PaddlePaddle Model Hub: see example Gradio demos at https://huggingface.co/PaddlePaddle
- participant ideas, try out your own ideas


**Prizes**

- [Hugging Face PRO subscription](https://huggingface.co/pricing) 1 month for top 25 winners and 1 year for 1st place winner!
- [Hugging Face PRO subscription](https://huggingface.co/pricing) badge on Hugging Face for winners
- Gradio badge for everyone participating!
- Blocks event badge on HF for all participants!
- Swag from [Hugging Face merch shop](https://huggingface.myshopify.com/): t-shirts, hoodies, mugs of your choice for top 10 winners!
- Embedding your Gradio Blocks demo in the Gradio Blog for 1st place winner
- Gradio Team Office Hour for 1st place winner

**Prizes Criteria**

- Staff Picks
- Most liked Spaces
- Community Pick (voting)
- Most Creative Space (voting)


**Creating a Gradio demo on Hugging Face Spaces**

Once a model has been picked from the choices above or feel free to try your own idea, you can share a model in a Space using Gradio

Read more about how to add Gradio spaces [here](https://huggingface.co/blog/gradio-spaces).
 
Steps to add Gradio Spaces to the Gradio Blocks Party org
1. Create an account on Hugging Face
2. Join the Gradio Blocks Party Organization by clicking "Join Organization" button in the organization page
3. Once your request is approved, add your space using the Gradio SDK
