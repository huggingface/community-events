# 🤗 Upload custom image dataset to the hub

This directory contains an example script that showcases how to upload a custom image dataset to the hub programmatically (using Python).

In this example, we'll upload all available datasets shared by the [CycleGAN authors](https://github.com/junyanz/CycleGAN/blob/master/datasets/download_dataset.sh) to the hub.

It leverages the [ImageFolder](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder) and `push_to_hub`
functionalities of the 🤗 [Datasets](https://huggingface.co/docs/datasets/index) library. 

It can be run as follows:

### 1. Make sure to have git-LFS installed on your system:
First, verify that you have git-LFS installed. This can be done by running:

```bash
git-lfs -v
```

If you get "command not found", then install it as follows:

```bash
sudo apt-get install git-lfs
```

### 2. Login with your HuggingFace account:
Next, one needs to provide a token for authentication with the hub. This can be done by either running:
 
```bash
huggingface-cli login
```

or 

```python
from huggingface_hub import notebook_login

notebook_login()
```

in case you're running in a notebook.

### 3. Upload!
Finally, uploading is as easy as:

```bash
python push_to_hub_example.py --dataset horse2zebra
````

The result can be seen [here](https://huggingface.co/datasets/huggan/horse2zebra).

Note that it's not required to programmatically upload a dataset to the hub: you can also do it in your browser as explained in [this guide](https://huggingface.co/docs/datasets/upload_dataset).
