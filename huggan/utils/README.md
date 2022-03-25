This directory contains a script that showcases how to upload a custom image dataset to the hub programmatically (using Python).

It leverages the [ImageFolder](https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder) and `push_to_hub`
functionalities of the 🤗 [Datasets](https://huggingface.co/docs/datasets/index) library. 

It can be run as follows:

### 1. Make sure to have git-LFS installed on your system:
To verify that you have git-LFS installed, run:

```bash
git-lfs -v
```

If you get "command not found", then install it as follows:

```bash
sudo apt-get install git-lfs
```

### 2. Login with your HuggingFace account:
 
```bash
huggingface-cli login
```

### 3. Upload!

```bash
python push_to_hub_example.py --dataset horse2zebra
````

Note that it's not required to programmatically upload a dataset to the hub: you can also do it in your browser as explained in [this guide](https://huggingface.co/docs/datasets/upload_dataset).
