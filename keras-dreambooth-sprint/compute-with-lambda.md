## Launch a Lambda Cloud GPU
Where possible, we encourage you to fine-tune Dreambooth on a local GPU machine. This will mean a faster set-up and more familiarity with your device. 

The training scripts can also be run as a notebook through Google Colab. We recommend you train on Google Colab if you have a "Colab Pro" or "Pro+" subscription. This is to ensure that you receive a sufficiently powerful GPU on your Colab for fine-tuning Stable Diffusion. 

If you do not have access to a local GPU or Colab Pro/Pro+, we'll endeavour to provide you with a cloud GPU instance.
We've partnered up with Lambda to provide cloud compute for this event. They'll be providing the NVIDIA A10 24 GB GPUs. The Lambda API makes it easy to spin-up and launch a GPU instance. In this section, we'll go through the steps for spinning up an instance one-by-one.

<p align="center" width="100%">
    <img width="50%" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hf_lambda.png">
</p>

This section is split into three parts:

- [Launch a Lambda Cloud GPU](#launch-a-lambda-cloud-gpu)
  - [Signing-Up with Lambda](#signing-up-with-lambda)
  - [Creating a Cloud Instance](#creating-a-cloud-instance)
- [Setting up your environment](#setting-up-your-environment)
  - [Deleting a Cloud Instance](#deleting-a-cloud-instance)

### Signing-Up with Lambda

1. Create an account with Lambda using your email address of choice: http://lambdalabs.com/HF-dreambooth-signup. If you already have an account, skip to step 2.
2. Using this same email address, email `cloud@lambdal.com` with the Subject line: `Lambda cloud account for HuggingFace Keras DreamBooth - payment authentication and credit request`.
3. Each user who emails as above will receive $20 in credits (amounting to 60 fine-tuning runs/30 hours of A10).
4. Register a valid payment method with Lambda in order to redeem the credits (see instructions below).

To redeem these credits, you will need to authorise a valid payment method with Lambda. Provided that you remain within $20 of compute spending, your card **will not** be charged 💸. Registering your card with Lambda is a mandatory sign-up step that we unfortunately cannot bypass. But we reiterate: you will not be charged provided you remain within $20 of compute.

Follow steps 1-4 in the next section [Creating a Cloud Instance](#creating-a-cloud-instance) to register your card. If you experience issues with registering your card, contact the Lambda team on Discord (see [Communications and Problems](#communication-and-problems)).

In order to maximise the free GPU hours you have available for training, we advise that you shut down GPUs when you are not using them and closely monitor your GPU usage. We've detailed the steps you can follow to achieve this in [Deleting a Cloud Instance](#deleting-a-cloud-instance).

### Creating a Cloud Instance
Estimated time to complete: 5 mins

*You can also follow our video tutorial to set up a cloud instance on Lambda* 👉️ [YouTube Video](https://www.youtube.com/watch?v=Ndm9CROuk5g&list=PLo2EIpI_JMQtncHQHdHq2cinRVk_VZdGW)

1. Click the link: http://lambdalabs.com/HF-dreambooth-instances
2. You'll be asked to sign in to your Lambda account (if you haven't done so already).
3. Once on the GPU instance page, click the purple button "Launch instance" in the top right.
4. Verify a payment method if you haven't done so already. IMPORTANT: if you have followed the instructions in the previous section, you will have received $20 in GPU credits. Exceeding 25 hours of 1x A10 usage may incur charges on your credit card. Contact the Lambda team on Discord if you have issues authenticating your payment method (see [Communications and Problems](#communication-and-problems))
5. Launching an instance:
   1. In "Instance type", select the instance type "1x A10 (24 GB PCle)". In case you run out or memory while training, come back here and choose instance of type "1x A100(40GB PCIe)" or "1x A100(40GB SXM4)".
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
   5. Give your SSH key a memorable name (e.g. `merve-ssh-key`)
   6. Click "Add SSH Key"
7. Select the SSH key from the drop-down menu and click "Launch instance"
8. Read the terms of use and agree
9. We can now see on the "GPU instances" page that our device is booting up!
10. Once the device status changes to "✅ Running", click on the SSH login ("ssh ubuntu@..."). This will copy the SSH login to your clipboard.
11. Now open a new command line window, paste the SSH login, and hit Enter.
12. If asked "Are you sure you want to continue connecting?", type "yes" and press Enter.
13. Great! You're now SSH'd into your A10 device! We're now ready to set up our Python environment!

You can see your total GPU usage from the Lambda cloud interface: https://cloud.lambdalabs.com/usage

Here, you can see the total charges that you have incurred since the start of the event. We advise that you check your total on a daily basis to make sure that it remains below the credit allocation of $20. This ensures that you are not inadvertently charged for GPU hours.

If you are unable to SSH into your Lambda GPU in step 11, there is a workaround that you can try. On the [GPU instances page](http://lambdalabs.com/HF-dreambooth-instances), under the column "Cloud IDE", click the button "Launch". This will launch a Jupyter Lab on your GPU which will be displayed in your browser. In the top left-hand corner, click "File" -> "New" -> "Terminal". This will open up a new terminal window. You can use this terminal window to set up Python environment and install dependencies and run scripts.


## Setting up your environment

You can establish an SSH tunnel to your instance using below command: 
```
ssh ubuntu@ADDRESS_OF_INSTANCE -L 8888:localhost:8888
```
This will establish the tunnel to a remote machine and also forward the SSH port to a local port, so you can open a jupyter notebook on the remote machine and access it from your own local machine. 
We will use **TensorFlow** and **Keras CV** to train DreamBooth model, and later use **diffusers** for conversion. In this section, we'll cover how to set up an environment with the required libraries. This section assumes that you are SSH'd into your GPU device. 

You can setup your environment like below. 
Below script:
1. Creates a python virtual environment,
2. Installs the requirements,
3. Does authentication for Hugging Face. 
After you run `huggingface-cli login`, pass your write token that you can get from [here](https://huggingface.co/settings/tokens). This will authenticate you to push your models to Hugging Face Hub.

We will use conda for this (follow this especially if you are training on A10). Install miniconda like below:
```bash
sudo wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sudo chmod +x Miniconda3-latest-Linux-x86_64.sh && ./Miniconda3-latest-Linux-x86_64.sh
```
Accept the terms by typing "yes", confirm the path by pressing enter and then confirm `conda init` by typing in yes again.
To make conda commands accessible in the current shell environment enter:
```bash
source ~/.bashrc
```
Disable the base virtual conda environment:
```bash
conda config --set auto_activate_base false
conda deactivate
```
Now activate conda and create your own environment (in this example we use `my_env` for simplicity).
```bash
conda create -n my_env python==3.10
conda activate my_env
 ```
As a next step, we may confirm that pip points to the correct path:
 ```bash
 which pip
 ```
The path should point to `/home/ubuntu/miniconda3/envs/my_env/bin/pip`.

** Note: Please make sure you are opening the notebook either in env (if you are using Python virtual environment by following above commands) or use ipykernel to add your environment to jupyter. For first one, you can get into env folder itself and create your notebook there and it should work.**

As a next step, we need to install necessary dependencies for CUDA Support to work properly and getting a jupyter notebook running. Ensure you are inside the `my_env` conda environment you created previously:
```bash
conda install nb_conda_kernels
ipython kernel install --user --name=my_env
conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
```
Next you need to setup XLA to the correct CUDA library path with following command:
 ```bash
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
 ```
 ** Note: you need to set this every time you close and open the terminal via SSH tunnel. If you do not do this, the `fit` method will fail. Please read through the error logs to see where to find the missing library and set the above path accordingly.

Now, we also must install Tensorflow inside our virtual environment. It is recommend, doing so with pip:

 ```bash
python -m pip install tensorflow
 ```
 To confirm the installed version, and the success of setting up our drivers in the conda environment:
 ```bash
 python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU')); print(tf.__version__)"
 ```
It should return True, and display an array with one physical device. The version should be equal to atleast 2.11.

We may now install the dependendencies necessary for the jupyter notebook:
 ```bash
pip install keras_cv===0.4.2 tensorflow_datasets===4.8.1 pillow===9.4.0 imutils opencv-python matplotlib huggingface-hub pycocotools
 ```
 
Now we can start our jupyter notebook instance:
```bash
jupyter notebook
```
Enter the URL in the browser or connect through VSCode. If this does not work, you likely forgot to forward the 8888 port.
When you open jupyter, select your environment `my_env` in `New` dropdown and it will create your notebook with conda environment you've created.

Now inside the notebook:

First check the pip and python are poinitng to right places by running following commands. First check for pip path by running:
```python
!which pip
```
It should point to `/home/ubuntu/miniconda3/envs/my_env/bin/pip`. If it is pointing to `/home/ubuntu/.local/bin/pip`, you have not have run `conda config --set auto_activate_base false`. Please run it again and activate `my_env` again. Also check that your notebook is running in the proper kernel `my_env`. Once inside the notebook, you can change it from the menu navigation `Kernel->Change Kernel -> my_env`. You should now see `my_env` in the top right of the notebook. 

Now check for python path aswell:
```python
!which python
```
It should point to: `/home/ubuntu/miniconda3/envs/my_env/bin/python`

Running below line in the notebook makes sure that we have installed the version of TensorFlow that supports GPU, and that TensorFlow can detect the GPUs. If everything goes right, it should return `True` and a list that consists of a GPU. The version should be equal to or greater than 2.11 to support the correct version of keras_cv.
```python
import tensorflow as tf
print(tf.test.is_built_with_cuda())
print(tf.config.list_logical_devices('GPU'))
print(tf.__version__)
```

You can either create your own notebook or clone the notebook `https://github.com/huggingface/community-events/blob/main/keras-dreambooth-sprint/Dreambooth_on_Hub.ipynb` if you haven't done so previously.

You're all set! You can simply launch a jupyter notebook and start training models! 🚀 

### Deleting a Cloud Instance

30 1x A10 hours should provide you with enough time for 60 fine-tuning runs for Dreambooth. To maximise the GPU time you have for training, we advise that you shut down GPUs over prolonged periods of time when they are not in use. So be smart and shut down your GPU when you're not training.

Creating an instance and setting it up for the first time may take up to 20 minutes. Subsequently, this process will be much faster as you gain familiarity with the steps, so you shouldn't worry about having to delete a GPU and spinning one up the next time you need one. You can expect to spin-up and delete 2-3 GPUs over the course of the fine-tuning event.

We'll quickly run through the steps for deleting a Lambda GPU. You can come back to these steps after you've performed your first training run and you want to shut down the GPU:

1. Go to the instances page: http://lambdalabs.com/HF-dreambooth-instances
2. Click the checkbox on the left next to the GPU device you want to delete
3. Click the button "Terminate" in the top right-hand side of your screen (under the purple button "Launch instance")
4. Type "erase data on instance" in the text box and press "ok"

Your GPU device is now deleted and will stop consuming GPU credits.
