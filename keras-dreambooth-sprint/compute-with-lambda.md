## Launch a Lambda Cloud GPU
Where possible, we encourage you to fine-tune Dreambooth on a local GPU machine. This will mean a faster set-up and more familiarity with your device. 

The training scripts can also be run as a notebook through Google Colab. We recommend you train on Google Colab if you have a "Colab Pro" or "Pro+" subscription. This is to ensure that you receive a sufficiently powerful GPU on your Colab for fine-tuning Whisper. 

If you do not have access to a local GPU or Colab Pro/Pro+, we'll endeavour to provide you with a cloud GPU instance.
We've partnered up with Lambda to provide cloud compute for this event. They'll be providing the NVIDIA A10 24 GB GPUs. The Lambda API makes it easy to spin-up and launch a GPU instance. In this section, we'll go through the steps for spinning up an instance one-by-one.

<p align="center" width="100%">
    <img width="50%" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hf_lambda.png">
</p>

This section is split into three parts:

1. [Signing-Up with Lambda](#signing-up-with-lambda)
2. [Creating a Cloud Instance](#creating-a-cloud-instance)
3. [Deleting a Cloud Instance](#deleting-a-cloud-instance)

### Signing-Up with Lambda

1. Create an account with Lambda using your email address of choice: https://cloud.lambdalabs.com/sign-up. If you already have an account, skip to step 2.
2. Using this same email address, email `cloud@lambdal.com` with the Subject line: `Lambda cloud account for HuggingFace Keras DreamBooth - payment authentication and credit request`.
3. Each user who emails as above will receive $15 in credits (amounting to 60 fine-tuning runs/25 hours of A10).
4. Register a valid payment method with Lambda in order to redeem the credits (see instructions below).

To redeem these credits, you will need to authorise a valid payment method with Lambda. Provided that you remain within $15 of compute spending, your card **will not** be charged 💸. Registering your card with Lambda is a mandatory sign-up step that we unfortunately cannot bypass. But we reiterate: you will not be charged provided you remain within $15 of compute.

Follow steps 1-4 in the next section [Creating a Cloud Instance](#creating-a-cloud-instance) to register your card. If you experience issues with registering your card, contact the Lambda team on Discord (see [Communications and Problems](#communication-and-problems)).

In order to maximise the free GPU hours you have available for training, we advise that you shut down GPUs when you are not using them and closely monitor your GPU usage. We've detailed the steps you can follow to achieve this in [Deleting a Cloud Instance](#deleting-a-cloud-instance).

### Creating a Cloud Instance
Estimated time to complete: 5 mins

*You can also follow our video tutorial to set up a cloud instance on Lambda* 👉️ [YouTube Video](https://www.youtube.com/watch?v=Ndm9CROuk5g&list=PLo2EIpI_JMQtncHQHdHq2cinRVk_VZdGW)

1. Click the link: https://cloud.lambdalabs.com/instances
2. You'll be asked to sign in to your Lambda account (if you haven't done so already).
3. Once on the GPU instance page, click the purple button "Launch instance" in the top right.
4. Verify a payment method if you haven't done so already. IMPORTANT: if you have followed the instructions in the previous section, you will have received $15 in GPU credits. Exceeding 25 hours of 1x A10 usage may incur charges on your credit card. Contact the Lambda team on Discord if you have issues authenticating your payment method (see [Communications and Problems](#communication-and-problems))
5. Launching an instance:
   1. In "Instance type", select the instance type "1x A10 (40 GB SXM4)"
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

Here, you can see the total charges that you have incurred since the start of the event. We advise that you check your 
total on a daily basis to make sure that it remains below the credit allocation of $15. This ensures that you are 
not inadvertently charged for GPU hours.

If you are unable to SSH into your Lambda GPU in step 11, there is a workaround that you can try. On the [GPU instances page](https://cloud.lambdalabs.com/instances), 
under the column "Cloud IDE", click the button "Launch". This will launch a Jupyter Lab on your GPU which will be displayed in your browser. In the 
top left-hand corner, click "File" -> "New" -> "Terminal". This will open up a new terminal window. You can use this 
terminal window to set up your Python environment in the next section [Set Up an Environment](#set-up-an-environment).

### Deleting a Cloud Instance

25 1x A10 hours should provide you with enough time for 50 fine-tuning runs for Dreambooth. To maximise the GPU time you have for training, we advise that you shut down GPUs over prolonged periods of time when they are not in use. Leaving a GPU running accidentally over the weekend will incur 48 hours of wasted GPU hours. That's nearly half of your compute allocation! So be smart and shut down your GPU when you're not training.

Creating an instance and setting it up for the first time may take up to 20 minutes. Subsequently, this process will be much faster as you gain familiarity with the steps, so you shouldn't worry about having to delete a GPU and spinning one up the next time you need one. You can expect to spin-up and delete 2-3 GPUs over the course of the fine-tuning event.

We'll quickly run through the steps for deleting a Lambda GPU. You can come back to these steps after you've performed your first training run and you want to shut down the GPU:

1. Go to the instances page: https://cloud.lambdalabs.com/instances
2. Click the checkbox on the left next to the GPU device you want to delete
3. Click the button "Terminate" in the top right-hand side of your screen (under the purple button "Launch instance")
4. Type "erase data on instance" in the text box and press "ok"

Your GPU device is now deleted and will stop consuming GPU credits.