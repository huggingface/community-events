# Example scripts (PyTorch)

This directory contains a few example scripts that allow you to train famous GANs on your own data using a bit of 🤗 magic.

More concretely, these scripts:
- leverage 🤗 [Datasets](https://huggingface.co/docs/datasets/index) to load any image dataset from the hub (including your own, possibly private, dataset)
- leverage 🤗 [Accelerate](https://huggingface.co/docs/accelerate/index) to instantly run the script on (multi-) CPU, (multi-) GPU, TPU environments, supporting fp16 and mixed precision as well as DeepSpeed
- leverage 🤗 [Hub](https://huggingface.co/) to push the model to the hub at the end of training, allowing to easily create a demo for it afterwards

Currently, it contains the following examples:

| Name      | Paper |
| ----------- | ----------- |
| [DCGAN](dcgan)  | [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)  |
| [pix2pix](pix2pix) | [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) |
| [CycleGAN](cyclegan)  | [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
| [Lightweight GAN](lightweight_gan) | [Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis](https://openreview.net/forum?id=1Fqg133qRaI)


