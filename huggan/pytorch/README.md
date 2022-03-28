This directory contains a few example scripts that allow you to train famous GANs on your own data using a bit of 🤗 magic.

More concretely, these scripts:
- leverage 🤗 Datasets to load any image dataset from the hub (including your own, possibly private, dataset)
- leverage 🤗 Accelerate to instantly run the script on (multi-) GPU environments, supporting fp16 and mixed precision
- leverage 🤗 Hub to push the model to the hub at the end of training, allowing to easily create a demo for it afterwards

# Example scripts

| Name      | Paper |
| ----------- | ----------- |
| [CycleGAN](pytorch/cyclegan/README.md)  | [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
| [DCGAN](pytorch/dcgan/README.md)  | [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)  |
| [pix2pix](pytorch/pix2pix/README.md) | [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) |
