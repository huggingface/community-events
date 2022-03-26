# GAN metrics

In order to track progress in (un)conditional image generation, a few quantitative metrics have been proposed. Below, we explain the most common ones.
For a more extensive overview, we refer the reader to (Borji, 2018)[https://arxiv.org/abs/1802.03446].

Quantitative metrics are of course just a proxy of image quality. The most widely used (Inception Score and FID) have several drawbacks [(Barratt et al., 2018)](https://arxiv.org/abs/1801.01973), [Sajjadi et al., 2018](https://arxiv.org/abs/1806.00035), [Kynkäänniemi et al., 2019](https://arxiv.org/abs/1904.06991).

## Inception score

## Fréchet Inception Distance (FID)

The FID metric was proposed in [(Heusel et al., 2018)](https://arxiv.org/abs/1706.08500), and is currently the most widely used metric for evaluating image generation. 

A widely used PyTorch implementation can be found [here](https://github.com/mseitzer/pytorch-fid).

The Fréchet distance meaures the distance between 2 multivariate Gaussian distributions. What does that mean? Concretely, the FID metric uses a pre-trained neural network (namely, Inceptionv3), and first forwards both real and generated images through it in order to get feature maps. Next, one computes statistics (namely, the mean and standard deviation) of the feature maps for both distributions. Finally, the distance between both distributions (real and fake) is computed based on these statistics.

The FID metric assumes that feature maps of a pre-trained neural net extracted on real vs. fake images should be similar (the authors argue that this is a good quantitative metric for assessing image quality). 

An important disadvantage of the FID metric is that is has an issue of generalization; a model that simply memorizes the training data can obtain a perfect score on these metrics [Razavi et al., 2019](https://arxiv.org/abs/1906.00446).

## Clean FID

In 2021, a paper by [Parmar et al.](https://arxiv.org/abs/2104.11222) indicated that the FID metric is often poorly computed, due to incorrect implementations of low-level image preprocessing (such as resizing of images) in popular frameworks such as PyTorch and TensorFlow. This can produce widely different values for the FID metric.

The official implementation of the cleaner FID version can be found [here](https://github.com/GaParmar/clean-fid).