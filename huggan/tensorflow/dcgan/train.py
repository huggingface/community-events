
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import huggingface_hub
import tensorflow as tf
from pathlib import Path
import os
import PIL
import argparse
from tensorflow.keras import layers
from huggingface_hub import push_to_hub_keras
from datasets import load_dataset



def stack_generator_layers(model, units):
    model.add(layers.Conv2DTranspose(1024, (4, 4), strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())  
    return model 

def create_generator(channel, hidden_size, latent_dim):
    generator = tf.keras.Sequential()
    generator.add(layers.Input((latent_dim,))) # 
    generator.add(layers.Dense(hidden_size*8, use_bias=False, input_shape=(100,)))
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU())

    generator.add(layers.Reshape((4, 4, 1024)))

    units = [hidden_size*8, hidden_size*4, hidden_size*2, hidden_size*2, hidden_size]
    for unit in units:
        generator = stack_generator_layers(generator, unit)

    generator.add(layers.Conv2DTranspose(channel, (4, 4), strides=1, padding='same', use_bias=False, activation='tanh'))
    return generator

def stack_discriminator_layers(model, units, use_batch_norm=False, use_dropout=False):
    model.add(layers.Conv2D(units, (4, 4), strides=(2, 2), padding='same'))
    if use_batch_norm:
        model.add(layers.BatchNormalization())
    if use_dropout:
        model.add(layers.Dropout(0.1))
    model.add(layers.LeakyReLU())
    return model

def create_discriminator(channel, hidden_size):
    discriminator = tf.keras.Sequential()
    discriminator.add(layers.Input((64, 64, channel)))
    discriminator = stack_discriminator_layers(discriminator, hidden_size, use_batch_norm = True, use_dropout = True)
    discriminator = stack_discriminator_layers(discriminator, hidden_size * 2)
    discriminator = stack_discriminator_layers(discriminator, hidden_size*2, use_batch_norm = True, use_dropout = True)
    discriminator = stack_discriminator_layers(discriminator, hidden_size*4, use_batch_norm = True)
    discriminator = stack_discriminator_layers(discriminator, hidden_size*8, use_batch_norm = True)

    discriminator.add(layers.Flatten())
    discriminator.add(layers.Dense(1))

    return discriminator


def discriminator_loss(real_image, generated_image):
    real_loss = cross_entropy(tf.ones_like(real_image), real_image)
    fake_loss = cross_entropy(tf.zeros_like(generated_image), generated_image)
    total_loss = real_loss + fake_loss
    return total_loss


@tf.function
def train_step(images):
    noise = tf.random.normal([256, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_image = discriminator(images, training=True)
      generated_image = discriminator(generated_images, training=True)
      # calculate loss inside train step
      gen_loss = cross_entropy(tf.ones_like(generated_image), generated_image)
      disc_loss = discriminator_loss(real_image, generated_image)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



def generate_and_save_images(model, epoch, test_input, output_dir):
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(16, 64))

  for i in range(predictions.shape[0]):
      plt.subplot(1, 4, i+1)
      plt.imshow(predictions[i, :, :, :])
      plt.axis('off')

  plt.savefig(f'{output_dir}/image_at_epoch_{epoch}.png')


def train(dataset, epochs, output_dir):
  for epoch in range(epochs):
    for image_batch in dataset:
      train_step(image_batch)

    generate_and_save_images(generator,
                             epoch + 1,
                             seed,
                             output_dir)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to load from the HuggingFace hub.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers when loading data")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size to use during training")
    parser.add_argument("--number_of_examples_to_generate", type=int, default=4, help="Number of examples to be generated in inference mode")
    parser.add_argument(
        "--generator_hidden_size",
        type=int,
        default=64,
        help="Hidden size of the generator's feature maps.",
    )
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimensionality of the latent space.")

    parser.add_argument(
        "--discriminator_hidden_size",
        type=int,
        default=64,
        help="Hidden size of the discriminator's feature maps.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Spatial size to use when resizing images for training.",
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=3,
        help="Number of channels in the training images. For color images this is 3.",
    )
    parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs of training")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--output_dir", type=Path, default=Path("./output"), help="Name of the directory to dump generated images during training.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the HuggingFace hub after training.",
        )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="Name of the model on the hub.",
    )
    parser.add_argument(
        "--organization_name",
        default="huggan",
        type=str,
        help="Organization name to push to, in case args.push_to_hub is specified.",
    )
    args = parser.parse_args()
    
    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."
        assert args.model_name is not None, "Need a `model_name` to create a repo when `--push_to_hub` is passed."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    return args



def train(dataset, epochs):
  for epoch in range(args.epochs):
    for image_batch in dataset:
      train_step(image_batch)

    generate_and_save_images(generator,
                             epoch + 1,
                             seed)



def preprocess_dataset(dataset, BATCH_SIZE):
    preprocessed_images = []
    print("Preprocessing dataset..")
    breakpoint()
    for img in dataset["train"]:
        img = img["image"].convert('RGB') 
        img = np.array(img) 
        img = tf.keras.utils.img_to_array(PIL.Image.fromarray(img))
        preprocessed_images.append(img)
    train_images = np.asarray(preprocessed_images)
    channel = train_images[0].shape[-1]
    train_images = (train_images - 127.5) / 127.5 
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE)
    return train_dataset, channel

if __name__ == "__main__":
    args = parse_args()
    print("Downloading dataset..")
    dataset = load_dataset(args.dataset)
    dataset, channel = preprocess_dataset(dataset, args.batch_size)
    generator = create_generator(channel, args.generator_hidden_size, args.latent_dim)
    discriminator = create_discriminator(channel, args.discriminator_hidden_size)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # create seed with dimensions of number of examples to generate and noise
    seed = tf.random.normal([args.number_of_examples_to_generate, args.latent_dim])

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    train(dataset, args.num_epochs, args.output.dir)

    push_to_hub_keras(generator, repo_path_or_name=args.output_dir / args.model_name,organization=args.organization_name)