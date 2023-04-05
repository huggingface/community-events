import argparse
import logging
import random

import cv2
import jsonlines
import numpy as np
import requests
from datasets import load_dataset
from PIL import Image

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example of a data preprocessing script."
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="The directory to store the dataset",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="The directory to store cache",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="number of examples in the dataset",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="number of processors to use in `dataset.map()`",
    )

    args = parser.parse_args()
    return args


# filter for `max_train_samples``
def filter_function(example):
    if example["clip_similarity_vitb32"] < 0.3:
        return False
    if example["watermark_score"] > 0.4:
        return False
    if example["aesthetic_score_laion_v2"] < 6.0:
        return False
    return True


def filter_dataset(dataset, max_train_samples):
    small_dataset = dataset.select(range(max_train_samples)).filter(filter_function)
    return small_dataset


if __name__ == "__main__":
    args = parse_args()

    # load coyo-700
    dataset = load_dataset(
        "kakaobrain/coyo-700m",
        cache_dir=args.cache_dir,
        split="train",
    )

    # estimation the % of images filtered
    filter_ratio = len(filter_dataset(dataset, 20000)) / 20000

    # esimate max_train_samples based on
    #   (1) filter_ratio we calculuted with 20k examples
    #   (2) assumption that only 80% of the URLs are still valid
    max_train_samples = int(args.max_train_samples / filter_ratio / 0.8)

    # filter dataset down to 1 million
    small_dataset = filter_dataset(dataset, max_train_samples)

    def preprocess_and_save(example):
        image_url = example["url"]
        try:
            # download original image
            image = Image.open(requests.get(image_url, stream=True, timeout=5).raw)
            image_path = f"{args.train_data_dir}/images/{example['id']}.png"
            image.save(image_path)

            # generate and save canny image
            processed_image = np.array(image)

            # apply random threholds
            #   note that this should normally be applied on the fly during training.
            #   But that's fine when dealing with a larger dataset like here.
            threholds = (
                random.randint(0, 255),
                random.randint(0, 255),
            )
            processed_image = cv2.Canny(processed_image, min(threholds), max(threholds))
            processed_image = processed_image[:, :, None]
            processed_image = np.concatenate(
                [processed_image, processed_image, processed_image], axis=2
            )
            processed_image = Image.fromarray(processed_image)
            processed_image_path = (
                f"{args.train_data_dir}/processed_images/{example['id']}.png"
            )
            processed_image.save(processed_image_path)

            # write to meta.jsonl
            meta = {
                "image": image_path,
                "conditioning_image": processed_image_path,
                "caption": example["text"],
            }
            with jsonlines.open(
                f"{args.train_data_dir}/meta.jsonl", "a"
            ) as writer:  # for writing
                writer.write(meta)

        except Exception as e:
            logger.error(f"Failed to process image{image_url}: {str(e)}")

    # preprocess -> image, processed image and meta.jsonl
    small_dataset.map(preprocess_and_save, num_proc=args.num_proc)

    print(f"created data folder at: {args.train_data_dir}")
