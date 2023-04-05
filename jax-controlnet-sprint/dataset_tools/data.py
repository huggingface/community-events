import os

import datasets
import pandas as pd

_VERSION = datasets.Version("0.0.2")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "conditioning_image": datasets.Image(),
        "text": datasets.Value("string"),
    },
)


_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)
DATA_DIR = "/mnt/disks/persist/data"


class coyo(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        metadata_path = f"{DATA_DIR}/meta.jsonl"
        images_dir = f"{DATA_DIR}/images"
        conditioning_images_dir = f"{DATA_DIR}/processed_images"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "images_dir": images_dir,
                    "conditioning_images_dir": conditioning_images_dir,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, images_dir, conditioning_images_dir):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            text = row["caption"]

            try:
                image_path = row["image"]
                image_path = os.path.join(images_dir, image_path)
                image = open(image_path, "rb").read()

                conditioning_image_path = row["conditioning_image"]
                conditioning_image_path = os.path.join(
                    conditioning_images_dir, row["conditioning_image"]
                )
                conditioning_image = open(conditioning_image_path, "rb").read()

                yield row["image"], {
                    "text": text,
                    "image": {
                        "path": image_path,
                        "bytes": image,
                    },
                    "conditioning_image": {
                        "path": conditioning_image_path,
                        "bytes": conditioning_image,
                    },
                }
            except Exception as e:
                print(e)
