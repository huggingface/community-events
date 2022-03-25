import argparse
from datasets import load_dataset
from tqdm import tqdm

# choose a dataset
available_datasets = ["apple2orange", "summer2winter_yosemite", "horse2zebra", "monet2photo", "cezanne2photo", "ukiyoe2photo", "vangogh2photo", "maps", "cityscapes", "facades", "iphone2dslr_flower", "ae_photos", "grumpifycat"]

def upload_dataset(dataset_name):
    if dataset_name not in available_datasets:
        raise ValueError("Please choose one of the supported datasets:", available_datasets)
    
    # step 1: load dataset
    dataset = load_dataset("imagefolder", data_files=f"https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/{dataset_name}.zip")

    # step 2: push to hub
    dataset.push_to_hub(f"huggan/{dataset_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="apple2orange", type=str, help="Dataset to upload")
    args = parser.parse_args()
    
    upload_dataset(args.dataset)


if __name__ == "__main__":
    main()
