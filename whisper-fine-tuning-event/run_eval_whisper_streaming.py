import argparse

from transformers import pipeline
from datasets import load_dataset, Audio
import evaluate

wer_metric = evaluate.load("wer")


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    else:
        raise ValueError(f"Sample: {sample.keys()} has no transcript.")


def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": item["norm_text"]}


def main(args):
    batch_size = args.batch_size
    whisper_asr = pipeline(
        "automatic-speech-recognition", model=args.model_id, device=args.device
    )
    whisper_asr.model.config.max_length = 128

    whisper_norm = whisper_asr.tokenizer._normalize

    def normalise(batch):
        batch["norm_text"] = whisper_norm(get_text(batch))
        return batch

    dataset = load_dataset(
        args.dataset, args.config, split=args.split, streaming=True, use_auth_token=True
    )

    # Only uncomment for debugging
    dataset = dataset.take(64)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(normalise)
    dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])

    predictions = []
    references = []

    # run streamed inference
    for out in whisper_asr(data(dataset), batch_size=batch_size):
        predictions.append(whisper_norm(out["text"]))
        references.append(out["reference"][0])

    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)

    print("WER:", wer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to evaluate the `model_id`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config of the dataset. *E.g.* `'en'`  for Common Voice",
    )
    parser.add_argument(
        "--split", type=str, required=True, help="Split of the dataset. *E.g.* `'test'`"
    )

    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    args = parser.parse_args()

    main(args)
