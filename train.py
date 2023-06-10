import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from DataParser import DataParser
from utils import make_deterministic, setup_logging, cleaner
import os
import ArgsParser
from datetime import datetime
import logging
import torch
from multiprocessing import cpu_count
from tqdm import tqdm

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Initial setup: parser, logging...
args = ArgsParser.parse_arguments()
start_time = datetime.now()
args.output_dir = os.path.join("logs", args.output_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))

setup_logging(args.output_dir,"info")
make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {cpu_count()} CPUs")


def preprocess_function(examples):
    return tokenizer(examples["sentence"], examples["context"], truncation=True)


def prepare_dataset(dataset_path=None, num_train_examples=3000, num_val_examples=500, num_test_example=500, save_flag=False):
    logging.info("##### PREPARING TRAIN, VAL, TEST DATASETS #####")
    train_dataset, val_dataset, test_dataset = None, None, None

    # Check if a path was provided and if a file exists at the path
    if dataset_path is not None and os.path.isdir(dataset_path):
        for data_type in ["train", "validation", "test"]:
            data_path = os.path.join(dataset_path, data_type)
            try:
                logging.info(f"Loading {data_type} dataset from {data_path}")
                if data_type == "train":
                    train_dataset = load_from_disk(data_path)
                elif data_type == "validation":
                    val_dataset = load_from_disk(data_path)
                elif data_type == "test":
                    test_dataset = load_from_disk(data_path)
            except FileNotFoundError:
                logging.info(f"No dataset {data_type.upper()} split found at {data_path}. Loading default.")

    if train_dataset is None or val_dataset is None or test_dataset is None:
        logging.info("Loading CNN DailyMail dataset from Hugging Face Hub")
        raw_dataset = load_dataset("cnn_dailymail", "3.0.0", num_proc=cpu_count())

        # Apply cleaning and parsing steps to training dataset
        if train_dataset is None:
            logging.info("Cleaning and parsing TRAIN split")
            cleaned_train_dataset = cleaner(raw_dataset['train'], num_train_examples)
            parser = DataParser(dataset=cleaned_train_dataset)
            train_dataset = parser()

        # Apply cleaning and parsing steps to validation dataset
        if val_dataset is None:
            logging.info("Cleaning and parsing VALIDATION split")
            cleaned_val_dataset = cleaner(raw_dataset['validation'], num_val_examples)
            parser = DataParser(dataset=cleaned_val_dataset)
            val_dataset = parser()

        # Apply cleaning and parsing steps to test dataset
        if test_dataset is None:
            logging.info("Cleaning and parsing TEST split")
            cleaned_test_dataset = cleaner(raw_dataset['test'], num_test_example)
            parser = DataParser(dataset=cleaned_test_dataset)
            test_dataset = parser()

        # If save_flag is set and a path is provided, save the dataset
        if dataset_path is None:
            dataset_path = os.path.join("dataset", args.output_dir)
        if not os.path.exists(dataset_path):
            logging.debug("Creating folder {dataset_path} to store the dataset splits")
            os.makedirs(dataset_path, exist_ok=True)
        if save_flag:
            for data_type, dataset in zip(["train", "validation", "test"], [train_dataset, val_dataset, test_dataset]):
                data_path = os.path.join(dataset_path, data_type)
                logging.ino(f"Saving dataset {data_type.upper()} split to {data_path}")
                dataset.save_to_disk(data_path, num_proc=cpu_count())
    return train_dataset, val_dataset, test_dataset


def main():
    batch_size = 16
    num_labels = 1

    logging.info("|-------------------------------------------------------------------------------------------|")
    logging.info(f"##### DOWNLOADING MODEL {model_name} #####")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    logging.info("|-------------------------------------------------------------------------------------------|")
    dataset_train, dataset_val, dataset_test = prepare_dataset(args.dataset_path,
                                                              args.num_train_examples,
                                                              args.num_val_examples,
                                                              args.num_test_examples, 
                                                              args.save_dataset_on_disk)

    print(dataset_train[0])

    dataset_train = dataset_train.map(preprocess_function, batched=True)
    dataset_val = dataset_val.map(preprocess_function, batched=True)
    # dataset_test = dataset_test.map(preprocess_function, batched=True)
    logging.info("|-------------------------------------------------------------------------------------------|")

    print("###########")
    print(dataset_train[0])
    print("###########")
    print(tokenizer.decode(dataset_train[0]["input_ids"]))
    exit(1)

    default_args = {
        "output_dir": "tmp",
        "evaluation_strategy": "epoch",
        "num_train_epochs": 2,
        "log_level": "info",
        "report_to": "none",
    }

    # OPTIMIZED PARAMETERS TRAINING
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        fp16=True,
        **default_args,
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        tokenizer=tokenizer
        # compute_metrics=compute_metrics
    )
    trainer.train()
    return


if __name__ == '__main__':
    main()
