from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils import make_deterministic, setup_logging, prepare_dataset, LogCallback
import os
import ArgsParser
from datetime import datetime
import logging
import torch
import numpy as np
from multiprocessing import cpu_count


# If the GPU is based on the nvdia Ampere architecture uncomment this line as it speed-up training up to 3x reducing memory footprint
# torch.backends.cuda.matmul.allow_tf32 = True


def tokenize_function(examples):
    return tokenizer(examples["sentence"], examples["context"], truncation="only_second", padding="max_length",
                     return_tensors="pt")


def combine_labels(example):
    rouge = np.array(example['rouge'])
    similarity = np.array(example['similarity'])

    batch_size = len(rouge)  # desired size of the array
    weight = np.full(batch_size, alpha, dtype=np.float16)
    label = weight * rouge + (np.ones(batch_size) - weight) * similarity
    example['labels'] = label
    return example


def main():
    # Initial setup: parser, logging...
    args = ArgsParser.parse_arguments()
    start_time = datetime.now()
    args.output_dir = os.path.join(args.output_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    setup_logging(os.path.join(args.output_dir, "logs"), "info")
    make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.output_dir}")
    logging.info(f"Using {torch.cuda.device_count()} GPUs and {cpu_count()} CPUs")
    model_name = args.model_name_or_path

    global tokenizer
    global alpha

    alpha = args.alpha

    logging.info("\n|-------------------------------------------------------------------------------------------|")
    logging.info(f"##### DOWNLOADING MODEL {model_name} #####")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    logging.info("\n|-------------------------------------------------------------------------------------------|")

    logging.info("##### PREPARING TRAIN, VAL, TEST DATASETS #####")
    dataset_train, dataset_val, _ = prepare_dataset(args.dataset_path,
                                                    args.num_train_examples,
                                                    args.num_val_examples,
                                                    args.num_test_examples,
                                                    args.save_dataset_on_disk,
                                                    args.output_dir,
                                                    args.seed)

    ###### #Change the labels to the desired combination

    dataset_train = dataset_train.map(combine_labels, batched=True)
    dataset_val = dataset_val.map(combine_labels, batched=True)

    dataset_train_tked = dataset_train.map(tokenize_function, batched=True)
    dataset_val_tked = dataset_val.map(tokenize_function, batched=True)
    # dataset_test = dataset_test.map(preprocess_function, batched=True)
    logging.info("\n|-------------------------------------------------------------------------------------------|")

    logging.debug("##### EXAMPLE DATAPOINT #####")
    logging.debug(dataset_train_tked[0])
    logging.debug("|-------------------------------------------------------------------------------------------|")

    model_out_dir = os.path.join(args.output_dir, "model")
    os.makedirs(model_out_dir, exist_ok=True)
    default_args = {
        "output_dir": model_out_dir,
        "evaluation_strategy": "steps",
        "eval_steps": 0.5,
        "save_strategy": "epoch",
        "seed": args.seed,
        "log_level": "passive",
        "log_level_replica": "passive",
        "report_to": "none"
    }

    # OPTIMIZED PARAMETERS TRAINING
    training_args = TrainingArguments(
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs_num,
        fp16=True,
        learning_rate=args.lr,
        optim="adamw_torch",
        disable_tqdm=False,
        save_safetensors=True,
        **default_args
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset_train_tked,
        eval_dataset=dataset_val_tked,
        tokenizer=tokenizer,
        callbacks=[LogCallback]
    )
    trainer.train()

    final_str = "##### FINAL RESULTS #####"
    for step in trainer.state.log_history:
        for k, v in step.items():
            final_str += str(k) + ': ' + str(v) + '\n'
        final_str += "---------------------------------\n"
    logging.info(final_str)

    return


if __name__ == '__main__':
    main()