from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils import make_deterministic, setup_logging, prepare_dataset
import os
import ArgsParser
from datetime import datetime
import logging
import torch
from multiprocessing import cpu_count

# If the GPU is based on the nvdia Ampere architecture uncomment this line as it speed-up training up to 3x reducing memory footprint
# torch.backends.cuda.matmul.allow_tf32 = True


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
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


def tokenize_function(examples):
    return tokenizer(examples["sentence"], examples["context"], truncation="only_second", padding="max_length",
                     return_tensors="pt")


def main():
    num_labels = 1

    logging.info("\n|-------------------------------------------------------------------------------------------|")
    logging.info(f"##### DOWNLOADING MODEL {model_name} #####")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    logging.info("\n|-------------------------------------------------------------------------------------------|")

    logging.info("##### PREPARING TRAIN, VAL, TEST DATASETS #####")
    dataset_train, dataset_val, dataset_test = prepare_dataset(args.dataset_path,
                                                               args.num_train_examples,
                                                               args.num_val_examples,
                                                               args.num_test_examples,
                                                               args.save_dataset_on_disk,
                                                               args.output_dir,
                                                               args.seed)

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
        "eval_steps": 0.25,
        "num_train_epochs": args.epochs_num,
        "seed": args.seed,
        "log_level": "info",
        "report_to": "none"
    }

    # OPTIMIZED PARAMETERS TRAINING
    training_args = TrainingArguments(
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        fp16=True,
        learning_rate=args.lr,
        optim="adamw_torch",
        disable_tqdm=False,
        **default_args
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset_train_tked,
        eval_dataset=dataset_val_tked,
        tokenizer=tokenizer
    )
    trainer.train()
    return


if __name__ == '__main__':
    main()
