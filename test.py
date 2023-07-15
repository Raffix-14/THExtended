import ArgsParser
import logging
import os
from datetime import datetime
from utils import setup_logging, make_deterministic, prepare_dataset, get_scores, compute_similarities, compute_rouges, compute_mrr_single_doc, trigram_blocking
import torch
import numpy as np
from multiprocessing import cpu_count
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def evaluate_model(dataset, model, tokenizer, args):
    """
    Evaluate the model on the test set
    :param dataset: the test set
    :param model: the model
    :param tokenizer: the tokenizer
    :param args: the arguments passed through command line
    :return: the average rouge scores, the average semantic similarity scores, the average mrr scores
    """
    num_highlights = args.num_highlights
    current_context = None
    current_article_sentences = []
    current_highlights = None
    rouges = []
    similarities = []
    mrrs = []

    progress_bar = tqdm(total=args.num_test_examples)
    progress_bar.set_description("Evaluating article")

    for example in dataset:
        sentence = example['sentence']
        context = example['context']
        highlights = example['highlights'].split("\n")

        # Check if context has changed
        if context != current_context:
            # Process previous article
            if current_context is not None:
                ranked_sents, ranked_scores = get_scores(current_article_sentences, current_context, model, tokenizer)
                ranked_sents = trigram_blocking(ranked_sents) if args.trigram_blocking == 1 else ranked_sents
                rouge_dict, similarity, mrr = evaluate_article(ranked_sents[:num_highlights], current_highlights)
                rouges.append(rouge_dict)
                similarities.append(similarity)
                mrrs.append(mrr)
                progress_bar.update(1)

            # Start a new article
            current_context = context
            current_highlights = highlights
            current_article_sentences = []

        # Append sentence to current article
        current_article_sentences.append(sentence)

    # Process the last article
    if current_context is not None:
        ranked_sents, ranked_scores = get_scores(current_article_sentences, current_context, model, tokenizer)
        ranked_sents = trigram_blocking(ranked_sents) if args.trigram_blocking == 1 else ranked_sents
        rouge_dict, similarity, mrr = evaluate_article(ranked_sents[:num_highlights], current_highlights)
        rouges.append(rouge_dict)
        similarities.append(similarity)
        mrrs.append(mrr)
        progress_bar.update(1)

    progress_bar.close()
    return compute_avg_dict(rouges), np.mean(similarities), np.mean(mrrs)


def evaluate_article(highlights_pred, highlights_gt):
    """
    Evaluate the model on a single article
    :param highlights_pred: the predicted highlights
    :param highlights_gt: the ground truth highlights
    :return: the average rouge scores, the average semantic similarity scores, the average mrr scores
    """
    rouges = compute_rouges(highlights_pred, highlights_gt, is_test=True)
    semantic_similarities = compute_similarities(highlights_pred, highlights_gt, similarity_model)
    mrr = compute_mrr_single_doc(highlights_pred, highlights_gt)
    return compute_avg_dict(rouges), np.mean(semantic_similarities), mrr


def compute_avg_dict(dict_list):
    """
    Compute the average of values contained in dictionaries in a list
    :param dict_list: the list of dictionaries
    :return: the average dictionary
    """
    avg_dict = {}
    keys = ["rouge-1", "rouge-2", "rouge-l"]
    metrics = ["f", "p", "r"]

    for key in keys:
        avg_dict[key] = {}
        for metric in metrics:
            avg_dict[key][metric] = sum(dictionary[key][metric] for dictionary in dict_list) / len(dict_list)

    return avg_dict


def main():
    """
    Main function for testing
    """
    args = ArgsParser.parse_arguments()
    start_time = datetime.now()
    args.output_dir = os.path.join(args.output_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))

    setup_logging(os.path.join(args.output_dir, "logs"), "info")
    make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.output_dir}")
    logging.info(f"Using {torch.cuda.device_count()} GPUs and {cpu_count()} CPUs")

    model_name = args.model_name_or_path

    global similarity_model

    logging.info("\n|-------------------------------------------------------------------------------------------|")
    logging.info(f"##### DOWNLOADING MODEL {model_name} #####")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model.to(torch.device("cuda:0"))
    model.eval()
    logging.info("\n|-------------------------------------------------------------------------------------------|")

    logging.info("##### PREPARING TEST DATASETS #####")
    _, _, dataset_test = prepare_dataset(args.dataset_path,
                                         args.num_train_examples,
                                         args.num_val_examples,
                                         args.num_test_examples,
                                         args.save_dataset_on_disk,
                                         args.output_dir,
                                         args.seed)
    logging.info("\n|-------------------------------------------------------------------------------------------|")

    logging.debug("##### EXAMPLE DATAPOINT #####")
    logging.debug(dataset_test[0])
    logging.debug("|-------------------------------------------------------------------------------------------|")

    logging.info("##### EVALUATING MODEL #####")
    results = evaluate_model(dataset_test, model, tokenizer, args)
    for key, value in results[0].items():
        logging.info(f"{key}: {value}")
    logging.info(f"Mean semantic similarity: {results[1]}")
    logging.info(f"Mean reciprocal rank: {results[2]}")
    logging.info("\n|-------------------------------------------------------------------------------------------|")


if __name__ == '__main__':
    main()
