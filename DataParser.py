import spacy
from datasets import Dataset, load_from_disk
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils import compute_rouge


class DataParser:

    def __init__(self, dataset=None, aggregation='max', is_test=False):
        self.dataset = dataset
        # Loading the spaCy model to split into sentences
        self.nlp = spacy.load('en_core_web_lg')
        self.aggregation = aggregation  # Aggregation method for ROUGE scores
        self.batch_size = int(np.ceil(len(self.dataset) // cpu_count()))  # Computing optimal the batch size
        self.parsedDataset = None
        self.is_test = is_test

    def split_sentence(self, text):
        return [self.clean_sentence(s.text) for s in self.nlp(text).sents]  # Splitting into sentences and cleaning

    @staticmethod
    def clean_sentence(s):
        return s.strip()  # Removing leading and trailing whitespaces

    def extract_context(self, article):
        sentences = self.split_sentence(article)[:3]
        return ' '.join(sent for sent in sentences)

    def process_row(self, row):
        # Unpack the parameter tuple
        article, summary = row
        article_sentences = self.split_sentence(article)
        context = self.extract_context(article)
        rouges = []

        if self.is_test:  # Optimization: skip rouge computation if test split for speed-up
            return {"sentences": article_sentences, "context": context, "highlights": summary}  # No need for labels

        highlights = summary.split("\n")  # Splitting the summary into single highlights
        for sentence in article_sentences:  # Computing ROUGE score (label) for each sentence
            score = compute_rouge(sentence, highlights, aggregation=self.aggregation, is_test=self.is_test)
            rouges.append(score)
        return {"sentences": article_sentences, "context": context, "labels": rouges}  # No need for highlights text

    def process_batch(self, batch):
        return [self.process_row(row) for row in batch]

    def __getitem__(self, idx):
        if self.parsedDataset is None:
            return {}
        return self.parsedDataset[idx]

    def __call__(self):
        articles = self.dataset["article"]
        summaries = self.dataset["highlights"]

        # Create a Pool object
        pool = Pool()
        # Apply pool.imap() with tqdm
        results = []
        with tqdm(total=cpu_count(), desc='Processing batch') as pbar:
            for result in pool.imap(self.process_batch,
                                    [zip(articles[i:min(i + self.batch_size, len(articles))],
                                         summaries[i:min(i + self.batch_size, len(articles))])
                                     for i in range(0, len(articles), self.batch_size)]):
                results.append(result)
                pbar.update()
        # Close the pool
        pool.close()
        pool.join()

        processed_data = [item for sublist in results for item in sublist]
        sentences_list = [item for sublist in processed_data for item in sublist["sentences"]]
        context_list = [context for sublist in processed_data for context in
                        [sublist["context"]] * len(sublist["sentences"])]
        # If test split, retrieve highlights, else retrieve label
        if self.is_test:
            highlights_list = [highlights for sublist in processed_data for highlights in
                               [sublist["highlights"]] * len(sublist["sentences"])]
            self.parsedDataset = Dataset.from_dict(
                {"sentence": sentences_list, "context": context_list, "highlights": highlights_list})
        else:
            labels_list = [label for sublist in processed_data for label in sublist["labels"]]
            self.parsedDataset = Dataset.from_dict(
                {"sentence": sentences_list, "context": context_list, "label": labels_list})
        return self.parsedDataset
