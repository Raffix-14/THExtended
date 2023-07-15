import spacy
from datasets import Dataset
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils import compute_rouges, compute_similarities
import math
from sentence_transformers import SentenceTransformer


class DataParser:
    """
    Class to parse the dataset into a useful format
    """
    def __init__(self, dataset=None, aggregation='max', is_test=False):
        self.dataset = dataset
        self.nlp = spacy.load('en_core_web_lg')
        self.aggregation = aggregation  
        self.batch_size = int(np.ceil(len(self.dataset) // cpu_count()))  
        self.parsedDataset = None
        self.is_test = is_test
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

    def split_sentence(self, text):
        """
        Split the text into sentences
        :param text: the text to split
        :return: a list of sentences
        """
        return [self.clean_sentence(s.text) for s in self.nlp(text).sents]  # Splitting into sentences and cleaning

    @staticmethod
    def clean_sentence(s):
        """
        Clean the sentence
        :param s: the sentence to clean
        :return: the cleaned sentence
        """
        return s.strip()  # Removing leading and trailing whitespaces

    def get_first_three_sentences(self, article):
        """
        Get the first three sentences of the article
        :param article: the article
        :return: the first three sentences
        """
        sentences = self.split_sentence(article)[:3]
        return ' '.join(sent for sent in sentences)

    def extract_context(self, article_sentences):
        """
        Extract the context from the article
        :param article_sentences: the article sentences
        :return: the context
        """
        cut_off = math.ceil(len(article_sentences) / 3)
        return ' '.join(sent for sent in article_sentences[:cut_off])

    def process_row(self, row):
        """
        Process a row of the dataset
        :param row: the row to process
        :return: the processed row
        """
        article, summary = row
        article_sentences = self.split_sentence(article)
        context = self.extract_context(article_sentences)

        if self.is_test:  
            return {"sentences": article_sentences, "context": context, "highlights": summary}  # No need for labels
        
        # Splitting the summary into single highlights
        highlights = summary.split("\n")  
        rouge_labels = compute_rouges(article_sentences, highlights, aggregation=self.aggregation)
        similarity_labels = compute_similarities(article_sentences, highlights,
                                                 aggregation=self.aggregation,
                                                 similarity_model=self.similarity_model)

        return {"sentences": article_sentences, "context": context, "highlights": summary,
                "similarity_labels": similarity_labels, "rouge_labels": rouge_labels}

    def process_batch(self, batch):
        """
        Process a batch of the dataset
        :param batch: the batch to process
        :return: the processed batch
        """
        return [self.process_row(row) for row in batch]

    def __getitem__(self, idx):
        """
        Get an item of the dataset
        :param idx: the index of the item
        :return: the item
        """
        if self.parsedDataset is None:
            return {}
        return self.parsedDataset[idx]

    def __call__(self):
        """
        Parse the dataset
        :return: the parsed dataset
        """
        articles = self.dataset["article"]
        summaries = self.dataset["highlights"]
        processed_data = self.process_batch(zip(articles, summaries))
        sentences_list = [item for sublist in processed_data for item in sublist["sentences"]]
        context_list = [context for sublist in processed_data for context in
                        [sublist["context"]] * len(sublist["sentences"])]
        highlights_list = [highlights for sublist in processed_data for highlights in
                           [sublist["highlights"]] * len(sublist["sentences"])]
        
        if self.is_test:
            self.parsedDataset = Dataset.from_dict(
                {"sentence": sentences_list, "context": context_list, "highlights": highlights_list})
        else:
            rouge_list = [label for sublist in processed_data for label in sublist["rouge_labels"]]
            similiarity_list = [label for sublist in processed_data for label in sublist["similarity_labels"]]
            self.parsedDataset = Dataset.from_dict(
                {"sentence": sentences_list, "context": context_list, "highlights": highlights_list,
                 "rouge": rouge_list, "similarity": similiarity_list})
        return self.parsedDataset