import spacy
from datasets import Dataset
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils import compute_labels
import math
from sentence_transformers import SentenceTransformer

class DataParser:

    def __init__(self, dataset=None, aggregation='max', is_test=False):
        self.dataset = dataset
        # Loading the spaCy model to split into sentences
        self.nlp = spacy.load('en_core_web_lg')
        self.aggregation = aggregation  # Aggregation method for ROUGE scores
        self.batch_size = int(np.ceil(len(self.dataset) // cpu_count()))  # Computing optimal the batch size
        self.parsedDataset = None
        self.is_test = is_test
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")        


    def split_sentence(self, text):
        return [self.clean_sentence(s.text) for s in self.nlp(text).sents]  # Splitting into sentences and cleaning

    @staticmethod
    def clean_sentence(s):
        return s.strip()  # Removing leading and trailing whitespaces

    def get_first_three_sentences(self, article):
        sentences = self.split_sentence(article)[:3]
        return ' '.join(sent for sent in sentences)

    def extract_context(self, article_sentences):
        # Compute the index for the first section
        cut_off = math.ceil(len(article_sentences) / 3)
        return ' '.join(sent for sent in article_sentences[:cut_off])

    def process_row(self, row):
        # Unpack the parameter tuple
        article, summary = row
        article_sentences = self.split_sentence(article)
        context = self.extract_context(article_sentences)
        rouge_labels, similarity_labels =list(),list()

        if self.is_test:  # Optimization: skip rouge computation if test split for speed-up
            return {"sentences": article_sentences, "context": context, "highlights": summary}  # No need for labels

        highlights = summary.split("\n")  # Splitting the summary into single highlights
        for sentence in article_sentences:  # Computing ROUGE score (label) for each sentence
            rouge_score, similarity_score = compute_labels(sentence, highlights, aggregation=self.aggregation, is_test=self.is_test, \
                                                           similarity_model = self.similarity_model)
            rouge_labels.append(rouge_score)
            similarity_labels.append(similarity_score)
        return {"sentences": article_sentences, "context": context, "rouge_labels": rouge_labels, "similarity_labels": similarity_labels, "highlights": summary}  

    def process_batch(self, batch):
        return [self.process_row(row) for row in batch]

    def __getitem__(self, idx):
        if self.parsedDataset is None:
            return {}
        return self.parsedDataset[idx]

    def __call__(self):
        articles = self.dataset["article"]
        summaries = self.dataset["highlights"]

        processed_data = self.process_batch(zip(articles,summaries))

        """
        # Create a Pool object
        pool = Pool(1)
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
        """
        sentences_list = [item for sublist in processed_data for item in sublist["sentences"]]
        context_list = [context for sublist in processed_data for context in
                        [sublist["context"]] * len(sublist["sentences"])]
        highlights_list = [highlights for sublist in processed_data for highlights in
                [sublist["highlights"]] * len(sublist["sentences"])]
        # If test split, retrieve highlights, else retrieve label
        if self.is_test:
            self.parsedDataset = Dataset.from_dict(
                {"sentence": sentences_list, "context": context_list, "highlights": highlights_list})
        else:
            rouge_list = [label for sublist in processed_data for label in sublist["rouge_labels"]]
            similiarity_list = [label for sublist in processed_data for label in sublist["similarity_labels"]]
            self.parsedDataset = Dataset.from_dict(
                {"sentence": sentences_list, "context": context_list, "rouge": rouge_list, "similarity": similiarity_list, "highlights": highlights_list})   #TO DOUBLE CHECK
        return self.parsedDataset