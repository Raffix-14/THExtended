import spacy
import statistics
from datasets import Dataset, load_from_disk
from rouge import Rouge
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class DataParser:

    def __init__(self, dataset=None, aggregation='max'):
        # If the dataset is a string, it is assumed to be a path to a pickle file.
        if isinstance(dataset, str):
            self.dataset = load_from_disk(dataset)
            self.from_pickle = True
        else:
            self.dataset = dataset
        self.from_pickle = False
        # Loading the spaCy model to split into sentences
        self.nlp = spacy.load('en_core_web_lg')
        self.aggregation = aggregation  # Aggregation method for ROUGE scores
        self.rouge = Rouge()  # Loading the rouge metric
        self.batch_size = int(np.ceil(len(self.dataset) // cpu_count()))  # Computing optimal the batch size
        self.parsedDataset = None

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
        highlights = summary.split("\n")  # Splitting the summary into single highlights
        for sentence in article_sentences:  # Computing ROUGE score (label) for each sentence
            score = self.compute_rouge(sentence, highlights, aggregation=self.aggregation)
            rouges.append(score)
        return {"sentences": article_sentences, "context": context, "labels": rouges}  # Returning the processed row

    def process_batch(self, batch):
        return [self.process_row(row) for row in batch]

    def compute_rouge(self, sentence, references, aggregation='max'):
        # Skip empty sentences or sentences without words
        if not sentence.strip() or not any(char.isalpha() for char in sentence):
            return 0.0

        scores = []
        # Compute ROUGE scores between the sentence and each highlight
        for reference in references:
            try:
                rouge_score = self.rouge.get_scores(sentence, reference)[0]["rouge-1"]['f']
            except:
                rouge_score = 0.0
                print("ERROR")
                print(sentence)
                print(reference)
                print(references)
            scores.append(rouge_score)

        # If no scores were computed, return 0.0
        if not scores or scores is None or len(scores) == 0:
            return 0.0
        # Based on the aggregation parameter select the right score
        if aggregation == 'max':
            rouge_score = max(scores)
        elif aggregation == 'average':
            rouge_score = sum(scores) / len(scores)
        elif aggregation == 'harmonic':
            rouge_score = statistics.harmonic_mean(scores)
        else:
            # If an invalid value is provided for `aggregation`, a `ValueError` is raised.
            raise ValueError(f"Invalid aggregation parameter: {aggregation}")

        return rouge_score

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
        labels_list = [label for sublist in processed_data for label in sublist["labels"]]
        self.parsedDataset = Dataset.from_dict(
            {"sentence": sentences_list, "context": context_list, "label": labels_list})
        return self.parsedDataset
