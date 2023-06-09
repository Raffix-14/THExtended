import re
import spacy
from datasets import Dataset
import pandas as pd
from math import *
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import torch
import time
from tqdm import tqdm
from evaluate import load


def cleaner(dataset, num_samples, seed=42):
    nlp = spacy.load("en_core_web_lg")
    # Shuffle the dataset, so you can be sure you're not selecting the first contiguous data points
    ds = dataset.shuffle(seed=seed)
    # Create the cleaned ds and init the index
    cleaned_ds = []
    i = -1
    # Thresholds
    before_th = 40

    # Start looping on the shuffled ds and collect the good, cleaned samples
    while len(cleaned_ds) < num_samples:

        # Select the data point
        i += 1
        data_point = ds[i]
        article = data_point["article"]
        highlights = nlp(data_point["highlights"])

        cleaned_high = ""
        n_highlight = 0
        # For each hl, check whether to discard it
        for current_h in highlights.sents:
            if not (len(current_h.text.split(" ")) <= 3 and any(char.isalpha() for char in current_h.text)):
                n_highlight += 1
                if cleaned_high != "":
                    cleaned_high += "\n"
                cleaned_high += current_h.text.replace("\n", "").strip()

        if 3 <= n_highlight <= 5:  # If the highlights are more than 3 and less than 5 is good.
            data_point['highlights'] = cleaned_high  # Updating the 'highlights' field of the considered data_point.
        else:
            continue

        # Check if "--" is in the article
        if "--" in article:
            # Check the length of the text before the "--"; if it's less than a threshold, remove it from the article
            text_before, text_after = article.split("--", 1)
            if len(text_before) <= before_th:
                article = text_after
                data_point["article"] = article
        # Now do the same with "(CNN)"
        if "(CNN)" in article:
            text_before, text_after = article.split("(CNN)", 1)
            if len(text_before) <= before_th:
                article = text_after
                data_point["article"] = article

        # ADDITIONAL CLEANING: SOMETIMES THE ARTICLES HAVE A BEGINNING LIKE word . word word . word .
        splits = article.split(".")
        for j, split in enumerate(splits):
            words = [word for word in split.split() if not any(char.isdigit() for char in word)]
            if len(words) <= 2:
                continue
            else:
                article = ".".join(splits[j:]).strip()
                data_point['article'] = article
                break

        ###########################################################################################

        # Finally, filter out articles that are too short
        if len(article) < 300:
            continue

        # Append the data point to the cleaned ds
        cleaned_ds.append(data_point)

    return Dataset.from_pandas(pd.DataFrame(data=cleaned_ds))


class Explorer:

    def __init__(self, dataset):
        self.nlp = spacy.load('en_core_web_lg')
        self.ds = dataset
        self.bertscore = load("bertscore")

    @staticmethod
    def split_sentence_article(nlp, text):
        doc = nlp(text)
        sentences = []
        for sent in doc.sents:
            sentence = sent.text.replace("\n", "")
            sentence = sentence.strip()
            sentences.append(sentence)
        return sentences

    @staticmethod
    def create_sections(sentences):
        # Compute the index ranges for each bucket
        n = len(sentences)
        cut1 = ceil(n / 3)
        cut2 = ceil(cut1 + ((n - cut1) / 2))
        # Create the sections
        section1 = sentences[:cut1]
        section2 = sentences[cut1:cut2]
        section3 = sentences[cut2:]
        return [section1, section2, section3]

    @staticmethod
    def compute_similarity(article, section):
        return self.bertscore.compute(predictions=[section], references=[article], model_type="allenai/longformer-base-4096")

    @staticmethod
    def plot_similarities(similarities):

        # Boxplots
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in similarities.items()]))
        # Melt the DataFrame to a long format
        df = df.melt(var_name='Section', value_name='Similarity')
        # Set the theme
        sns.set_theme(style="whitegrid")
        # Create the boxplots
        plt.figure(figsize=(10, 7))
        sns.boxplot(x='Section', y='Similarity', data=df, palette="Set3")
        plt.title('Similarities by Section', fontsize=20)
        plt.xlabel('Section', fontsize=15)
        plt.ylabel('Similarity', fontsize=15)
        plt.ylim(0, 1)
        plt.show()

        # Heatmap
        # Convert the dictionary to a DataFrame and transpose it
        df = pd.DataFrame(similarities)

        sns.set_theme()

        plt.figure(figsize=(10, 7))
        sns.heatmap(df, annot=False, cmap="YlGnBu")
        plt.title('Similarities by Section and Article', fontsize=20)
        plt.ylabel('Article', fontsize=15)
        plt.xlabel('Section', fontsize=15)
        plt.show()

    def explore(self):
        # For each data point
        start = time.time()
        similarities = defaultdict(list)
        for data_point in tqdm(self.ds, desc=" Iterating cleaned dataset"):
            # Take its article
            article = data_point["article"]
            # Split it in sentences
            sentences = self.split_sentence_article(self.nlp, article)
            # Divide the sentences in 3 buckets (start, middle, finish)
            sections = self.create_sections(sentences)
            # For each section, compute the BertSimilarity with the whole article
            for i, section in enumerate(sections, start=1):
                section_text = ' '.join(section)
                similarity = self.compute_similarity(article, section_text)
                similarities[i].append(similarity['f1'][0])
                # Cleaning up
                del section_text, similarity
                torch.cuda.empty_cache()
                gc.collect()
        end = time.time()
        print(f"Done i {end - start} seconds")
        self.plot_similarities(similarities)