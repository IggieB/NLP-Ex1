import spacy
import math
from datasets import load_dataset


def import_train_data():
    """
    # import the training data mentioned in the exercise file
    :return: a dataset of type "datasets.arrow_dataset.Dataset"
    """
    text_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    return text_dataset


def tokenize_dataset(dataset: dict) -> None:
    """
    # within the dataset (dict -> key = 'text', value = list with all lines)
    go over every line (every element in the list) and tokenize it using Spacy's nlp
    :param dataset:
    :return: nothing. Changes the existing dataset.
    """
    nlp = spacy.load("en_core_web_sm")
    for key in dataset:
        dataset[key] = list(map(nlp, dataset[key]))


def unigram_word_dict(dataset: dict) -> dict:
    """
    this function filters numbers abd punctuation, and creates a word dictionary for the
    unigram model
    :param dataset:
    :return: a dictionary in which each key is the lemma form of a word in the dataset,
    and the value is a list:
    list[0] = counts number of the word in the dataset.
    list[1] = probability of the word in the dataset (word counts/all counts)
    """
    word_freq_dict = {}  # the new dict
    for doc in dataset['text']:
        for word in doc:
            if word.is_alpha and word.lemma_ in word_freq_dict:
                #  if the word already exists in the dict
                word_freq_dict[word.lemma_] += 1
            elif word.is_alpha:
                #  if the word is not yet in the dict
                word_freq_dict[word.lemma_] = 1
    # a variable of all word counts in the text
    all_count_sum = sum(word_freq_dict.values())
    for word, count_list in word_freq_dict.items():
        # calculating the probability in log space
        probability = math.log(count_list/all_count_sum)
        # converting the dict value to a list with 2 elements
        word_freq_dict[word] = [count_list]
        word_freq_dict[word].append(probability)
    return word_freq_dict


def bigram_clean_and_add_start(dataset: dict) -> dict:
    """
    this function takes the original dataset, cleans all punctuation and numbers,
    adds start in the beginning of a new doc (line) and adds the words in their
    lemma form.
    :param dataset: the tokenized dataset of documents (Spacy doc objects)
    :return: a new clean dict of lemmas with a start-of-line marker
    """
    # new dataset
    clean_dataset = {'text': []}
    start_symbol = "<START> "
    for doc in dataset['text']:
        new_doc = ""
        for word in doc:
            # clean punctuation/numbers and add start at the beginning of a new doc
            if word.is_alpha and len(new_doc) == 0:
                new_doc += start_symbol + word.lemma_ + " "
            # clean punctuation/numbers only
            elif word.is_alpha:
                new_doc += word.lemma_ + " "
        # avoid adding empty strings
        if len(new_doc) > 0:
            clean_dataset['text'].append(new_doc)
    return clean_dataset


def calculate_bigram_probabilities(dataset: dict) -> dict:
    """
    this function calculates the probability for all possible pairs of words within
    the docs of the given dataset
    :param dataset: the dataset post cleaning and with added START markers
    :return: a dictionary in which each key is a possible pair (lemma form) and the
    value is a list:
    list[0] = counts number of the pair in the dataset.
    list[1] = probability of the pair in the dataset (pair counts/all counts)
    """
    pair_freq_dict = {}
    prev_word = ""
    for doc in dataset['text']:
        pair = ""
        for word in doc.split(" "):
            if len(prev_word) == 0 and len(word):
                prev_word = word
                pair = prev_word + " "
            else:
                pair += word
                if pair in pair_freq_dict:
                    pair_freq_dict[pair] += 1
                    pair = ""
                    prev_word = ""
                else:
                    pair_freq_dict[pair] = 1
                    pair = ""
                    prev_word = ""
    return pair_freq_dict



if __name__ == '__main__':
    sample_size = 20  # temp value for time management
    sentences_dataset = import_train_data()  # data type here is "datasets.arrow_dataset.Dataset"
    temp_dataset = sentences_dataset[:sample_size]  # slicing changes data type to dict
    # key is 'text', value is a list the length of the slicing specified
    tokenize_dataset(temp_dataset)
    # unigram_count_dict = unigram_word_dict(temp_dataset)  # DONE FOR NOW
    bigram_dataset = bigram_clean_and_add_start(temp_dataset)
    print(bigram_dataset)
    count_pairs = calculate_bigram_probabilities(bigram_dataset)
    print(count_pairs)
