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


def add_start(dataset: dict) -> dict: #TODO: change to bigram dict
    """
    this function add a start-of-sentence symbol at the beginning of each doc
    for the bigram model.
    :param dataset: the tokenized dataset of documents (Spacy doc objects)
    :return: nothing. changes the existing dataset.
    """
    new_dict = {'text': []}
    nlp = spacy.load("en_core_web_sm")
    start_symbol = '<START> '
    for doc in dataset['text']:
        new_doc = start_symbol + str(doc)
        new_dict['text'].append(nlp(new_doc))
    return new_dict


if __name__ == '__main__':
    sample_size = 20  # temp value for time management
    sentences_dataset = import_train_data()  # data type here is "datasets.arrow_dataset.Dataset"
    temp_dataset = sentences_dataset[:sample_size]  # slicing changes data type to dict
    # key is 'text', value is a list the length of the slicing specified
    tokenize_dataset(temp_dataset)
    unigram_count_dict = unigram_word_dict(temp_dataset)
