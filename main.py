import spacy
import math
from datasets import load_dataset
from collections import Counter
from operator import itemgetter
from typing import Optional


def import_train_data():
    """
    # import the training data mentioned in the exercise file
    :return: a dataset of type "datasets.arrow_dataset.Dataset"
    """
    text_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    return text_dataset


def tokenize_dataset(dataset: list[str]) -> list:
    """
    tokenize every string element in a list using Spacy's nlp
    :param dataset: A list of strings
    :return: A list of documents (each containing tokens)
    """
    string_tokens = []
    nlp = spacy.load("en_core_web_sm")
    for string in dataset:
        string = nlp(string)
        string_tokens.append(string)
    return string_tokens


def filter_punctuation_numbers_add_start(list_of_tokens: list) -> list:
    filtered_documents = []
    for doc in list_of_tokens:
        doc_list = [word.lemma_ for word in doc if word.is_alpha]
        doc_string = " ".join(doc_list)
        if doc_string:
            filtered_documents.append("<START> " + doc_string)
    return filtered_documents
# TODO: add comments


def unigram_probability_dict(dataset: list) -> dict:
    all_words_list = [word for doc in dataset for word in doc.split()]
    word_freq_dict = dict(Counter(all_words_list))  # the new dict
    del word_freq_dict['<START>']
    # a variable of all word counts in the text
    all_count_sum = sum(word_freq_dict.values())
    for word, count in word_freq_dict.items():
        # calculating the probability
        probability = count/all_count_sum
        word_freq_dict[word] = probability
    sorted_unigrams = {k: v for k, v in sorted(word_freq_dict.items(), key=itemgetter(1), reverse=True)}
    return sorted_unigrams
# TODO: add comments


def bigram_probability_dict(dataset: list) -> dict:
    all_bigrams_list = [b for l in dataset for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
    bigram_freq_dict = dict(Counter(all_bigrams_list))
    # start_bigrams = {k: v for k, v in bigram_freq_dict.items() if k[0] == '<START>'}
    all_count_sum = sum(bigram_freq_dict.values())
    for pair, count in bigram_freq_dict.items():
        # calculating the probability
        probability = count / all_count_sum
        bigram_freq_dict[pair] = probability
    sorted_bigrams = {k: v for k, v in sorted(bigram_freq_dict.items(), key=itemgetter(1), reverse=True)}
    return sorted_bigrams
# TODO: add comments


def calc_bigram_transition(bigrams_dict: dict, last_word: str) -> Optional[tuple]:
    potential_bigrams = {k: v for k, v in bigrams_dict.items() if k[0] == last_word}
    if not len(potential_bigrams):
        return None
    sorted_potential_bigrams = {k: v for k, v in sorted(potential_bigrams.items(), key=itemgetter(1), reverse=True)}
    next_word = list(sorted_potential_bigrams.items())[0]
    return next_word
# TODO: add comments


def complete_sentence(sentence: str, bigrams_dict: dict) -> str:
    last_word = sentence.split(" ")[-1]
    next_word_calc = calc_bigram_transition(bigrams_dict, last_word)
    if not next_word_calc:
        return "Cannot predict the next word :("
    next_word = next_word_calc[0][1]
    return sentence + " " + next_word
# TODO: add comments


def compute_sentence_bigram_probability(sentence: str, bigrams_dataset: dict) -> float:
    sentence_probability = 0
    sentence_bigrams = [b for  b in zip(sentence.split(" ")[:-1], sentence.split(" ")[1:])]
    for bigram in sentence_bigrams:
        last_word = bigram[0]
        bigram_calc = calc_bigram_transition(bigrams_dataset, last_word)
        if not bigram_calc:
            return 0
        bigram_probability = bigram_calc[1]
        sentence_probability += math.log(bigram_probability)
    return sentence_probability
# TODO: add comments


def compute_bigram_perplexity(sentences: list, bigrams_dataset: dict) -> float:
    overall_sentences_probability = 0
    bigrams_number = 0
    for sentence in sentences:
        sentence_bigrams = len([b for  b in zip(sentence.split(" ")[:-1], sentence.split(" ")[1:])])
        bigrams_number += sentence_bigrams
        sentence_probability = compute_sentence_bigram_probability(sentence, bigrams_dataset)
        if sentence_probability:
            overall_sentences_probability += sentence_probability
        else:
            overall_sentences_probability += float('-inf')
    return math.pow(2, -(overall_sentences_probability/bigrams_number))
# TODO: add comments


def compute_interpolation_transition_probability(sentence: str, unigram_dataset: dict,
                                               start_dataset: dict, bigrams_dataset: dict) -> float:
    sentence_probability = 0
    sentence_unigrams = sentence.split(" ")
    sentence_list = ["<START> " + sentence]
    sentence_bigrams = [b for l in sentence_list for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
    # check if first word is a doc opener:
    try:
        transition_probability = 0
        transition_probability += (1 / 3) * math.exp(unigram_dataset[sentence_unigrams[0]][1])
        transition_probability += (2 / 3) * math.exp(start_dataset[sentence_bigrams[0]][1])
        sentence_probability += math.log(transition_probability)
    except:
        pass
    for i in range(len(sentence_unigrams)-1):
        try:
            transition_probability += (1/3) * math.exp(unigram_dataset[sentence_unigrams[i]][1])
            transition_probability += (2/3) * math.exp()
        except:
            pass
        print(sentence_unigrams[i], sentence_bigrams[i])
    # TODO: exp the probabilities to add them and then re-log!


if __name__ == '__main__':
    # ############ Prep #############
    sentences_dataset = import_train_data()  # data type here is "datasets.arrow_dataset.Dataset"
    dict_dataset = sentences_dataset[0:]  # change to dict key is 'text', value is a strings list
    tokens_list = tokenize_dataset(dict_dataset['text'])
    filtered_tokens = filter_punctuation_numbers_add_start(tokens_list)
    # ############ Question 1 #############
    unigrams_probability_dict = unigram_probability_dict(filtered_tokens)
    bigrams_probability_dict = bigram_probability_dict(filtered_tokens)
    # ############ Question 2 #############
    # sentence = "I have a house in"
    # tokenized_sentence = tokenize_dataset([sentence])
    # filtered_sentence = filter_punctuation_numbers_add_start(tokenized_sentence)[0]
    # print(complete_sentence(filtered_sentence, bigrams_probability_dict))
    # ############ Question 3 a #############
    test_set = ["Brad Pitt was born in Oklahoma", "The actor was born in USA"]
    tokenized_test_set = tokenize_dataset(test_set)
    filtered_test_set = filter_punctuation_numbers_add_start(tokenized_test_set)
    # for sentence in filtered_test_set:
    #     print(compute_sentence_bigram_probability(sentence, bigrams_probability_dict))
    # ############ Question 3 b #############
    # bigram_perplexity = compute_bigram_perplexity(filtered_test_set, bigrams_probability_dict)
    # print(bigram_perplexity)
    # ############ Question 4 #############
    # ############ linear interpolation smoothing probability
    test_sentence = ["Valkyria Chronicles III"]
    # test_sentence_lemma = convert_test_set(test_sentence)
    # compute_interpolation_transition_probability(test_sentence_lemma[0], unigrams_dataset,
    #                                              start_probability_dict, bigrams_probability_dict)
