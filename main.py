import spacy
import math
from datasets import load_dataset
from collections import Counter
from operator import itemgetter


def import_train_data():
    """
    Import the training data mentioned in the exercise file.
    :return: a dataset of type "datasets.arrow_dataset.Dataset"
    """
    text_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    return text_dataset


def tokenize_dataset(dataset: list[str]) -> list:
    """
    Tokenize every string element in a list using Spacy's nlp.
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
    """
    Filter all non-alphabetic characters/words from a set of documents.
    :param list_of_tokens: original dataset tokenized
    :return: a list of filtered strings
    """
    filtered_documents = []
    for doc in list_of_tokens:
        doc_list = [word.lemma_ for word in doc if word.is_alpha]
        doc_string = " ".join(doc_list)
        if doc_string:
            filtered_documents.append("<START> " + doc_string)
    return filtered_documents


def unigram_probability_dict(dataset: list) -> dict:
    """
    Creates a dictionary of words and their probability in the dataset.
    :param dataset: filtered dataset
    :return: a dict, key is the word, value is its probability
    """
    all_words_list = [word for doc in dataset for word in doc.split()]
    word_freq_dict = dict(Counter(all_words_list))  # the new dict
    del word_freq_dict['<START>']  # ignore start-of-sentence marker
    all_count_sum = sum(word_freq_dict.values())  # a variable of all word counts in the text
    for word, count in word_freq_dict.items():
        probability = count/all_count_sum  # calculating the probability
        word_freq_dict[word] = probability
    sorted_unigrams = {k: v for k, v in sorted(word_freq_dict.items(), key=itemgetter(1), reverse=True)}
    return sorted_unigrams


def bigram_probability_dict(dataset: list) -> dict:
    """
    Creates a dictionary of bigrams and their probability in the dataset.
    :param dataset: filtered dataset
    :return: a dict, key is the bigram as a tuple, value is its probability
    """
    all_bigrams_list = [b for l in dataset for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
    bigram_freq_dict = dict(Counter(all_bigrams_list))
    all_count_sum = sum(bigram_freq_dict.values())  # a variable of all bigram counts in the text
    for pair, count in bigram_freq_dict.items():
        probability = count / all_count_sum  # calculating the probability
        bigram_freq_dict[pair] = probability
    sorted_bigrams = {k: v for k, v in sorted(bigram_freq_dict.items(), key=itemgetter(1), reverse=True)}
    return sorted_bigrams


def calc_unigram_transition(unigrams_dict: dict, cur_word: str) -> float:
    """
    Using the unigram model, calculate the probability of a given current word.
    :param unigrams_dict: the unigrams probability dictionary
    :param cur_word: the current word of a given sentence
    :return: the probability of the given current word
    """
    try:
        return unigrams_dict[cur_word]
    except KeyError:  # if word does not appear in the training data
        return 0


def calc_bigram_transition(bigrams_dict: dict, last_word: str):
    """
    Using the bigram model, for a given word, predict the next word and its probability.
    :param bigrams_dict: the bigrams probability dictionary
    :param last_word: the last word of a given sentence
    :return: 0 if the word does not appear in the training dataset, tuple of the next word
    and its probability otherwise
    """
    potential_bigrams = {k: v for k, v in bigrams_dict.items() if k[0] == last_word}
    if not len(potential_bigrams):
        return 0
    sorted_potential_bigrams = {k: v for k, v in sorted(potential_bigrams.items(), key=itemgetter(1), reverse=True)}
    next_word = list(sorted_potential_bigrams.items())[0]
    return next_word


def complete_sentence(sentence: str, bigrams_dict: dict) -> str:
    """
    Using the bigram model, predict the next word for a given sentence.
    :param sentence: the given sentence
    :param bigrams_dict: the bigrams probability dictionary
    :return: the completed sentence with the additional word
    """
    last_word = sentence.split(" ")[-1]
    next_word_calc = calc_bigram_transition(bigrams_dict, last_word)
    if not next_word_calc:  # the word does not appear in the training data
        return "Cannot predict the next word :("
    next_word = next_word_calc[0][1]
    return sentence + " " + next_word


def compute_sentence_bigram_probability(sentence: str, bigrams_dataset: dict) -> float:
    """
    Compute the probability of a given sentence using the bigram model.
    :param sentence: the given sentence
    :param bigrams_dataset: the bigrams probability dictionary
    :return: the probability of the sentence in log
    """
    sentence_probability = 0
    sentence_bigrams = [b for b in zip(sentence.split(" ")[:-1], sentence.split(" ")[1:])]
    for bigram in sentence_bigrams:
        last_word = bigram[0]
        bigram_calc = calc_bigram_transition(bigrams_dataset, last_word)
        if not bigram_calc:  # if on of the bigrams does not appear in the training dataset
            return 0
        bigram_probability = bigram_calc[1]
        sentence_probability += math.log(bigram_probability)
    return sentence_probability


def compute_interpolation_probability(sentence: str, unigrams_dataset: dict, bigrams_dataset: dict) -> float:
    """
    Using linear interpolation, compute the probability of a given sentence.
    :param sentence: the given sentence
    :param unigrams_dataset: the unigrams probability dictionary
    :param bigrams_dataset: the bigrams probability dictionary
    :return: the probability of the sentence in log
    """
    sentence_probability = 0
    sentence_bigrams = [b for b in zip(sentence.split(" ")[:-1], sentence.split(" ")[1:])]
    sentence_unigrams = sentence.split()[1:]
    for i in range(len(sentence_unigrams)):
        word_uni_prob = calc_unigram_transition(unigrams_dataset, sentence_unigrams[i])
        word_bi_prob = calc_bigram_transition(bigrams_dataset, sentence_bigrams[i][1])
        if type(word_bi_prob) is tuple:  # the bigram returns a tuple if word in dict, 0 otherwise
            word_bi_prob = word_bi_prob[1]
        weighted_prob = (((1/3) * word_uni_prob) + ((2/3) * word_bi_prob))
        if not weighted_prob:  # if the term is unfamiliar to the unigram and bigram models
            sentence_probability += float('-inf')
        if weighted_prob:
            sentence_probability += math.log(weighted_prob)
    return sentence_probability


def compute_model_perplexity(model: str, sentences: list, unigrams_dataset: dict, bigrams_dataset: dict) -> float:
    """
    Compute the perplexity of the bigram or the linear model
    :param model: model type (Bigram or linear)
    :param sentences: the sentences test set
    :param unigrams_dataset: the unigrams probability dictionary
    :param bigrams_dataset: the bigrams probability dictionary
    :return: the model's perplexity
    """
    overall_sentences_probability = 0
    bigrams_number = 0
    for sentence in sentences:
        sentence_bigrams = len([b for b in zip(sentence.split(" ")[:-1], sentence.split(" ")[1:])])
        bigrams_number += sentence_bigrams
        if model == "Bigram": sentence_probability = compute_sentence_bigram_probability(sentence, bigrams_dataset)
        if model == "Linear": sentence_probability = compute_interpolation_probability(sentence, unigrams_dataset,
                                                                                       bigrams_dataset)
        if sentence_probability: overall_sentences_probability += sentence_probability
        else: overall_sentences_probability += float('-inf')  # if sentence probability is 0
    return math.pow(2, -(overall_sentences_probability/bigrams_number))


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
    sentence = "I have a house in"
    tokenized_sentence = tokenize_dataset([sentence])
    filtered_sentence = filter_punctuation_numbers_add_start(tokenized_sentence)[0]
    print("The completed sentence is: ", complete_sentence(filtered_sentence, bigrams_probability_dict))
    # ############ Question 3 a #############
    test_set = ["Brad Pitt was born in Oklahoma", "The actor was born in USA"]
    tokenized_test_set = tokenize_dataset(test_set)
    filtered_test_set = filter_punctuation_numbers_add_start(tokenized_test_set)
    print("Bigram probability")
    for sentence in filtered_test_set:
        print("The bigram probability for ", sentence, " is:")
        print(compute_sentence_bigram_probability(sentence, bigrams_probability_dict))
    # ############ Question 3 b #############
    bigram_perplexity = compute_model_perplexity("Bigram", filtered_test_set, unigrams_probability_dict,
                                                 bigrams_probability_dict)
    print("The perplexity of the bigram model is: ", bigram_perplexity)
    # ############ Question 4 #############
    # ############ linear interpolation probability #############
    print("Linear probability")
    for sentence in filtered_test_set:
        print("The linear probability for ", sentence, " is:")
        print(compute_interpolation_probability(sentence, unigrams_probability_dict, bigrams_probability_dict))
    # ############ linear interpolation perplexity #############
    linear_perplexity = compute_model_perplexity("Linear", filtered_test_set, unigrams_probability_dict,
                                                 bigrams_probability_dict)
    print("The perplexity of the Linear model is: ", linear_perplexity)
