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


def count_all_bigrams(dataset: dict) -> dict:
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
            # if this is a start of a new pair and the current word in not white space
            if len(prev_word) == 0 and len(word) > 0:
                # fill markers for previous word and start the pair
                prev_word = word
                pair = prev_word + " "
            else:
                # if there is already a previous word complete the pair
                pair += word
                # if the pair is already a key in the dict add to counter
                # and initialize the markers for the next pair
                if pair in pair_freq_dict:
                    pair_freq_dict[pair] += 1
                    pair = word + " "
                    prev_word = word
                else:
                    # if the pair is not already in the dict verify, and it doesn't
                    # contain whitespace, add it to the dict and initialize the
                    # markers for the next pair
                    if len(pair.split(" ")[0]) >= 1 and len(pair.split(" ")[1]) >= 1:
                        pair_freq_dict[pair] = 1
                    pair = word + " "
                    prev_word = word
    return pair_freq_dict


def calculate_all_bigrams_probabilities(dataset: dict) -> tuple[dict, dict]:
    """
    this function calculates the probability of all bigrams in the dataset and the odds
    of a word opening a sentence (being a part of a <START> bigram)
    :param dataset: a dict containing all bigrams found in the dataset and their counts.
    :return: 2 dicts
    the first one is regular bigrams, the key will be the pair and the value a list with:
    list[0] - number of counts, list[1] - probability in log space
    the second dict has the same structure, but for <START> bigrams (that is, the odds
    of a word opening a sentence)
    """
    # new dict for regular bigrams and for beginning of sentence
    bigrams_without_start = {}
    bigrams_with_start = {}
    # summing all counts of regular bigrams and beginning of sentence (<START> bigrams)
    all_pairs_count = sum(count for pair, count in dataset.items() if pair[0] != "<")
    start_pairs_count = sum(count for pair, count in dataset.items() if pair[0] == "<")
    # dividing the bigrams to the 2 dicts and calculating their probabilities
    for pair, count in dataset.items():
        if pair[0] != "<":
            bigrams_without_start[pair] = [count, math.log(count/all_pairs_count)]
        else:
            bigrams_with_start[pair] = [count, math.log(count/start_pairs_count)]
    return bigrams_without_start, bigrams_with_start


def complete_sentence(sentence: str, dataset: dict) -> str:
    """
    using the bigram this function completes the given sentence, adding the most
    probable next word.
    :param sentence: a string
    :param dataset: a dictionary with the probabilities of all bigrams calculated
    from the training data
    :return: the completed sentence with the most probable word according to
    the training dataset
    """
    last_word = sentence.split(" ")[-1]
    sorted_dataset = sorted(dataset.items(), key=lambda x: x[1][0], reverse=True)
    # returns the dict as a list in which each element is:
    # ('word1 word2', [counts, probability])
    for element in sorted_dataset:
        # The first element to include the relevant word will have the highest
        # probability after sorting, and therefore will be returned as the next
        # predicted word
        if last_word == element[0].split(" ")[0]:
            next_word = element[0].split(" ")[1]
            return sentence + " " + next_word
    # in case the sentence's last word does not exist in the training set and
    # therefore the next word cannot be predicted
    return "Cannot predict the next word :("


def compute_sentence_bigram_probability(sentence: str, start_dataset: dict, bigrams_dataset: dict) -> float:
    """
    this function computes the probability of a given sentence by breaking it into
    bigrams, checking their probability using the bigram model and returning the
    overall result
    :param sentence: the sentence for which the probability will be computed
    :param start_dataset: a dataset of probabilities of word as sentence openers
    :param bigrams_dataset: a dataset of all bigrams' probabilities based on the
    training dataset
    :return: a value of the probability of the whole sentence
    """
    sentence_w_start = ["<START> " + sentence]
    sentence_probability = 0
    # create a list of all bigrams in the given sentence
    bigrams_list = [bigram for sentence in sentence_w_start for bigram in
               zip(sentence.split(" ")[:-1], sentence.split(" ")[1:])]
    # if the first word does not exist as a sentence opener stop and return 0
    if " ".join(bigrams_list[0]) not in start_dataset.keys():
        return float('-inf')
    # otherwise add its probability as a first word to the overall sentence probability
    sentence_probability += start_dataset[" ".join(bigrams_list[0])][1]
    # go over each bigram, if it exists in the training data, add its probability
    # to the overall sentence probability
    for bigram in bigrams_list[1:]:
        # if one of the bigrams is unknown stop and return zero
        if " ".join(bigram) not in bigrams_dataset:
            sentence_probability = float('-inf')
            return sentence_probability
        # otherwise keep summing the sentence's bigrams probabilities to get the
        # probability of the whole sentence
        sentence_probability += bigrams_dataset[" ".join(bigram)][1]
    return sentence_probability


def compute_bigram_perplexity(sentences: list, start_dataset: dict, bigrams_dataset: dict) -> float:
    """
    this function takes a given test set of 1 or more sentences and uses it to
    calclate the perplexity of the bigram model
    :param sentences: a test set including 1 or more sentences
    :param start_dataset: a dataset of probabilities of word as sentence openers
    :param bigrams_dataset: a dataset of all bigrams' probabilities based on the
    training dataset
    :return: a value of the model's perplexity using the test set
    """
    overall_sentences_probability = 0
    bigrams_number = 0
    for sentence in sentences:
        # calculate all bigrams in the sentences of the test set
        bigrams_number += len(sentence.split(" ")) - 1
        # handle cases of probability = 0
        if compute_sentence_bigram_probability(sentence, start_dataset, bigrams_dataset) == 0:
            overall_sentences_probability -= float('-inf')
        else:
            overall_sentences_probability -= compute_sentence_bigram_probability(
                sentence, start_dataset, bigrams_dataset)
    return math.pow(2, overall_sentences_probability / bigrams_number)


def compute_sentence_unigram_probability(sentence: str, unigram_dataset: dict) -> float:
    """
    this function computes the probability of a given sentence using the unigram model
    :param sentence: the sentence for which the probability will be computed
    :param unigram_dataset: a dataset of all unigrams' probabilities based on the
    training dataset
    :return: a value of the sentence's probability
    """
    sentence_probability = 0
    for word in sentence.split(" "):
        try:
            sentence_probability += unigram_dataset[word][1]
        except:
            sentence_probability += float('-inf')
    return sentence_probability


def compute_sentence_interpolation_probability(sentence: str, unigram_dataset: dict,
                                               start_dataset: dict, bigrams_dataset: dict) -> float:
    """

    :param sentence:
    :param unigram_dataset:
    :param start_dataset:
    :param bigrams_dataset:
    :return:
    """
    sentence_interpolation_probability = 0
    weights = [1/3, 2/3]
    unigram_probability = weights[0] * compute_sentence_unigram_probability(sentence, unigram_dataset)
    bigram_probability = weights[1] * compute_sentence_bigram_probability(sentence, start_dataset, bigrams_dataset)
    sentence_interpolation_probability = unigram_probability + bigram_probability
    return sentence_interpolation_probability



if __name__ == '__main__':
    # ############ Prep #############
    sample_size = 20  # temp value for time management
    sentences_dataset = import_train_data()  # data type here is "datasets.arrow_dataset.Dataset"
    temp_dataset = sentences_dataset[:sample_size]  # slicing changes data type to dict
    # key is 'text', value is a list the length of the slicing specified
    tokenize_dataset(temp_dataset)
    # ############ Question 1 #############
    unigrams_dataset = unigram_word_dict(temp_dataset)
    bigram_dataset = bigram_clean_and_add_start(temp_dataset)
    count_pairs = count_all_bigrams(bigram_dataset)
    bigrams_probability_dict, start_probability_dict = calculate_all_bigrams_probabilities(count_pairs)
    # ############ Question 2 #############
    # print(complete_sentence("I have a house in", bigrams_probability_dict))
    # ############ Question 3 a #############
    # print(compute_sentence_bigram_probability("Brad Pitt was born in Oklahoma", start_probability_dict,
    #                                    bigrams_probability_dict))
    # print(compute_sentence_bigram_probability("The actor was born in USA", start_probability_dict,
    #                                    bigrams_probability_dict))
    # ############ Question 3 b #############
    # test_set = ["Brad Pitt was born in Oklahoma", "The actor was born in USA"]
    # bigram_perplexity = compute_bigram_perplexity(test_set, start_probability_dict, bigrams_probability_dict)
    # print(bigram_perplexity)
    # ############ Question 4 #############
    # ############ linear interpolation smoothing probability
    test_sentence1 = "Brad Pitt was born in Oklahoma"
    print(compute_sentence_interpolation_probability(test_sentence1, unigrams_dataset, start_probability_dict,
                                               bigrams_probability_dict))
    test_sentence2 = "The actor was born in USA"
    print(compute_sentence_interpolation_probability(test_sentence2, unigrams_dataset, start_probability_dict,
                                                     bigrams_probability_dict))
    # ############ linear interpolation smoothing perplexity


