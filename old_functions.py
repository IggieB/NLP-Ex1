import math

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
    this function computes a sentence's probability using linear interpolation
    :param sentence: the sentence for which the probability will be computed
    :param unigram_dataset: a dataset of all unigrams' probabilities based on the
    training dataset
    :param start_dataset: a dataset of probabilities of word as sentence openers
    :param bigrams_dataset: a dataset of all bigrams' probabilities based on the
    training dataset
    :return: a value of the sentence's probability
    """
    weights = [1/3, 2/3]
    unigram_probability = weights[0] * math.exp(compute_sentence_unigram_probability(sentence, unigram_dataset))
    bigram_probability = weights[1] * math.exp(compute_sentence_bigram_probability(sentence, start_dataset, bigrams_dataset))
    sentence_interpolation_probability = unigram_probability + bigram_probability
    if sentence_interpolation_probability == 0:
        return float('-inf')
    return math.log(sentence_interpolation_probability)


def compute_unigram_perplexity(sentences: list, unigram_dataset: dict) -> float:
    """
    this function takes a given test set of 1 or more sentences and uses it to
    calculate the perplexity of the unigram model
    :param sentences: a test set including 1 or more sentences
    :param unigram_dataset: a dataset of all unigrams' probabilities based on the
    training dataset
    :return: a value of the model's perplexity using the test set
    """
    overall_sentences_probability = 0
    unigrams_number = 0
    for sentence in sentences:
        # get the overall number of unigrams
        unigrams_number += len(sentence.split(" "))
        # calculate the probability of each sentence
        sentence_probability = compute_sentence_unigram_probability(sentence, unigram_dataset)
        overall_sentences_probability -= sentence_probability
    # calculate the perplexity
    return math.pow(2, overall_sentences_probability / unigrams_number)