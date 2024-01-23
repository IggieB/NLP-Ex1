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
    sentence_probability = 0
    sentence_list = [sentence]
    # create a list of all bigrams in the given sentence
    bigrams_list = [b for l in sentence_list for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
    # go over each bigram, if it exists in the training data, add its probability
    # to the overall sentence probability
    for bigram in bigrams_list[:-1]:
        # if one of the bigrams is unknown stop and return zero
        if " ".join(bigram) not in bigrams_dataset:
            sentence_probability = float('-inf')
            return sentence_probability
        # otherwise keep summing the sentence's bigrams probabilities to get the
        # probability of the whole sentence
        sentence_probability += bigrams_dataset[" ".join(bigram)][1]
    return sentence_probability


def compute_bigram_perplexity(sentences: list, bigrams_dataset: dict) -> float:
    overall_sentences_probability = 0
    bigrams_number = 0
    for sentence in sentences:
        # calculate all bigrams in the sentences of the test set
        bigrams_number += len(sentence.split(" ")) - 1
        # handle cases of probability = 0
        if compute_sentence_bigram_probability(sentence, bigrams_dataset) == 0:
            overall_sentences_probability -= float('-inf')
        else:
            overall_sentences_probability -= compute_sentence_bigram_probability(sentence, bigrams_dataset)
    return math.pow(2, overall_sentences_probability / bigrams_number)


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