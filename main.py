import spacy
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


def filter_non_alphabetic_characters(dataset: dict) -> dict:
    """
    filter out all tokens that are recognized as none alphabetic (numbers, punctuation, etc.).
    :param dataset:
    :return: a new filtered dictionary
    """
    new_dict = {'text': []}
    for key in dataset:
        # TODO: shorten this part
        for doc in dataset[key]:
            new_doc = ""
            for token in doc:
                if token.is_alpha:
                    new_doc = new_doc + token.text_with_ws
            new_dict['text'].append(new_doc)
    return new_dict


if __name__ == '__main__':
    sample_size = 20  # temp value for time management
    sentences_dataset = import_train_data()  # data type here is "datasets.arrow_dataset.Dataset"
    temp_dataset = sentences_dataset[:sample_size]  # slicing changes data type to dict
    # key is 'text', value is a list the length of the slicing specified
    tokenize_dataset(temp_dataset)
    clean_dict = filter_non_alphabetic_characters(temp_dataset)
    tokenize_dataset(clean_dict)
    print(clean_dict)
