import spacy
from datasets import load_dataset
nlp = spacy.load("en_core_web_sm")


def import_train_data():
    text_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    return text_dataset


def add_lemmas_column(example):
    new_column = nlp(example["text"])


def convert_to_lemmas(txt_to_convert):
    # convert all words to their base form to neutralize inflections (their lemmas)
    pass


def clean_punctuation_and_nums(txt_to_clean):
    # clean all non-characters from the text
    pass


def temp:
    pass


def temp_function(example):
    print(type(example["text"]))
    example["text"] = nlp(str(example["text"]))
    print(type(example["text"]))


if __name__ == '__main__':
    sentences_dataset = import_train_data()
    print(sentences_dataset)
    docs_dataset = sentences_dataset.map(temp_function, batched=True)
    print(docs_dataset)
    print(temp_function(docs_dataset[3]))
    # doc = nlp(sentences_dataset[3]["text"])
    # for token in doc:
    #     print(token, token.lemma_)
    # docs_dataset = sentences_dataset.map(convert_to_doc, batched=True)
