import spacy
from datasets import load_dataset
nlp = spacy.load("en_core_web_sm")

def import_train_data():
    text_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    return text_dataset

def temp_function(example):
    print(type(example["text"]))
    example["text"] = nlp(example["text"])
    print(type(example["text"]))
    return example["text"]


def new_test_func():
    pass


if __name__ == '__main__':
    sentences_dataset = import_train_data()
    # docs_dataset = sentences_dataset.map(temp_function, batched=True)
    sentences_dataset[3]["text"] = temp_function(sentences_dataset[3])
    print(type(sentences_dataset[3]["text"]))
