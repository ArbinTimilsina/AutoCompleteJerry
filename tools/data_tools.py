import csv
import numpy as np
from keras.utils import to_categorical
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer

def remove_nonascii(word):
    return ''.join([char if ord(char) < 128 else '' for char in word])

def make_lower(word):
    return word.lower()

def remove_dots(word):
    return word.replace("...", "")

def replace_by_eos(word, eos):
    word = word.replace(".", eos)
    word = word.replace("?", eos)
    word = word.replace("!", eos)
    return word

def clean_word(word, eos):
    processed = remove_nonascii(word)
    processed = make_lower(processed)
    processed = remove_dots(processed)
    processed = replace_by_eos(processed, eos)
    return processed

class GetData:
    def __init__(self, sequence_max_len, eos):
        self.sequence_max_len = sequence_max_len
        self.eos = eos

    def get_data(self, input_file_path):
        dialogue_by_jerry = []
        with open(input_file_path) as input_file:
            input_data = csv.DictReader(input_file)
            for row in input_data:
                if(row['Character'] == 'JERRY'):
                    dialogue_by_jerry.append([clean_word(word, self.eos) for word in word_tokenize(row['Dialogue'])])

            # Create a tokenizer
            tokenizer = Tokenizer()

            # And build the word index
            tokenizer.fit_on_texts(dialogue_by_jerry)

            # This is how we can recover the word index that was computed
            word_index = tokenizer.word_index

            # Vocabulary size
            vocabulary_size = len(word_index) + 1

            prefix_word = []
            target_word = []
            for dialogue in dialogue_by_jerry:
                for i in range (len(dialogue) - self.sequence_max_len):
                    prefix_word.append(dialogue[i: i + self.sequence_max_len])
                    target_word.append(dialogue[i + self.sequence_max_len])

            print(prefix_word)

            # This turns strings into lists of integer indices.
            prefix_sequences = tokenizer.texts_to_sequences(prefix_word)
            target_sequences = tokenizer.texts_to_sequences(target_word)

            X = np.array(prefix_sequences)
            y = np.array(target_sequences)

            # Normalize
            X = X / float(vocabulary_size)
            y = to_categorical(y, num_classes=vocabulary_size)

            return X, y, tokenizer, word_index, vocabulary_size
