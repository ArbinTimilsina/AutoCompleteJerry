import csv
import string
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize, sent_tokenize

# Don't filter . and ?
filter='!"#$%&\'()*+,-/:;<=>@[\\]^_`{|}~0123456789'

def remove_punctuations_digits(sentence):
    return sentence.translate(str.maketrans('', '', filter))

def remove_dots(sentence):
    sentence = sentence.replace('..', ' ')
    sentence = sentence.replace('...', ' ')
    return sentence

def remove_nonascii(word):
    return ''.join([char if ord(char) < 128 else '' for char in word])

def make_lower(word):
    return word.lower()

def clean_word(word):
    processed = remove_nonascii(word)
    processed = make_lower(processed)
    return processed

class GetData:
    def __init__(self, sequence_max_len):
        self.sequence_max_len = sequence_max_len

    def get_data(self, input_file_path):
        sentences_by_jerry = []
        with open(input_file_path) as input_file:
            input_data = csv.DictReader(input_file)
            for row in input_data:
                if(row['Character'] == 'JERRY'):
                    if(row['Character'] == 'JERRY'):
                        for sentence in sent_tokenize(row['Dialogue']):
                            sentence = remove_dots(sentence)
                            sentence = remove_punctuations_digits(sentence)
                            sentences_by_jerry.append([clean_word(word) for word in word_tokenize(sentence)])

            # Create a tokenizer
            tokenizer = Tokenizer(filters=filter)

            # And build the word index
            tokenizer.fit_on_texts(sentences_by_jerry)

            # This is how we can recover the word index that was computed
            word_index = tokenizer.word_index

            # Vocabulary size
            vocabulary_size = len(word_index) + 1

            prefix_word = []
            target_word = []
            for sentence in sentences_by_jerry:
                for i in range (len(sentence) - self.sequence_max_len):
                    prefix_word.append(sentence[i: i + self.sequence_max_len])
                    target_word.append(sentence[i + self.sequence_max_len])

            # This turns strings into lists of integer indices.
            prefix_sequences = tokenizer.texts_to_sequences(prefix_word)
            target = tokenizer.texts_to_sequences(target_word)

            target_sequences = []
            for sequence in target:
                for seq in sequence:
                    target_sequences.append(seq)

            X = np.array(prefix_sequences)
            y = np.array(target_sequences)

            # Normalize
            X = X / float(vocabulary_size)
            y = to_categorical(y, num_classes=vocabulary_size)

            return X, y, tokenizer, word_index, vocabulary_size
