from keras.utils import to_categorical
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer

import numpy as np
import csv

class GetData:
    def __init__(self, sequence_max_len, eos):
        self.sequence_max_len = sequence_max_len
        self.eos = eos

    def get_data(self, input_file_path):
        words_by_jerry = []
        with open(input_file_path) as input_file:
            input_data = csv.DictReader(input_file)
            for row in input_data:
                if(row['Character'] == 'JERRY'):
                    for words in row['Dialogue'].split():
                        for word in word_tokenize(words):
                            word = word.replace("...", "")
                            word = word.replace(".", self.eos)
                            word = word.replace("?", self.eos)
                            word = word.replace("!", self.eos)
                            if word.isalpha():
                                # Convert to lower letter
                                words_by_jerry.append(word.lower())

            # Create a tokenizer
            tokenizer = Tokenizer()

            # And build the word index
            tokenizer.fit_on_texts(words_by_jerry)

            # This is how we can recover the word index that was computed
            word_index = tokenizer.word_index

            # Vocabulary size
            vocabulary_size = len(word_index) + 1

            # This turns strings into lists of integer indices.
            text_sequences = tokenizer.texts_to_sequences(words_by_jerry)

            sequences = []
            for sequence in text_sequences:
                for seq in sequence:
                    sequences.append(seq)

            # Lists to hold the prefixes and targets
            prefix_sequences = []
            target_word = []
            for i in range (len(sequences) - self.sequence_max_len):
                prefix_sequences.append(sequences[i: i + self.sequence_max_len])
                target_word.append(sequences[i + self.sequence_max_len])

            X = np.array(prefix_sequences)

            # Normalize
            X = X / float(vocabulary_size)
            y = to_categorical(target_word, num_classes=vocabulary_size)

            return X, y, tokenizer, word_index, vocabulary_size
