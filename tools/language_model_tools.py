import matplotlib
matplotlib.use('Agg')
import spacy, re
import numpy as np
from keras.models import Model
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.models import Sequential
from nltk.tokenize import word_tokenize
from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tools.data_tools import remove_punctuations_digits, clean_word
from keras.layers import Input, Embedding, Dropout, LSTM, Dense

def improve_output(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    #Tag: The detailed part-of-speech tag. NN: common noun, NNS: plural, NNP: proper noun
    tagged_sentence = [(w.text, w.tag_) for w in doc]
    normalized_sent = [w.capitalize() if t in ["NNP", "NNPS"] else w for (w, t) in tagged_sentence]
    normalized_sent[0] = normalized_sent[0].capitalize()
    return re.sub(" (?=[\.,'!?:;])", "", ' '.join(normalized_sent))

def sample_predictions(predictions, temperature):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    predictions = np.exp(predictions)
    predictions = predictions / np.sum(predictions)
    sampled_predictions= np.random.multinomial(1, predictions, 1)
    return np.argmax(sampled_predictions)

def make_prediction(model, text, temperature, tokenizer, word_index, max_len):
    text = remove_punctuations_digits(text)
    output = [text]
    # Limit the prediction to 25 words
    for _ in range(25):
        clean_text = []
        for word in word_tokenize(text):
            clean_text.append(clean_word(word))

        text_sequences = tokenizer.texts_to_sequences(clean_text)

        # Truncate sequences to a fixed length
        text_padded = pad_sequences(text_sequences, maxlen=max_len, truncating='pre')

        predictions = model.predict(text_padded, verbose=0)[0]
        # Stochastic sampling
        next_index = sample_predictions(predictions, temperature)

        # map predicted word index to word
        next_word = ''
        EOS = ".?"
        for word, index in word_index.items():
            if index == next_index:
                next_word = word
                break

        if next_word in EOS:
            break

        output.append(next_word)
        text += " " + next_word

    return improve_output(text)

class LanguageModel:
    def __init__(self, vocabulary_size, sequence_max_len, embedding_dim, embedding_matrix):
        self.vocabulary_size = vocabulary_size
        self.sequence_max_len = sequence_max_len
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix


    def build_model(self, dropout=0.25):
        regularizer = l2(1e-4)

        inputs = Input(shape=(self.sequence_max_len,))

        embedding = Embedding(self.vocabulary_size, self.embedding_dim, input_length=self.sequence_max_len)(inputs)
        embedding = Dropout(dropout)(embedding)

        LSTM1 = LSTM(self.embedding_dim,
                       return_sequences=True,
                       dropout=dropout, recurrent_dropout=dropout,
                       kernel_regularizer=regularizer, recurrent_regularizer=regularizer, bias_regularizer=regularizer)(embedding)
        LSTM2 = LSTM(self.embedding_dim,
                       return_sequences=True,
                       dropout=dropout, recurrent_dropout=dropout,
                       kernel_regularizer=regularizer, recurrent_regularizer=regularizer, bias_regularizer=regularizer)(LSTM1)

        concat = concatenate([embedding, LSTM1, LSTM2])

        LSTM3 = LSTM(self.embedding_dim,
                       return_sequences=False,
                       dropout=dropout, recurrent_dropout=dropout,
                       kernel_regularizer=regularizer, recurrent_regularizer=regularizer, bias_regularizer=regularizer)(concat)

        dense1 = Dense(self.embedding_dim, activation='relu', kernel_regularizer=regularizer, bias_regularizer=regularizer)(LSTM3)
        dense1 = Dropout(dropout)(dense1)

        outputs = Dense(self.vocabulary_size, activation='softmax')(dense1)

        model = Model(inputs=[inputs], outputs=[outputs])

        model.layers[1].set_weights([self.embedding_matrix])
        model.layers[1].trainable = False

        return model

    def plot_loss_history(self, history, file_path):
        # Summarize history for loss
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.savefig(file_path, bbox_inches='tight')
