import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split

def sample_predictions(predictions, temperature):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    predictions = np.exp(predictions)
    predictions = predictions / np.sum(predictions)
    sampled_predictions= np.random.multinomial(1, predictions, 1)
    return np.argmax(sampled_predictions)

def make_prediction(model, text, temperature, tokenizer, word_index, max_len, eos):
    text = text.lower()
    output = [text]
    # Limit the prediction to 25 words
    for _ in range(25):
        clean_text = []
        for word in word_tokenize(text):
            if word.isalpha():
                clean_text.append(word)

        text_sequences = tokenizer.texts_to_sequences(clean_text)
        sequences = []
        for sequence in text_sequences:
            for seq in sequence:
                sequences.append(seq)

        # Truncate sequences to a fixed length'
        test_text_encoded = pad_sequences([sequences], maxlen=max_len, truncating='pre')
        predictions = model.predict(test_text_encoded, verbose=0)[0]
        # Stochastic sampling
        next_index = sample_predictions(predictions, temperature)

        # map predicted word index to word
        next_word = ''
        for word, index in word_index.items():
            if index == next_index:
                next_word = word
                break

        if next_word == eos:
            break

        output.append(next_word)
        text += " " + next_word

    return text


class LanguageModel:
    def __init__(self, vocabulary_size, sequence_max_len, embedding_dim, embedding_matrix):
        self.vocabulary_size = vocabulary_size
        self.sequence_max_len = sequence_max_len
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix


    def build_model(self, dropout=0.25, recurrent_dropout=0.25, activation='softmax'):
        model = Sequential()
        model.add(Embedding(self.vocabulary_size, self.embedding_dim, input_length=self.sequence_max_len))
        model.add(LSTM(self.embedding_dim, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))
        model.add(LSTM(self.embedding_dim, dropout=dropout, recurrent_dropout=recurrent_dropout))
        model.add(Dense(self.embedding_dim, activation='relu'))
        model.add(Dropout(dropout))

        model.add(Dense(self.vocabulary_size, activation='softmax'))

        model.layers[0].set_weights([self.embedding_matrix])
        model.layers[0].trainable = False

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1E-3))

        return model

    def train_model(self, model, X, y, epochs, batch_size, file_path):
        # Reduce learning rate when a metric has stopped improving
        reduce_lr = ReduceLROnPlateau(factor=0.3, patience=3, cooldown=3, verbose=1)

        # Save the best model after every epoch
        check_point = ModelCheckpoint(filepath=file_path, verbose=1, save_best_only=False)

        # Split data into train and validation set (85/15)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0,
                            validation_data=(X_val, y_val),
                            callbacks=[check_point, reduce_lr])

        return history

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
