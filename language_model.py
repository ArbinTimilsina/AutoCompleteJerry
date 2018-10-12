from keras import layers
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

class LanguageModel:
    def __init__(self, sequence_max_len, chars, chars_mapping):
        self.sequence_max_len = sequence_max_len
        self.chars = chars
        self.chars_mapping = chars_mapping

    def build_model(self, dropout=0.1, recurrent_dropout=0.1, activation='softmax'):
        model = Sequential()
        model.add(layers.LSTM(128, 
                              dropout=dropout,
                              recurrent_dropout=recurrent_dropout,
                              input_shape=(self.sequence_max_len, len(self.chars_mapping))))
        model.add(layers.Dense(len(self.chars_mapping), activation=activation))

        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01), metrics=['acc'])

        return model

    def train_model(self, model, X, y, epochs=5, batch_size=128, filepath='saved_models/model_weights.hdf5'):
        # Stop training when a monitored quantity has stopped improving after 20 epochs
        early_stop = EarlyStopping(patience=20, verbose=1)

        # Reduce learning rate when a metric has stopped improving
        reduce_lr = ReduceLROnPlateau(factor=0.3, patience=5, cooldown=5, verbose=1)

        # Save the best model after every epoch
        check_point = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)

        # Split data into train and validation set (85/15) 
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0,
                            validation_data=(X_val, y_val),
                            callbacks=[check_point, early_stop, reduce_lr])
        
        return history

    def save_model(self, model, filepath='saved_models/model.hdf5'):
       model.save(filepath=filepath)

    def plot_history(self, history, path_accuracy='plots/model_accuracy.pdf', path_loss='plots/model_loss.pdf'):
        # Summarize history for accuracy
        plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.savefig(path_accuracy, bbox_inches='tight')

        # Summarize history for loss
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.savefig(path_loss, bbox_inches='tight')
                      
    def load_model(self, model_path='saved_models/model.hdf5', weight_path='saved_models/model_weights.hdf5'):
        model = load_model(model_path)
        model.load_weights(weight_path)
        return model

    def sample_predictions(self, predictions, temperature=1.0):
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions) / temperature
        predictions = np.exp(predictions)
        predictions = predictions / np.sum(predictions)
        sampled_predictions= np.random.multinomial(1, predictions, 1)
        return np.argmax(sampled_predictions)

    def make_prediction(self, model, text, temperature):
        break_at_char = ["?", ".", "!"]

        output = [text]
        # Limit the prediction to 150 characters
        for _ in range(150):
            test_text_encoded = [self.chars_mapping[char] for char in text]
    
            # Truncate sequences to a fixed length
            test_text_encoded = pad_sequences([test_text_encoded], maxlen=self.sequence_max_len, truncating='pre')
    
            # One-hot encoding
            test_text_encoded = to_categorical(test_text_encoded, num_classes=len(self.chars_mapping))
            test_text_encoded = test_text_encoded.reshape(1, test_text_encoded.shape[0], test_text_encoded.shape[1])
    
            # Get numpy array of predictions for the input samples
            predictions = model.predict(test_text_encoded, verbose=0)[0]
            
            # Stochastic sampling
            next_index = self.sample_predictions(predictions, temperature)
            next_char = self.chars[next_index]
            output.append(next_char)
    
            text += next_char
            
            # Continue until ?, ., or ! is met
            if next_char in break_at_char:
                break

        return output
