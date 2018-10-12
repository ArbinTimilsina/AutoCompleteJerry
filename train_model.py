import keras
from keras import backend as K
import string
from pickle import dump
from get_data import GetData
from language_model import LanguageModel

def get_character_mapping():
    # Each character is asigned a specific integer value
    chars = list(string.printable)
    chars_mapping = dict((char, i) for i, char in enumerate(chars))
    return chars, chars_mapping

def main():
    sequence_max_len = 40
    epochs = 200
    input_file_path = 'input_files/complete_ seinfeld_scripts.csv'

    chars, chars_mapping = get_character_mapping()

    #Get data
    get_data = GetData(sequence_max_len, chars_mapping)
    text_corpus = get_data.get_corpus(input_file_path)
    X, y = get_data.get_x_y(text_corpus)

    # Train the model
    language_model = LanguageModel(sequence_max_len, chars, chars_mapping)
    model = language_model.build_model()
    history = language_model.train_model(model, X, y, epochs=epochs)
    language_model.save_model(model)
    language_model.plot_history(history)

    # Needed for 'object has no attribute 'TF_DeleteStatus'' error
    K.clear_session()

    # Save max_len, chars, mapping for later
    for_serve = [sequence_max_len, chars, chars_mapping]
    dump(for_serve, open('saved_models/for_server.pkl', 'wb'))

if __name__ == "__main__":
    main()
