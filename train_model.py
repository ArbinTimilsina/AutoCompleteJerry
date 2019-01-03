import argparse
from os import path
from keras import backend as K
from pickle import dump
import numpy as np
from tools.data_tools import GetData
from tools.language_model_tools import LanguageModel

def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--sequence_lenght", required=True,
	   help="Choose the sequence lenght.'")
    ap.add_argument("-e", "--epoch", required=True,
	   help="Choose the number of epoch.")
    return vars(ap.parse_args())

def get_embedding_matrix(embedding_dim, word_index, vocabulary_size):
    glove_dir = 'glove.6B'
    embeddings_index = {}
    f = open(path.join(glove_dir, 'glove.6B.{}d.txt'.format(embedding_dim)))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((vocabulary_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i < vocabulary_size:
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


def main():
    args = argument_parser()
    try:
        SEQUENCE_MAX_LEN = int(args["sequence_lenght"])
    except ValueError:
        print("\nWarning: Sequence lenght should be an integer.")
        SEQUENCE_MAX_LEN = 7
        print("Setting it to {}.".format(SEQUENCE_MAX_LEN))

    try:
        NUM_NUM_EPOCHS = int(args["epoch"])
    except ValueError:
        print("\nError: Epoch should be an integer.")
        print("Exiting!\n")
        sys.exit(1)

    input_file_path = path.join("input_files", "complete_seinfeld_scripts.csv")

    #Get data
    EOS = "eos"
    data = GetData(SEQUENCE_MAX_LEN, EOS)
    X, y, tokenizer, word_index, vocabulary_size = data.get_data(input_file_path)

    # Save word_index
    for_serve = [tokenizer, word_index, SEQUENCE_MAX_LEN, EOS]
    dump(for_serve, open('saved_models/for_server.pkl', 'wb'))

    # Options are 50, 100, 200, 300
    embedding_dim = 50
    embedding_matrix = get_embedding_matrix(embedding_dim, word_index, vocabulary_size)

    # Train the model
    language_model = LanguageModel(vocabulary_size, SEQUENCE_MAX_LEN, embedding_dim, embedding_matrix)
    model = language_model.build_model()

    model_and_weights = path.join("saved_models", "model_and_weights.hdf5")
    # If weights exist, load them before training
    if(path.isfile(model_and_weights)):
        print("Old weights found!")
        try:
            model.load_weights(model_and_weights)
            print("Old weights loaded successfully!")
        except:
            print("Old weights couldn't be loaded successfully, will continue!")

    history = language_model.train_model(model, X, y, epochs=NUM_NUM_EPOCHS, batch_size=128, file_path=model_and_weights)

    loss_path = path.join("plots", "loss_vs_epoch.pdf")
    language_model.plot_loss_history(history, loss_path)

    # Needed for 'object has no attribute 'TF_DeleteStatus'' error
    K.clear_session()

if __name__ == "__main__":
    main()
