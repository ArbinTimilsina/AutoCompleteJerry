import os
import argparse
import numpy as np
from pickle import dump
from keras import backend as K
from keras.utils import plot_model
from keras.optimizers import RMSprop
from tools.data_tools import GetData
from sklearn.model_selection import train_test_split
from tools.language_model_tools import LanguageModel
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epoch", required=True,
	   help="Choose the number of epoch.")
    return vars(ap.parse_args())

def get_embedding_matrix(embedding_dim, word_index, vocabulary_size):
    glove_dir = 'glove.6B'
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.{}d.txt'.format(embedding_dim)))
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
    SEQUENCE_MAX_LEN = 3

    args = argument_parser()
    try:
        NUM_NUM_EPOCHS = int(args["epoch"])
    except ValueError:
        print("\nError: Epoch should be an integer.")
        print("Exiting!\n")
        sys.exit(1)

    input_file_path = os.path.join("input_files", "complete_seinfeld_scripts.csv")

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

    model_and_weights = os.path.join("saved_models", "model_and_weights.hdf5")
    # If weights exist, load them before training
    if(os.path.isfile(model_and_weights)):
        print("Old weights found!")
        try:
            model.load_weights(model_and_weights)
            print("Old weights loaded successfully!")
        except:
            print("Old weights couldn't be loaded successfully, will continue!")

    learning_rate = 1e-4;
    model.compile(optimizer=RMSprop(lr=learning_rate), loss='categorical_crossentropy')

    # Print model summary
    model.summary()

    # Plot the model architecture
    model_path = os.path.join("plots", "model.pdf")
    plot_model(model, to_file=model_path, show_shapes=True)

    # Stop training when a monitored quantity has stopped improving after certain epochs
    early_stop = EarlyStopping(patience=15, verbose=1)

    # Reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(factor=0.2, patience=3, cooldown=3, verbose=1)

    # Save the best model after every epoch
    check_point = ModelCheckpoint(filepath=model_and_weights, verbose=1, save_best_only=True)

    # Split data into train and validation set (85/15)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    history = model.fit(X_train, y_train, batch_size=128, epochs=NUM_NUM_EPOCHS, verbose=1,
                        validation_data=(X_val, y_val),
                        callbacks=[check_point, early_stop, reduce_lr])

    loss_path = os.path.join("plots", "loss_vs_epoch.pdf")
    language_model.plot_loss_history(history, loss_path)

    # Needed for 'object has no attribute 'TF_DeleteStatus'' error
    K.clear_session()

if __name__ == "__main__":
    main()
