import os
import json
import argparse
import tensorflow as tf
from pickle import load
from keras.models import load_model
from flask import Flask, request, jsonify
from tools.language_model_tools import make_prediction

app = Flask(__name__)

def get_model():
    # Load max_len, chars, mapping
    for_server = load(open('saved_models/for_server.pkl', 'rb'))
    tokenizer, word_index, max_len = for_server[0], for_server[1], for_server[2]

    # Following is needed if trained in a different environment
    # See: https://github.com/keras-team/keras/issues/9099
    tokenizer.oov_token = None

    # Get the model
    model_path = os.path.join("saved_models", "model_and_weights.hdf5")
    model = load_model(model_path)

    # Needed for Tensor is not an element of this graph error
    graph = tf.get_default_graph()
    return model, graph, tokenizer, word_index, max_len

# Get the model information once
model, graph, tokenizer, word_index, max_len = get_model()

@app.route("/autocomplete")
def make_completions():
    try:
        seed_text = request.args.get("seed")
        with graph.as_default():
            suggestions = auto_complete(seed_text)
            return json.dumps({"Seed": seed_text, "Suggested completions": [x.strip() for x in suggestions]}, indent=2)
    except Exception as e:
        print(e)

def auto_complete(seed_text):
    auto_completion = []
    # Higher temperature results in sampling distribution that will generate more surprising result
    temperatures = [0.8, 1.0, 1.2, 2.5]
    for temp in temperatures:
        auto_completion.append("".join(make_prediction(model, seed_text, temp, tokenizer, word_index, max_len)))
        # Add full stop at the end of the sentence
        auto_completion.append(".")
    return auto_completion

def main(host="localhost", port=5050):
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()
