from language_model import LanguageModel
from flask import Flask, request, jsonify
from pickle import load

app = Flask(__name__)

def get_model():
    # Load max_len, chars, mapping
    for_server = load(open('saved_models/for_server.pkl', 'rb'))
    sequence_max_len, chars, chars_mapping = for_server[0], for_server[1], for_server[2]

    language_model = LanguageModel(sequence_max_len, chars, chars_mapping)
    model = language_model.load_model()

    return language_model, model

# Get the model information once
language_model, model = get_model() 

@app.route("/autocomplete")
def make_completions():
    try:
        seed_text = request.args.get("seed")
        suggestions = auto_complete(seed_text, language_model, model)
        return jsonify({"Seed": seed_text, "Suggested completions": [x.strip() for x in suggestions]})
    except Exception as e:
        print(e)

def auto_complete(seed_text, language_model, model):
    auto_completion = []
    # Higher temperature results in sampling distribution that will generate more surprising result
    temperatures = [0.1, 0.5, 1.0]
    for temp in temperatures:
        auto_completion.append("".join(language_model.make_prediction(model, seed_text, temp)))
    
    return auto_completion

def main(host="localhost", port=5050):
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()

