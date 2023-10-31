import os
import sys

from flask import Flask, jsonify, request
from gensim import models

import metaphor.word2vec as w2v

BEHIND_THE_WORDS_DIR = "./"


app = Flask(__name__)

sys.path.append(os.path.join(BEHIND_THE_WORDS_DIR, 'metaphor/'))

w2v_model_path = os.path.join(
    BEHIND_THE_WORDS_DIR, "data/gensim/word2vec-google-news-300.gz")
print("Loading Word2Vec (word2vec-google-news-300)")
w2v_model = models.KeyedVectors.load_word2vec_format(
    w2v_model_path, binary=True)
print("Loaded")

word2vec = w2v.Word2Vec(w2v_model)


@app.route("/", methods=["POST"])
def get_vec():
    content = request.json

    words = content["words"]

    return jsonify(word2vec.get_vec(words).tolist())


if __name__ == '__main__':
    app.run(debug=False, port="7070")
