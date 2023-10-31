from importlib import reload

import requests
from gensim import models

import utils.word2vec as w2v

reload(w2v)


def load_word2vec(local_w2v_path, remote_url=None):
    if remote_url:
        response = requests.post(remote_url, json={"words": []})

        if response.status_code == 200:
            print(f"Using remote w2v_model {remote_url}")
            return w2v.Word2Vec(remote_url)
        else:
            print(
                f"Provided remote_url {remote_url} responded with status code {response.status_code}")

    print("Using local w2v_model")
    return w2v.Word2Vec(models.KeyedVectors.load_word2vec_format(local_w2v_path, binary=True))
