import numpy as np
import requests
import torch


class Word2Vec():
    def __init__(self, model):
        self.model = model

    def get_vec(self, words = []):
        if type(self.model) == str:
            embedded_words = requests.post(self.model, json={ "words": words }).json()
        else:
            embedded_words = [self.model[word] if word in self.model else [0] * self.model.vector_size for word in words]

        return torch.Tensor(np.array(embedded_words, dtype="float32"))
    