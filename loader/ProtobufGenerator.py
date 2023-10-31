

import math
import random
from importlib import reload

import numpy as np
import tensorflow as tf

import loader.ProtobufReader as ProtobufReader

reload(ProtobufReader)


class ProtobufGenerator(tf.keras.utils.Sequence):
    def __init__(self, readers, labels, batch_size=32):
        if len(readers) != len(labels):
            raise Exception(
                f"length of 'readers' ({len(readers)}) and 'labels' ({len(labels)}) does not match")

        if len(readers) > 0:
            self.reader = ProtobufReader.ConcatenatedProtobufReader(readers)
        else:
            raise Exception(f"length of readers is 0")

        self.labels = []

        for _labels in labels:
            self.labels.extend(_labels)

        self.batch_size = batch_size

        self.shuffled_index = list(range(len(self.reader)))
        # random.seed(0)
        # random.shuffle(self.shuffled_index)

    # def on_epoch_end(self):
    #     random.shuffle(self.shuffled_index)

    def __getitem__(self, index):
        start = index * self.batch_size
        stop = min(start + self.batch_size, len(self.reader))

        embeddings = []
        labels = []

        for i in self.shuffled_index[start:stop]:
            embeddings.append(self.reader[i])
            labels.append(self.labels[i])

        embeddings = np.array(embeddings)

        return (embeddings.reshape(-1, *embeddings.shape[1:], 1), np.array(labels))

    def get(self, index):
        result = np.array(self.reader[index])
        return (result.reshape(-1, *result.shape[1:], 1), self.labels[index])

    def __len__(self):
        return math.ceil(self.reader.__len__() / self.batch_size)
