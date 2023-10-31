import json
from importlib import reload

import numpy as np

import loader.ProtobufReader as ProtobufReader
import preprocessor.DataPreprocessor as DataPreprocessor

reload(DataPreprocessor)
reload(ProtobufReader)


class CNNPredictionPreprocessor(DataPreprocessor.DataPreprocessor):
    def __init__(self, text_to_embedding, dataset_jsonl_dataloader, **kwargs):
        super().__init__(ext="jsonl", **kwargs)

        self.text_to_embedding = text_to_embedding
        self.dataset_jsonl_dataloader = dataset_jsonl_dataloader

    def preprocess(self, in_filename, model_cnn, **kwargs):
        super().preprocess(in_filename, in_filename, model_cnn=model_cnn, **kwargs)

    def preprocess_entry(self, entry, **kwargs):
        text = entry["text"]
        model_cnn = kwargs["model_cnn"]

        data = np.array([self.text_to_embedding(text)])
        data = data.reshape(-1, *data.shape[1:], 1)

        prediction = model_cnn.predict(data, verbose=None).tolist()[0][0]

        return {
            "_id": self.get_entry_id(entry),
            "prediction": prediction,
            "_text": text,
        }

    def open_file_kwargs(self, _dir):
        return {
            "file": _dir,
            "mode": "a"
        }

    def write_preprocessed_entry(self, file, preprocessed_entry):
        file.write(json.dumps(preprocessed_entry) + "\n")

    def load_dataset(self, filename, skip=0, take=None):
        datasets = self.dataset_jsonl_dataloader.get(
            filename, skip=skip, take=take, include_label=True)

        return [(label, dataset.iloc) for label, dataset in datasets]

    def get_existing_ids(self, _dir):
        existing_ids = set()

        with open(_dir, 'r') as f:
            for line in f.readlines():
                entry = json.loads(line)
                existing_ids.add(self.get_preprocessed_entry_id(entry))

        return existing_ids

    def get_preprocessed_entry_id(self, preprocessed_entry):
        return str(preprocessed_entry["_id"])

    def get_entry_id(self, entry):
        return str(entry["id"])
