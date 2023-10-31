import json
from importlib import reload

import loader.ProtobufReader as ProtobufReader
import preprocessor.DataPreprocessor as DataPreprocessor

reload(DataPreprocessor)
reload(ProtobufReader)


class RelevantFeaturesPreprocessor(DataPreprocessor.DataPreprocessor):
    def __init__(self, relevant_features, dataset_jsonl_dataloader, **kwargs):
        super().__init__(ext="jsonl", **kwargs)

        self.relevant_features = relevant_features
        self.dataset_jsonl_dataloader = dataset_jsonl_dataloader

    def preprocess_entry(self, entry):
        text = entry["text"]

        res = self.relevant_features.get_features_raw(text)

        return {
            "_id": self.get_entry_id(entry),
            "_text": text,
            **res,
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
