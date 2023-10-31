from importlib import reload

import loader.ProtobufReader as ProtobufReader
import preprocessor.DataPreprocessor as DataPreprocessor
import protobufs.word_embedding_pb2 as EssayEmbedding

reload(DataPreprocessor)
reload(ProtobufReader)


class WordEmbeddingPreprocessor(DataPreprocessor.DataPreprocessor):
    def __init__(self, text_to_embedding, dataset_jsonl_dataloader, **kwargs):
        super().__init__(ext="binpb", **kwargs)

        self.text_to_embedding = text_to_embedding
        self.dataset_jsonl_dataloader = dataset_jsonl_dataloader

    def preprocess(self, in_filename, max_token_length=384):
        super().preprocess(in_filename,
                           f"{in_filename}-{max_token_length}", max_token_length=max_token_length)

    def preprocess_entry(self, entry, **kwargs):
        text = entry["text"]
        embedding = self.text_to_embedding(text, kwargs["max_token_length"])
        embedding_protobuf = self.embedding_as_protobuf(
            str(entry["id"]), text, embedding)

        return embedding_protobuf.SerializeToString()

    def open_file_kwargs(self, _dir):
        return {
            "file": _dir,
            "mode": "ab"
        }

    def write_preprocessed_entry(self, file, preprocessed_entry):
        serialized = preprocessed_entry
        length = len(serialized)

        file.write(length.to_bytes(8, "big"))
        file.write(serialized)

    def load_dataset(self, filename, skip=0, take=None):
        datasets = self.dataset_jsonl_dataloader.get(
            filename, skip=skip, take=take, include_label=True)

        return [(label, dataset.iloc) for label, dataset in datasets]

    def get_existing_ids(self, _dir):
        existing_ids = set()

        with open(_dir, 'r') as f:
            reader = ProtobufReader.ProtobufReader(
                EssayEmbedding.EssayEmbedding, _dir)

            for i in range(len(reader)):
                entry = reader[i]
                existing_ids.add(self.get_preprocessed_entry_id(entry))

        return existing_ids

    def get_preprocessed_entry_id(self, preprocessed_entry):
        return str(preprocessed_entry._id)

    def get_entry_id(self, entry):
        return str(entry["id"])

    def embedding_as_protobuf(self, _id, _text, embeddings):
        essay_embedding = EssayEmbedding.EssayEmbedding()
        essay_embedding._id = _id
        essay_embedding._text = _text

        for embedding in embeddings:
            _embedding = EssayEmbedding.WordEmbedding()
            for value in embedding:
                _embedding.value.append(value)

            essay_embedding.embedding.append(_embedding)

        return essay_embedding
