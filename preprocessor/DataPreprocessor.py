import os

from tqdm.auto import tqdm


class DataPreprocessor:
    def __init__(self, _dir, ext, dataset_version="v1", in_folders=[], out_folders=[]):
        self.dir = _dir
        self.dataset_version = dataset_version
        self.in_folders = [f"dataset-{dataset_version}", *in_folders]
        self.out_folders = [f"processed-{dataset_version}", *out_folders]
        self.ext = ext

    def preprocess(self, in_filename, out_filename = None, skip = 0, take = None, **kwargs):
        if out_filename == None:
            out_filename = in_filename

        for label, dataset in self.load_dataset(os.path.join(*self.in_folders, in_filename), skip = skip, take = take):
            _dir = os.path.join(
                self.dir, *self.out_folders, ".".join([out_filename, label, self.ext])
            )

            print(f"Preprocessing {_dir}")
            os.makedirs(os.path.dirname(_dir), exist_ok=True)

            with open(_dir, "a") as f:
                pass

            existing_ids = self.get_existing_ids(_dir)
            print(f"Found ({len(existing_ids)}) existing ids")

            with open(**self.open_file_kwargs(_dir)) as f:
                for entry in tqdm(dataset):
                    _id = self.get_entry_id(entry)

                    if _id in existing_ids:
                        continue

                    try:
                        preprocessed_entry = self.preprocess_entry(entry, **kwargs)

                        self.write_preprocessed_entry(f, preprocessed_entry)
                    except Exception as e:
                        print(e)
                        print(f"Failed: {_dir} {_id}")

    def preprocess_entry(self, entry, **kwargs):
        raise Exception("Not yet implemeneted")

    def open_file_kwargs(self, dir):
        raise Exception("Not yet implemeneted")

    def write_preprocessed_entry(self, file, preprocessed_entry):
        raise Exception("Not yet implemeneted")

    def load_dataset(self, filename, skip = 0, take = None):
        raise Exception("Not yet implemeneted")

    def get_existing_ids(self):
        raise Exception("Not yet implemeneted")

    def get_entry_id(self, entry):
        raise Exception("Not yet implemeneted")
    
    def get_preprocessed_entry_id(self, preprocessed_entry):
        raise Exception("Not yet implemeneted")
