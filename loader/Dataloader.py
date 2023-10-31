import os


class Dataloader():
    def __init__(self, _dir, ext):
        self.dir = _dir
        self.ext = ext

    
    def get(self, name, folders = [], skip = 0, take = None, distribution = [("train", 6), ("test", 3), ("valid", 1)], include_label = False):
        DIR = os.path.join(self.dir, *folders)

        distribution_total = sum(map(lambda d: d[1], distribution))

        datasets = []

        for label, value in distribution:
            _dir = os.path.join(DIR, ".".join([name, label, self.ext]))
            _skip = int(skip * value / distribution_total)

            if take:
                _take = int(take * value / distribution_total)
            else:
                _take = None

            os.makedirs(os.path.dirname(_dir), exist_ok=True)
            dataset = self._loader(_dir=_dir, skip=_skip, take=_take)

            print(f"{_dir} skip={_skip} take={_take} length={len(dataset)}")

            if include_label:
                datasets.append((label, dataset))
            else:
                datasets.append(dataset)

        return tuple(datasets)
    
    def _loader(self, _dir, skip = 0, take = None):
        return []