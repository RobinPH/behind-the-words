from importlib import reload

import loader.Dataloader as Dataloader
import loader.JSONLReader as JSONLReader

reload(Dataloader)
reload(JSONLReader)


class JSONLDataloader(Dataloader.Dataloader):
    def __init__(self, _dir):
        super().__init__(_dir, "jsonl")

    def _loader(self, _dir, skip=0, take=None):
        return JSONLReader.JSONLReader(_dir, skip=skip, take=take)
