from importlib import reload

import loader.Dataloader as Dataloader
import loader.ProtobufReader as ProtobufReader

reload(Dataloader)
reload(ProtobufReader)


class ProtobufDataloader(Dataloader.Dataloader):
    def __init__(self, _dir, parser=None, protobuf=None):
        super().__init__(_dir, "binpb")

        self.protobuf = protobuf
        self.parser = parser

    def _loader(self, _dir, skip=0, take=None):
        return ProtobufReader.ProtobufReader(self.protobuf, _dir, skip=skip, take=take, parser=self.parser)
