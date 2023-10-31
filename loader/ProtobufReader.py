class ProtobufReader():
    def __init__(self, protobuf, file_path, parser=None, skip = 0, take = None):
        self.protobuf = protobuf
        self.file_path = file_path
        self.f = None
        self.skip = skip
        self._init_file()
        self.take = take if take != None else max(0, len(self.line_indices) - self.skip)
        self.parser = parser

    def _get_line_indices(self):
        current_index = 0
        indices = []

        while length_byte := self.f.read(8):
            indices.append(current_index)

            length = ProtobufReader.byte_to_int(length_byte)
            self.f.seek(length, 1)

            current_index += 8
            current_index += length

        self.f.seek(0, 0)

        return indices

    def _init_file(self):
        if self.f != None:
            return

        self.f = open(self.file_path, "rb")
        self.line_indices = self._get_line_indices()
    
    def _close(self):
        if self.f:
            self.f.close()
    
    def __enter__(self):
        self._init_file()

        return self

    def __exit__(self, *args):
        self._close()

    def __del__(self):
        self._close()

    @staticmethod
    def byte_to_int(byte):
        return int.from_bytes(byte, byteorder="big")
    
    def parse_line(self, start_byte=None):
        if start_byte != None:
            self.f.seek(start_byte, 0)

        if (length_byte := self.f.read(8)):
            length = ProtobufReader.byte_to_int(length_byte)

            protobuf = self.protobuf()
            protobuf.ParseFromString(self.f.read(length))

            if self.parser:
                return self.parser(protobuf)
            else:
                return protobuf
        
        return None

    def get(self, skip = 0, take = None):
        indices = self.line_indices[skip:take]

        return [self.parse_line(index) for index in indices]
    
    def __getitem__(self, index):
        self.f.seek(0, 0)

        if type(index) == slice:
            start = index.start if index.start != None else 0
            stop = index.stop if index.stop != None else len(self)
            take = min(self.take, max(0, stop - start))

            start += self.skip

            indices = self.line_indices[start:(start + take)]

            return [self.parse_line(index) for index in indices]
        else:
            return self.parse_line(self.line_indices[index])
        
    def __len__(self):
        return min(self.take, max(0, len(self.line_indices) - self.skip))
    
    def __add__(self, other):
        if type(other) == ProtobufReader:
            return ConcatenatedProtobufReader([self, other])
        elif type(other) == ConcatenatedProtobufReader:
            return ConcatenatedProtobufReader([self, *other.readers])
        
    def __iter__(self):
        self.current_index = 0
        return self
    
    def __next__(self):
        if self.current_index < len(self):
            x = self[self.current_index]
            self.current_index += 1
            return x
        raise StopIteration
    
class ConcatenatedProtobufReader():
    def __init__(self, readers = []):
        self.readers = readers
        self._lengths = [len(reader) for reader in self.readers]
        self._lengths_prefix_sum = [0]

        for length in self._lengths:
            self._lengths_prefix_sum.append(self._lengths_prefix_sum[-1] + length)
            
        self._length = sum(self._lengths)

    def get_reader_by_index(self, idx):
        length_sum = 0

        for i, reader in enumerate(self.readers):
            length_sum += len(reader)

            if idx < length_sum:
                return (reader, i)
            
        return (None, None)


    def __getitem__(self, index):
        if type(index) == slice:
            start = index.start if index.start != None else 0
            stop = index.stop if index.stop != None else len(self)

            reader_start, start_index = self.get_reader_by_index(start)
            reader_stop, stop_index = self.get_reader_by_index(stop - 1)

            result = []
            _start_in_reader = start - self._lengths_prefix_sum[start_index]
            _stop_in_reader = stop - self._lengths_prefix_sum[stop_index]

            if start_index != None and stop_index != None:
                if start_index != stop_index:
                    for i in range(start_index, stop_index + 1):
                        if start_index == i:
                            result.extend(self.readers[i][_start_in_reader:])
                        elif stop_index == i:
                            result.extend(self.readers[i][:_stop_in_reader])
                        else:
                            result.extend(self.readers[i][:])
                else:
                    result.extend(reader_start[_start_in_reader:_stop_in_reader])
            elif start_index != None:
                result.extend(reader_start[_start_in_reader:])
                for i in range(start_index + 1, len(self.readers)):
                    result.extend(self.readers[i][:])
            elif stop_index != None:
                for i in range(0, stop_index):
                    result.extend(self.readers[i][:])
                result.extend(reader_stop[:_stop_in_reader])
            else:
                for reader in self.readers:
                    result.extend(reader[:])

            return result
        else:
            reader, idx = self.get_reader_by_index(index)

            if reader == None:
                return None
            
            _index = index - self._lengths_prefix_sum[idx]
            
            return reader[_index]


    def __len__(self):
        return self._length
    

    def __add__(self, other):
        if type(other) == ProtobufReader:
            return ConcatenatedProtobufReader([*self.readers, other])
        elif type(other) == ConcatenatedProtobufReader:
            return ConcatenatedProtobufReader([self, *other.readers])
        
    def __iter__(self):
        self.current_index = 0
        return self
    
    def __next__(self):
        if self.current_index < len(self):
            x = self[self.current_index]
            self.current_index += 1
            return x
        raise StopIteration