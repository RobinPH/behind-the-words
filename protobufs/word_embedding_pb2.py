# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: word-embedding.proto
# Protobuf Python Version: 4.25.0-rc2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14word-embedding.proto\x12\x0e\x65ssayembedding\"^\n\x0e\x45ssayEmbedding\x12\x0b\n\x03_id\x18\x01 \x01(\t\x12\r\n\x05_text\x18\x02 \x01(\t\x12\x30\n\tembedding\x18\x03 \x03(\x0b\x32\x1d.essayembedding.WordEmbedding\"\"\n\rWordEmbedding\x12\x11\n\x05value\x18\x01 \x03(\x02\x42\x02\x10\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'word_embedding_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_WORDEMBEDDING'].fields_by_name['value']._options = None
  _globals['_WORDEMBEDDING'].fields_by_name['value']._serialized_options = b'\020\001'
  _globals['_ESSAYEMBEDDING']._serialized_start=40
  _globals['_ESSAYEMBEDDING']._serialized_end=134
  _globals['_WORDEMBEDDING']._serialized_start=136
  _globals['_WORDEMBEDDING']._serialized_end=170
# @@protoc_insertion_point(module_scope)