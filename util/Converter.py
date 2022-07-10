import codecs
import json
import pickle

from pandas import json_normalize


def encode(value):
    pickled = str(codecs.encode(pickle.dumps(value), "base64").decode())
    return pickled


def decode(value):
    unpickled = pickle.loads(codecs.decode(value.encode(), "base64"))
    return unpickled


def toJson(value):
    return json.loads(value)

def toDataFrame(value):
    return json_normalize(value)
