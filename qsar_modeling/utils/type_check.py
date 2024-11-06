import collections
import copy

import pandas as pd


def is_iterable(thing):
    return any([[type(thing) is t for t in [pd.Index, list, tuple, set, collections.Container]]])


def make_str_type(thing, cls_meth, copy_thing=True):
    if copy_thing:
        thing = copy.deepcopy(thing)
    if is_iterable(thing):
        return [make_str_type(t) for t in thing]
    if type(thing) is not str:
        thing = str(thing)
    return cls_meth(thing)


def make_lowercase(thing, copy_thing=True):
    return make_str_type(thing, cls_meth=str.lower, copy_thing=copy_thing)
