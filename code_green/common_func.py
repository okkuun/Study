# -*- coding: utf-8 -*-
# Store hyperparameters and expected values of probability functions.
# import numpy as np


class Struct(dict):
    def __getattr__(self, name):
        if name == "__getstate__":
            raise AttributeError
        elif name in self:
            return self[name]

    def __setattr__(self, name, value):
        self[name] = value
