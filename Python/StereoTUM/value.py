#!/usr/bin/env python3

import numpy as np


class Value(object):

    def __init__(self, dataset, stamp, reference):
        self._dataset = dataset
        self._stamp = stamp
        if reference not in dataset._refs:
            raise ValueError("Cannot find the reference %s" % reference)

        self._reference = reference

    @property
    def stamp(self):
        return self._stamp

    @property
    def reference(self):
        return self._reference

    @property
    def _transform(self):
        return np.array(self._dataset._refs[self._reference]['transform'])

    def __str__(self):
        return "%s (%s/%.2f)" % (type(self).__name__, self.reference, self.stamp)

    def __lshift__(self, parent) -> np.ndarray:
        if isinstance(parent, str):
            if parent not in self._dataset._refs:
                raise ValueError("Cannot find the (static) parent reference %s" % parent)
            tparent = np.array(self._dataset._refs[parent]['transform'])

        elif isinstance(parent, Value):
            tparent = parent._transform
        else:
            raise TypeError("Cannot only lookup transforms for type string or StereoTUM.Value")

        tchild = self._transform
        return np.dot(np.linalg.inv(tparent), tchild)

    def __rshift__(self, child) -> np.ndarray:
        if isinstance(child, str):
            if child not in self._dataset._refs:
                raise ValueError("Cannot find the (static) parent reference %s" % child)
            tchild = np.array(self._dataset._refs[child]['transform'])
        elif isinstance(child, Value):
            tchild = child._transform
        else:
            raise TypeError("Cannot only lookup transforms for type string or StereoTUM.Value")

        tparent = self._transform
        return np.dot(np.linalg.inv(tparent), tchild)


