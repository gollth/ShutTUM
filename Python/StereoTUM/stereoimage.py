#!/usr/bin/env python3

import numpy as np

import StereoTUM as api


class StereoImage (api.Value):

    @staticmethod
    def extrapolate(value, shutter, method='closest'):
        if method == 'closest':
            f = value._dataset.raw.frames
            i = np.abs(f[:,1] - value.stamp).argmin()
            return StereoImage(value._dataset, f[i, :], shutter)

        if method == 'next':
            f = value._dataset.raw.frames
            frame = f[f[:,1] > value.stamp,:]
            if frame.size == 0: return None
            return StereoImage(value._dataset, frame[0], shutter)

        if method == 'prev':
            f = value._dataset.raw.frames
            frame = f[f[:, 1] < value.stamp, :]
            if frame.size == 0: return None
            return StereoImage(value._dataset, frame[-1], shutter)

        if method == 'exact':
            f = value._dataset.raw.frames
            frame = f[f[:,1] == value.stamp, :]
            if frame.size == 0: return None
            return StereoImage(value._dataset, frame[0], shutter)

        raise ValueError('Unknown extrapolation method: %s (supported are "closest", "next", "prev" and "exact")'%method)


    def __init__(self, dataset, data, shutter):
        self._data = data
        self._dataset = dataset
        self._left  = api.Image(self, shutter, left=True)
        self._right = api.Image(self, shutter, left=False)

        # Timestamp is in second column
        super().__init__(dataset, self._data[1], self._left.reference)

    def __str__(self):
        return "StereoImage ({%s|%s}/%05d" % (self._left.reference, self._right.reference, self.ID)

    def _opposite(self, img):
        if img is self._left: return self._right
        if img is self._right: return self._left
        raise ValueError("Image %s unknown, cannot find opposite")

    @property
    def ID(self):
        return int(self._data[0])

    @property
    def exposure(self):
        return self._data[2]

    @property
    def L(self):
        return self._left

    @property
    def R(self):
        return self._right

    @property
    def imu(self):
        i = self._dataset.raw.imu
        match = i[i[:, 0] == self.stamp]
        if match.size > 0: return api.ImuValue(self._dataset, match[0])

        raise ValueError("It seems that %s has no matching IMU value" % self)

