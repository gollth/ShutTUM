#!/usr/bin/env python3

import numpy as np

import StereoTUM as api


class StereoImage (api.Value):
    r"""A stereo image contains of two individual :class:`Image`s, a left and a right one.
    It is more or less a container for the two.
    
    Note that for internal reasons a stereo image derives from :class:`Value`. However, you should
    not use the transform functions (``<<``and ``>>``) with this, since a stereo image contains two 
    reference frames, one for each camera::
    
        stereo = dataset.cameras('rolling')[0]
        
        # The following is ambiguous
        notok = stereo << "world"   # which camera of the two do you mean?
        
        # Better would be
        ok = stereo.L << "world"
        
    
    """
    @staticmethod
    def extrapolate(value, shutter, method='closest'):
        r"""
        Find a matching image for a certain :class:`Value` based on a extrapolation method
        
        :param value: The value for which to find a matching image
        :param shutter: The shutter type of the images to find ("global", "rolling", **not** "both") 
        :param method: An optional extrapolation method to determine the rules for a "match":
            * **"closest"**: the image with the least difference to value.stamp is chosen
            * **"next"**: the image with the next larger time stamp than value.stamp is chosen
            * **"prev"**: the image with the next smaller time stamp than value.stamp is chosen
            * **"exact"**: the image where value.stamp == image.stamp holds is chosen, None otherwise
        :return: The matching stereo image or None if no was found
        """
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
        r"""The frame ID as int of this image. This number, prepended to 5 digits is also the name of the JPEG file"""
        return int(self._data[0])

    @property
    def exposure(self):
        r"""The exposure time as float in milli seconds, that this image was illuminated. This is constant for :attr:`L` and :attr:`R`"""
        return self._data[2]

    @property
    def L(self):
        r""" The rerference to the left :class:`Image`"""
        return self._left

    @property
    def R(self):
        r""" The rerference to the right :class:`Image`"""
        return self._right

    @property
    def imu(self):
        r"""The matching :class:`ImuValue` for this image. Since the capture of an image is synchronized with the IMU,
                no interpolation is needed."""
        i = self._dataset.raw.imu
        match = i[i[:, 0] == self.stamp]
        if match.size > 0: return api.ImuValue(self._dataset, match[0])

        raise ValueError("It seems that %s has no matching IMU value" % self)

