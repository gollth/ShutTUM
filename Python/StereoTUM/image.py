import cv2
import os.path as p
import numpy as np
import transforms3d as tf
import StereoTUM as api


class Image (api.Value):
    r"""
    An image is a :class:`Value` with its reference set to either "cam1" ... "cam4".
        
    Each image is recorded by a :class:`StereoCamera`, which has a shutter type, so as its images. Note, though,
    that each camera might differ its shutter methods between dataset records, to achieve statistical independence. 
    That means, that you cannot rely on e.g. "cam1" always having "global" or "rolling" shutter, nor as the other cams.
    
    The cameras record data at approximately 20 FPS, but sometimes their might exist frame drops. 
    
    You can query a lot of information from an image such as its :attr:`shutter`, :attr:`exposure` time and :attr:`ID`.
    Since it is a :class:`Value` all transform shenanigans apply.
    """
    def __init__(self, stereo, shutter, left):
        self._left = left
        self._stereo = stereo
        self._shutter = shutter

        if shutter not in ['global', 'rolling']:
            raise ValueError('Shutter type can only be "global" or "rolling"')

        for cam in stereo._dataset._cams:
            # Check if the shutter matches
            if stereo._dataset._cams[cam]['shutter']['type'] != shutter: continue

            # Check if the camera position (left/right) matches the camera name
            if     left and cam not in ['cam1', 'cam3']: continue
            if not left and cam not in ['cam2', 'cam4']: continue

            # Now we have found a camera matching the wanted shutter type and position
            super().__init__(stereo._dataset, stereo._data[1], cam)
            break   # from any further for loop iteration

    def __str__(self):
        return "Image (%s/#%05d/%.2f)" % (self.reference, self.ID, self.stamp)

    @property
    def shutter(self):
        r""" The shutter method with which this image was captured as string, either "rolling" or "global" """
        return self._shutter

    @property
    def ID(self):
        r"""The frame ID as int of this image. This number, prepended to 5 digits is also the name of the JPEG file"""
        return self._stereo.ID

    @property
    def opposite(self):
        r"""This holds the reference of the opposite image, that is the image taken from the other camera with the 
        same shutter."""
        return self._stereo._opposite(self)

    @property
    def exposure(self):
        r"""The exposure time as float in milli seconds, that this image was illuminated"""
        return self._stereo.exposure

    @property
    def path(self):
        r"""The path to the JPEG file of this image, relative to the the construction parameter of :func:`Dataset.__init__`"""
        return p.join(self._stereo._dataset._path, 'frames', self.reference, '%05d.jpeg' % self.ID)

    @property
    def imu(self):
        r"""The matching :class:`ImuValue` for this image. Since the capture of an image is synchronized with the IMU,
        no interpolation is needed."""
        return self._stereo.imu

    def groundtruth(self, position_interpolation=api.Interpolation.linear,
                          orientation_interpolation=api.Interpolation.slerp):
        r"""
        Find the matching :class:`GroundTruth` value for this image. Since the motion capture system and the cameras
        are not synced, we need to interpolate between ground truths by image's time stamp.
        
        :param position_interpolation: a predefined or custom interpolation function to interpolate positions 
        :param orientation_interpolation: a predefined or custom interpolation function to interpolate quaternions
        :return: the matching interpolated ground truth
         
        .. seealso:: :func:`GroundTruth.interpolate`
        """
        # The ground truth value is from world to our reference
        # Therefore we first get the gt from world -> cam1 ...
        gt1 = api.GroundTruth.interpolate(self._dataset, self.stamp, position_interpolation, orientation_interpolation)

        # ... then from cam1 -> us
        t1 = self << 'cam1'

        # which delivers from world -> us
        A = gt1.pose.dot(t1)
        T, R, _, _ = tf.affines.decompose44(A)
        q = tf.quaternions.mat2quat(R)
        p = np.array(T)
        return api.GroundTruth(self._dataset, np.concatenate(([gt1.stamp], p, q)))

    def load(self):
        r"""Loads the JPEG itself into memory::
        
            image = ...
            pixels = image.load()   # uses cv2.imread()
            print(pixels.shape)     # (1024, 1280)
            print(type(pixels))     # numpy.ndarray
            
        """
        return cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
