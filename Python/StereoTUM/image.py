import cv2
import os.path as p
import numpy as np
import transforms3d as tf
import StereoTUM as api


class Image (api.Value):

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
        return self._shutter

    @property
    def ID(self):
        return self._stereo.ID

    @property
    def opposite(self):
        return self._stereo._opposite(self)

    @property
    def exposure(self):
        return self._stereo.exposure

    @property
    def path(self):
        return p.join(self._stereo._dataset._path, 'frames', self.reference, '%05d.jpeg' % self.ID)

    @property
    def imu(self):
        return self._stereo.imu

    @property
    def groundtruth(self):
        # The ground truth value is from world to our reference
        # Therefore we first get the gt from world -> cam1 ...
        gt1 = api.GroundTruth.interpolate(self._dataset, self.stamp)

        # ... then from cam1 -> us
        t1 = self << 'cam1'

        # which delivers from world -> us
        A = gt1.pose.dot(t1)
        T, R, _, _ = tf.affines.decompose44(A)
        q = tf.quaternions.mat2quat(R)
        p = np.array(T)
        return api.GroundTruth(self._dataset, np.concatenate(([gt1.stamp], p, q)))

    def load(self):
        return cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)