#!/usr/bin/env python

import math
import numpy as np
import transforms3d as tf
import StereoTUM as api


class GroundTruth (api.Value):

    @staticmethod
    def interpolate(dataset, stamp, position_interpolation=api.Interpolation.linear, orientation_interpolation=api.Interpolation.slerp):
        poses = dataset.raw.groundtruth
        idx = np.searchsorted(poses[:, 0], stamp)
        p = np.ones((1,3)) * np.nan
        q = np.ones((1,4)) * np.nan
        if idx != 0 and idx != poses.shape[0]:
            t = (stamp - poses[idx - 1, 0]) / (poses[idx, 0] - poses[idx - 1, 0])

            # Compute previous values for interpolation method, if existant
            a0, b0 = np.zeros((3,1)), np.zeros((3,1))
            qa0, qb0 = np.array((1,0,0,0)), np.array((1,0,0,0))
            if idx-1 >= 0:             a0, qa0 = poses[idx-2, 1:4], poses[idx-2, 4:8]
            if idx+1 < poses.shape[0]: b0, qb0 = poses[idx+1,1:4], poses[idx+1, 4:8]

            p = position_interpolation(poses[idx-1,1:4], poses[idx, 1:4], t, a0, b0)
            q = orientation_interpolation(poses[idx - 1, 4:8], poses[idx, 4:8], t, qa0, qb0)
        return GroundTruth(dataset, np.concatenate(([stamp], p, q)))

    def __init__(self, dataset, data):
        if len(data) < 8:
            raise ValueError("Data must have at least 8 entries [time1, position3, orientation4] but has %d"% len(data))
        super().__init__(dataset, stamp=data[0], reference='world')
        self._data = data

    @property
    def position(self):
        return self._data[1:4]

    @property
    def quaternion(self):
        return self._data[4:8]

    @property
    def rotation(self):
        t = np.eye(4)
        t[0:3,0:3] = tf.quaternions.quat2mat(self.quaternion)
        return t

    @property
    def translation(self):
        t = np.eye(4)
        t[0:3,3] = self.position
        return t

    @property
    def pose(self):
        return tf.affines.compose(
            T=self.position,
            R=tf.quaternions.quat2mat(self.quaternion),
            Z=np.ones(3),
            S=np.zeros(3)
        )

    @property
    def _transform(self):
        return np.linalg.inv(self.pose)

    def stereo(self, shutter, extrapolation='closest'):
        return api.StereoImage.extrapolate(self, shutter, method=extrapolation)
