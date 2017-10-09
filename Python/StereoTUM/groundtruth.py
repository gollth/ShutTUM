import numpy as np
import transforms3d as tf
import StereoTUM as api


class GroundTruth (api.Value):
    r"""
    A ground truth is a :any:`Value` with the reference ``"world"``.
    The ground truth is taken with a higher frequency than all other values (around 100 Hz), but since the 
    :any:`Mocap` system is stationary in one room only, it might not cover the whole duration of the datset 
    (depending on the record).
    """

    @staticmethod
    def interpolate(dataset, stamp, position_interpolation=api.Interpolation.linear,
                                    orientation_interpolation=api.Interpolation.slerp):
        r"""
        This function enables you to find the interpolated groundtruth of a record given a certain timestamp.
        
        :param dataset: the dataset which holds all ground truth values to interpolate over 
        :param float stamp: the time at which to interpolate (in seconds, with decimal places) 
        :param position_interpolation: A predefined or custom interpolation function
        :param orientation_interpolation: A predefined or custom interpolation function
        :return: A :any:`GroundTruth` -Value
        
        .. seealso:: :any:`Interpolation`
        """
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
        r"""The position of this ground truth as 3D `ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_"""
        return self._data[1:4]

    @property
    def quaternion(self):
        r"""The orientation in quaternion representation with the scalar (w) component as first element in a 4D `ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_"""
        return self._data[4:8]

    @property
    def rotation(self):
        r"""The rotation matrix only **WITHOUT** the translational part as 4x4 `ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_ """
        t = np.eye(4)
        t[0:3,0:3] = tf.quaternions.quat2mat(self.quaternion)
        return t

    @property
    def translation(self):
        r"""The translation matrix only **WITHOUT** the rotational part as 4x4 `ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_"""
        t = np.eye(4)
        t[0:3,3] = self.position
        return t

    @property
    def pose(self):
        r"""The complete pose of the ground truth including both translation and orientation"""
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
        r"""
        Find a matching stereo image pair for this ground truth value.
        
        :param shutter: The shutter type of the images to find (``"global"``, ``"rolling"``, **not** ``"both"``) 
        :param extrapolation: An optional extrapolation method to determine the rules for a "match" (one of ``"closest"``, ``"next"``, ``"prev"``, ``"exact"``)
        :return: The matching stereo image or None if no was found
        
        .. seealso:: :any:`StereoTUM.StereoImage.extrapolate`
        """
        return api.StereoImage.extrapolate(self, shutter, method=extrapolation)

