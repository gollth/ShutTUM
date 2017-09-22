import numpy as np
import StereoTUM as api


class ImuValue (api.Value):
    r""" An ImuValue represents the measurement of the :class:`Imu` at a specific time.
    
    Since it is a :class:`Value` you can use it to calculate transforms with it. Also 
    the :class:`Imu` is synchronized in a way, that it measures exactly three times 
    per image. Any ImuValue consist of three acceleration measurements in X,Y and Z 
    and three angular velocity measurements around the X, Y, and Z axis.
    
    .. seealso:: :class:`Interpolation`
    """
    @staticmethod
    def interpolate(dataset, stamp, accelaration_interpolation=api.Interpolation.linear, angvelo_interpolation=api.Interpolation.linear):
        r"""
        This function enables you to find the interpolated imu values of a record given a certain timestamp.
        
        :param dataset: the dataset which holds all imu values to interpolate over 
        :param float stamp: the time at which to interpolate (in seconds, with decimal places) 
        :param accelaration_interpolation: A predefined or custom interpolation function
        :param angvelo_interpolation: A predefined or custom interpolation function
        :return: A :class:`ImuValue`
        
        .. seealso:: :class:`Interpolation`
        """

        imu = dataset.raw.imu
        idx = np.searchsorted(imu[:, 0], stamp)
        acc = np.ones((1, 3)) * np.nan
        gyr = np.ones((1, 4)) * np.nan
        if idx != 0 and idx != imu.shape[0]:
            ta = accelaration_interpolation(imu[idx-1, 0], imu[idx, 0], stamp)
            acc = (1 - ta) * imu[idx - 1, 1:4] + ta * imu[idx, 1:4]
            tg = angvelo_interpolation(imu[idx-1, 0], imu[idx, 0], stamp)
            gyr = (1 - tg) * imu[idx - 1, 1:4] + tg * imu[idx, 1:4]

        return ImuValue(dataset, np.concatenate(([stamp], acc, gyr)))

    def __init__(self, dataset, data):
        if len(data) < 7: raise ValueError("Data must have at least 7 entries [time1, acc3, gyro3] but has %d" % len(data))
        super().__init__(dataset, stamp=data[0], reference='imu')
        self._acc = np.array(data[1:4])
        self._gyro = np.array(data[4:7])

    @property
    def acceleration(self):
        r"""The acceleration 3D vector [x,y,z] of this measurement as numpy.ndarray"""
        return self._acc

    @property
    def angular_velocity(self):
        r"""The angular velocity 3D vector around [x,y,z] of this measurement as numpy.ndarray"""
        return self._gyro

    def stereo(self, shutter, extrapolation='closest'):
        r"""
        The matching stereo image for this imu measurement
        :param shutter: The shutter type of the images to find ("global", "rolling", **not** "both") 
        :param extrapolation: An optional extrapolation method to determine the rules for a "match":
            * **"closest"**: the image with the least difference to value.stamp is chosen
            * **"next"**: the image with the next larger time stamp than value.stamp is chosen
            * **"prev"**: the image with the next smaller time stamp than value.stamp is chosen
            * **"exact"**: the image where value.stamp == image.stamp holds is chosen, None otherwise
        :return: The matching stereo image or None if no was found
        
        .. seealso:: :func:`StereoImage.extrapolate`
        """
        return api.StereoImage.extrapolate(self, shutter, method=extrapolation)

    def groundtruth(self, position_interpolation=api.Interpolation.linear,
                          orientation_interpolation=api.Interpolation.slerp):
        r"""
        Find the matching :class:`GroundTruth` value for this imu value. Since the motion capture system and the :class:`Imu`
        are not synced, we need to interpolate between ground truths by time stamp of the imu value.
        
        :param position_interpolation: a predefined or custom interpolation function to interpolate positions 
        :param orientation_interpolation: a predefined or custom interpolation function to interpolate quaternions
        :return: the matching interpolated ground truth
         
        .. seealso:: :func:`GroundTruth.interpolate`
        """
        return api.GroundTruth.interpolate(self._dataset, self.stamp, position_interpolation, orientation_interpolation)
