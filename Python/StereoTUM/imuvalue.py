import numpy as np
import StereoTUM as api


class ImuValue (api.Value):

    @staticmethod
    def interpolate(dataset, stamp, accelaration_interpolation=api.Interpolation.linear, angvelo_interpolation=api.Interpolation.linear):
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
        return self._acc

    @property
    def angular_velocity(self):
        return self._gyro

    def stereo(self, shutter, extrapolation='closest'):
        return api.StereoImage.extrapolate(self, shutter, method=extrapolation)

    @property
    def groundtruth(self):
        return api.GroundTruth.interpolate(self._dataset, self.stamp)
