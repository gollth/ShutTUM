import unittest
import os.path as p
import numpy as np
import StereoTUM as api


class TestImuValue (unittest.TestCase):
    def setUp(self):
        d = p.dirname(p.realpath(__file__))
        self.path = p.join(d, 'valid')
        self.dataset = api.Dataset(self.path)

    def test_imuvalue_created_correctly(self):
        imu = api.ImuValue(self.dataset, [0, 0,0,-1, 0,0,0])
        self.assertIsInstance(imu.acceleration, np.ndarray)
        self.assertIsInstance(imu.angular_velocity, np.ndarray)

        self.assertTrue(np.allclose(imu.acceleration, np.array((0,0,-1))))
        self.assertTrue(np.allclose(imu.angular_velocity, np.array((0,0,0))))

    def test_stereo_closest_extrapolation(self):
        time = 0.175
        imu = api.ImuValue(self.dataset, [time, 0,0,-1, 0,0,0])
        stereo = imu.stereo('rolling', extrapolation='closest')
        delta = abs(stereo.stamp - time)
        for image in self.dataset.cameras('rolling'):
            if abs(image.stamp - time) < delta:
                self.fail("Found an image %s, which is closer to %s than %s" % (image, imu, stereo))

    def test_stereo_next_extrapolation(self):
        time = 0.16
        imu = api.ImuValue(self.dataset, [time, 0,0,-1, 0,0,0])
        stereo = imu.stereo('rolling', extrapolation='next')
        for image in self.dataset.cameras('rolling'):
            if image.stamp < time: continue
            self.assertEqual(image.ID, stereo.ID)
            return
        self.fail("Unknown reason")

    def test_stereo_next_extrapolation_returns_none_if_no_more_frames_exist(self):
        time = 200
        imu = api.ImuValue(self.dataset, [time, 0, 0, -1, 0, 0, 0])
        stereo = imu.stereo('rolling', extrapolation='next')
        self.assertIsNone(stereo)

    def test_stereo_prev_extrapolation(self):
        time = 0.19
        imu = api.ImuValue(self.dataset, [time, 0, 0, -1, 0, 0, 0])
        stereo = imu.stereo('rolling', extrapolation='prev')
        previous = None
        for image in reversed(self.dataset.cameras('rolling')):
            if image.stamp > time: continue
            previous = image
            break

        if previous is None: self.fail("No previous image to %s found" % imu)
        self.assertEqual(previous.ID, stereo.ID)

    def test_stereo_prev_extrapolation_returns_none_if_no_earier_frames_exist(self):
        time = 0
        imu = api.ImuValue(self.dataset, [time, 0,0,-1,0,0,0])
        stereo = imu.stereo('rolling', extrapolation='prev')
        self.assertIsNone(stereo)

    def test_stereo_exact_extrapolation(self):
        for observation in self.dataset.imu:
            images = [img for img in self.dataset.cameras('rolling') if img.stamp == observation.stamp]
            match = observation.stereo('rolling', extrapolation='exact')
            if not images: self.assertIsNone(match)
            else:          self.assertEqual(images[0].ID, match.ID)

    def test_stereo_extrapolation_raises_value_error_on_unknown_extrapolation_method(self):
        with self.assertRaises(ValueError):
            self.dataset.imu[0].stereo('rolling', extrapolation='unknown')

    def test_ground_truth_interpolation_is_correct(self):
        gti = np.genfromtxt(p.join(self.path, 'data', 'ground_truth_interpolated.csv'), skip_header=1)
        for observation in self.dataset.imu:
            gt = observation.groundtruth >> observation
            expected = gti[gti[:, 0] == observation.stamp, :]
            if expected.size == 0: self.fail("No interpolated gt in file found for time %.3f" % observation.stamp)
            expected = api.GroundTruth(self.dataset, expected[0])
            self.assertTrue(np.allclose(gt, expected.pose))

if __name__ == '__main__':
    unittest.main()
