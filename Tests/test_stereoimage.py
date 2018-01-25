import unittest
import math
import numpy     as np
import os.path   as p
from StereoTUM.sequence import Sequence
from StereoTUM.values import StereoImage, GroundTruth
from collections import Iterable


class TestStereoImage (unittest.TestCase):
    def setUp(self):
        d = p.dirname(p.realpath(__file__))
        self.path = p.join(d, 'valid')
        self.sequence = Sequence(self.path)
        self.data = self.sequence.raw.frames[0, :]
        self.stereo = StereoImage(self.sequence, self.data, 'rolling')

    def test_stereoimage_created_correctly(self):
        self.assertEqual(self.stereo.L.reference, 'cam1')
        self.assertEqual(self.stereo.R.reference, 'cam2')

        self.assertEqual(self.stereo.L.ID, self.data[1])
        self.assertEqual(self.stereo.R.ID, self.data[1])
        self.assertEqual(self.stereo.stamp, self.data[0])
        self.assertEqual(self.stereo.stamp, self.stereo.L.stamp)
        self.assertEqual(self.stereo.stamp, self.stereo.R.stamp)

    def test_dt_is_correct(self):
        for shutter in ['global', 'rolling']:
            img1 = StereoImage(self.sequence, self.sequence.raw.frames[0, :], shutter)
            img2 = StereoImage(self.sequence, self.sequence.raw.frames[1, :], shutter)
            self.assertEqual(img2.dt(), img2.stamp - img1.stamp)

    def test_timestamp_matches_frame_id(self):
        for image in self.sequence.cameras('rolling'):
            if image.ID == self.stereo.ID:
                self.assertEqual(self.stereo.stamp, image.stamp)
                return

        self.fail()

    def test_distortion_models(self):
        for shutter in ['global', 'rolling']:
            stereo = StereoImage(self.sequence, self.sequence.raw.frames[0, :], shutter)
            omega_l = stereo.L.distortion(model='fov')
            omega_r = stereo.R.distortion(model='fov')
            self.assertIsInstance(omega_l, float)
            self.assertIsInstance(omega_r, float)

            radtan_l = stereo.L.distortion(model='radtan')
            radtan_r = stereo.R.distortion(model='radtan')
            self.assertIsInstance(radtan_l, Iterable)
            self.assertIsInstance(radtan_r, Iterable)

            with self.assertRaises(ValueError) as ctx:
                stereo.L.distortion(model='what???')

    def test_opposite_image_finds_L_when_asked_on_R(self):
        self.assertIs(self.stereo.L.opposite, self.stereo.R)

    def test_opposite_image_find_R_when_asked_on_L(self):
        self.assertIs(self.stereo.R.opposite, self.stereo.L)

    def test_exposure_is_equal_in_both_images(self):
        self.assertEqual(self.stereo.L.exposure, self.stereo.R.exposure)
        limits = self.sequence.exposure_limits
        self.assertGreaterEqual(self.stereo.L.exposure, limits.min)
        self.assertLessEqual   (self.stereo.L.exposure, limits.max)

    def test_image_path_is_correct(self):
        expected_left = p.join(self.path, 'frames', 'cam1', '00001.jpeg')
        expected_right = p.join(self.path, 'frames', 'cam2', '00001.jpeg')
        self.assertEqual(self.stereo.L.path, expected_left)
        self.assertEqual(self.stereo.R.path, expected_right)

    def test_image_loading_returns_ndarray(self):
        image = self.stereo.L.load()
        self.assertIsInstance(image, np.ndarray)
        self.assertTupleEqual(image.shape, (1024, 1280))

    def test_correct_match_between_image_and_imu(self):
        self.assertEqual(self.stereo.imu.stamp, self.stereo.stamp)
        self.assertEqual(self.stereo.L.imu.stamp, self.stereo.stamp)

    def test_ground_truth_match_is_correctly_interpolated(self):
        gti = np.genfromtxt(p.join(self.path,'data', 'ground_truth_interpolated.csv'), skip_header=1)
        for image in self.sequence.cameras('rolling'):
            expected = gti[gti[:, 0] == image.stamp, :]
            gt = image.L.groundtruth(max_stamp_delta=20e-3)
            if expected.size == 0:
                self.assertIsNone(gt, 'expected that ground truth at stamp %f is none' % image.stamp)
                continue

            actual = gt >> 'cam1'
            expected = GroundTruth(self.sequence, expected[0]) >> 'cam1'
            self.assertTrue(np.allclose(actual, expected))

    def test_ground_truth_is_correct_for_both_images(self):
        wl = self.stereo.L.groundtruth() >> 'cam1'
        wr = self.stereo.R.groundtruth() >> 'cam2'
        rl = self.stereo.L << self.stereo.R

        wl2 = wr.dot(rl)

        self.assertFalse(np.allclose(wl, wr))
        self.assertTrue(np.allclose(wl, wl2))

    def test_illuminance_is_float_normally(self):
        img = self.sequence.cameras('rolling')[2]
        expected = self.data[3]
        actual = img.illuminance
        self.assertEqual(expected, actual)
        self.assertIsInstance(actual, float)

    def test_illuminance_is_nan_if_n_a(self):
        img = self.sequence.cameras('rolling')[4]
        lx = img.illuminance
        self.assertTrue(math.isnan(lx), "N/A illuminance is not nan but %s" % lx);


if __name__ == '__main__':
    unittest.main()
