import unittest
import numpy     as np
import os.path   as p
from StereoTUM.dataset import Dataset
from StereoTUM.values import StereoImage, GroundTruth


class TestStereoImage (unittest.TestCase):
    def setUp(self):
        d = p.dirname(p.realpath(__file__))
        self.path = p.join(d, 'valid')
        self.dataset = Dataset(self.path)
        self.data = self.dataset.raw.frames[0,:]
        self.stereo = StereoImage(self.dataset, self.data , 'rolling')

    def test_stereoimage_created_correctly(self):
        self.assertEqual(self.stereo.L.reference, 'cam1')
        self.assertEqual(self.stereo.R.reference, 'cam2')

        self.assertEqual(self.stereo.L.ID, self.data[0])
        self.assertEqual(self.stereo.R.ID, self.data[0])
        self.assertEqual(self.stereo.stamp, self.data[1])
        self.assertEqual(self.stereo.stamp, self.stereo.L.stamp)
        self.assertEqual(self.stereo.stamp, self.stereo.R.stamp)

    def test_timestamp_matches_frame_id(self):
        for image in self.dataset.cameras('rolling'):
            if image.ID == self.stereo.ID:
                self.assertEqual(self.stereo.stamp, image.stamp)
                return

        self.fail()

    def test_opposite_image_finds_L_when_asked_on_R(self):
        self.assertIs(self.stereo.L.opposite, self.stereo.R)

    def test_opposite_image_find_R_when_asked_on_L(self):
        self.assertIs(self.stereo.R.opposite, self.stereo.L)

    def test_exposure_is_equal_in_both_images(self):
        self.assertEqual(self.stereo.L.exposure, self.stereo.R.exposure)
        limits = self.dataset.exposure_limits
        self.assertGreaterEqual(self.stereo.L.exposure, limits.min)
        self.assertLessEqual   (self.stereo.L.exposure, limits.max)

    def test_image_path_is_correct(self):
        expected_left = p.join(self.path, 'frames', 'cam1', '00002.jpeg')
        expected_right = p.join(self.path, 'frames', 'cam2', '00002.jpeg')
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
        for image in self.dataset.cameras('rolling'):
            gt = image.L.groundtruth() >> image.L
            expected = gti[gti[:,0] == image.stamp,:]
            if expected.size == 0: self.fail("No interpolated gt in file found for time %.3f" % image.stamp)
            expected = GroundTruth(self.dataset, expected[0])
            self.assertTrue(np.allclose(gt, expected.pose))

    def test_ground_truth_is_correct_for_both_images(self):
        wl = self.stereo.L.groundtruth().pose
        wr = self.stereo.R.groundtruth().pose
        rl = self.stereo.L << self.stereo.R

        wl2 = wr.dot(rl)

        self.assertFalse(np.allclose(wl, wr))
        self.assertTrue(np.allclose(wl, wl2))

    def test_framedrop_will_raise_value_error(self):
        with self.assertRaises(ValueError) as context:
            StereoImage(self.dataset, [117, 100, 1], "rolling")



if __name__ == '__main__':
    unittest.main()
