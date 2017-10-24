import unittest
import os.path as p
import numpy as np
from StereoTUM import Interpolation
from StereoTUM.dataset import Dataset
from StereoTUM.values import GroundTruth


class TestGroundTruth(unittest.TestCase):
    def setUp(self):
        d = p.dirname(p.realpath(__file__))
        self.dataset = Dataset(p.join(d, 'valid'))
        self.raw = self.dataset.raw.groundtruth[0, :]
        self.gt = GroundTruth(self.dataset, self.raw)

    def test_groundtruth_raises_error_with_too_few_data(self):
        with self.assertRaises(ValueError):
            GroundTruth(self.dataset, [1,2,3,4,5,6])

    def test_groundtruth_created_correctly(self):
        self.assertListEqual(list(self.gt.position), list(self.raw[1:4]))
        self.assertListEqual(list(self.gt.quaternion), list(self.raw[4:8]))
        self.assertEqual(self.gt.stamp, self.raw[0])

    def test_mocap_iteration_is_possible(self):
        self.assertGreater(len(self.dataset.mocap), 0)
        for _ in self.dataset.mocap:
            self.assertTrue(True)
            return

        self.fail('Iterated zero times through imu')

    def test_dt_is_correct(self):
        gt2 = GroundTruth(self.dataset, self.dataset.raw.groundtruth[1, :])
        self.assertEqual(gt2.dt, gt2.stamp - self.gt.stamp)

    def test_rotation_times_translation_equals_pose(self):
        t = self.gt.translation.dot(self.gt.rotation)
        self.assertTrue(np.allclose(t, self.gt.pose))

    def test_tf_lookup_from_cam1_is_correct(self):
        ta = self.gt << 'cam1'
        tb = self.gt << self.dataset.cameras('rolling')[0].L.reference

        expected = GroundTruth(self.dataset, [0.0, -0.4032,0.5288,1.3405,0.6874,-0.7259,-0.0069,0.0234])
        expected = np.linalg.inv(expected.pose)

        self.assertIsInstance(ta, np.ndarray)
        self.assertIsInstance(tb, np.ndarray)
        self.assertTrue(np.allclose(ta, tb))
        self.assertTrue(np.allclose(ta, expected))
        
    def test_tf_lookup_to_cam1_is_correct(self):
        ta = self.gt >> 'cam1'
        tb = self.gt >> self.dataset.cameras('rolling')[0].L.reference

        expected = GroundTruth(self.dataset, [0.0, -0.4032, 0.5288, 1.3405, 0.6874, -0.7259, -0.0069, 0.0234])

        self.assertIsInstance(ta, np.ndarray)
        self.assertIsInstance(tb, np.ndarray)
        self.assertTrue(np.allclose(ta, tb))
        self.assertTrue(np.allclose(ta, expected.pose))

    def test_image_lookup_closest_extrapolation_is_correct(self):
        img1 = self.dataset.cameras('rolling')[0]
        img2 = self.dataset.cameras('rolling')[1]

        # Create an Ground Truth pose, which is at 80% between img1 and img2
        t  = Interpolation.linear(img1.stamp, img2.stamp, .8)
        gt = GroundTruth.interpolate(self.dataset, t)

        # Check if the extrapolated image, is really to the closer one (i.e. img2)
        stereo = gt.stereo('rolling', extrapolation='closest')
        self.assertEqual   (stereo.ID, img2.ID)
        self.assertNotEqual(stereo.ID, img1.ID)

    def test_image_lookup_next_extrapolation_is_correct(self):
        img1 = self.dataset.cameras('rolling')[0]
        img2 = self.dataset.cameras('rolling')[1]

        # Create a Ground truth pose which is at 20% between img1 and img2
        t  = Interpolation.linear(img1.stamp, img2.stamp, .2)
        gt = GroundTruth.interpolate(self.dataset, t)

        # Check if the extrapolated image is really to the next one (i.e. img2 not img1)
        stereo = gt.stereo('rolling', extrapolation='next')
        self.assertEqual   (stereo.ID, img2.ID)
        self.assertNotEqual(stereo.ID, img1.ID)

    def test_image_lookup_prev_extrapolation_is_correct(self):
        img1 = self.dataset.cameras('rolling')[0]
        img2 = self.dataset.cameras('rolling')[1]

        # Create a Ground truth pose which is at 80% between img1 and img2
        t = Interpolation.linear(img1.stamp, img2.stamp, .8)
        gt = GroundTruth.interpolate(self.dataset, t)

        # Check if the extrapolated image is really to the prev one (i.e. img1 not img2)
        stereo = gt.stereo('rolling', extrapolation='prev')
        self.assertEqual   (stereo.ID, img1.ID)
        self.assertNotEqual(stereo.ID, img2.ID)

    def test_image_lookup_exact_extrapolation_fails_with_nones(self):
        img1 = self.dataset.cameras('rolling')[0]
        img2 = self.dataset.cameras('rolling')[1]

        # Create a Ground truth pose which is somewhere between img1 & img2
        t = Interpolation.linear(img1.stamp, img2.stamp, .5)
        gt = GroundTruth.interpolate(self.dataset, t)

        stereo = gt.stereo('rolling', extrapolation='exact')
        self.assertIsNone(stereo)

if __name__ == '__main__':
    unittest.main()
