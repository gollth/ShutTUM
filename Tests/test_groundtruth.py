import unittest
import os.path as p
import numpy as np
from StereoTUM import Interpolation
from StereoTUM.sequence import Sequence
from StereoTUM.values import GroundTruth


class TestGroundTruth(unittest.TestCase):
    def setUp(self):
        d = p.dirname(p.realpath(__file__))
        self.sequence = Sequence(p.join(d, 'valid'))
        self.raw = self.sequence.raw.groundtruth[0, :]
        self.gt = GroundTruth(self.sequence, self.raw)

    def test_groundtruth_raises_error_with_too_few_data(self):
        with self.assertRaises(ValueError):
            GroundTruth(self.sequence, [1, 2, 3, 4, 5, 6])

    def test_groundtruth_created_correctly(self):
         self.assertEqual(self.gt.stamp, self.raw[0])

    def test_groundtruth_length_matches_raw_value(self):
        n1 = len(list(self.sequence.mocap))
        n2 = sum([1 for value in self.sequence.mocap])
        n  = self.sequence.raw.groundtruth.shape[0]
        self.assertEqual(n, n1)
        self.assertEqual(n, n2)

    def test_mocap_iteration_is_possible(self):
        self.assertGreater(len([self.sequence.mocap]), 0)
        for _ in self.sequence.mocap:
            self.assertTrue(True)
            return

        self.fail('Iterated zero times through imu')

    def test_dt_is_correct(self):
        gt2 = GroundTruth(self.sequence, self.sequence.raw.groundtruth[1, :])
        self.assertEqual(gt2.dt(), gt2.stamp - self.gt.stamp)

    def test_tf_lookup_from_marker_is_correct(self):
        ta = self.gt << 'marker'
        expected = GroundTruth(self.sequence, [0.0, -0.4032, 0.5288, 1.3405, 0.6874, -0.7259, -0.0069, 0.0234]) << 'marker'

        self.assertIsInstance(ta, np.ndarray)
        self.assertTrue(np.allclose(ta, expected))
        
    def test_tf_lookup_to_marker_is_correct(self):
        ta = self.gt >> 'marker'
        expected = GroundTruth(self.sequence, [0.0, -0.4032, 0.5288, 1.3405, 0.6874, -0.7259, -0.0069, 0.0234]) >> 'marker'

        self.assertIsInstance(ta, np.ndarray)
        self.assertTrue(np.allclose(ta, expected))

    def test_image_lookup_closest_extrapolation_is_correct(self):
        img1 = self.sequence.cameras('rolling')[1]
        img2 = self.sequence.cameras('rolling')[2]

        # Create an Ground Truth pose, which is at 80% between img1 and img2
        t  = Interpolation.linear(img1.stamp, img2.stamp, .8)
        gt = GroundTruth.interpolate(self.sequence, t)

        # Check if the extrapolated image, is really to the closer one (i.e. img2)
        stereo = gt.stereo('rolling', extrapolation='closest')
        self.assertEqual   (stereo.ID, img2.ID)
        self.assertNotEqual(stereo.ID, img1.ID)

    def test_image_lookup_next_extrapolation_is_correct(self):
        img1 = self.sequence.cameras('rolling')[1]
        img2 = self.sequence.cameras('rolling')[2]

        # Create a Ground truth pose which is at 20% between img1 and img2
        t  = Interpolation.linear(img1.stamp, img2.stamp, .2)
        gt = GroundTruth.interpolate(self.sequence, t)

        # Check if the extrapolated image is really to the next one (i.e. img2 not img1)
        stereo = gt.stereo('rolling', extrapolation='next')
        self.assertEqual   (stereo.ID, img2.ID)
        self.assertNotEqual(stereo.ID, img1.ID)

    def test_image_lookup_prev_extrapolation_is_correct(self):
        img1 = self.sequence.cameras('rolling')[1]
        img2 = self.sequence.cameras('rolling')[2]

        # Create a Ground truth pose which is at 80% between img1 and img2
        t = Interpolation.linear(img1.stamp, img2.stamp, .8)
        gt = GroundTruth.interpolate(self.sequence, t)

        # Check if the extrapolated image is really to the prev one (i.e. img1 not img2)
        stereo = gt.stereo('rolling', extrapolation='prev')
        self.assertEqual   (stereo.ID, img1.ID)
        self.assertNotEqual(stereo.ID, img2.ID)

    def test_image_lookup_exact_extrapolation_fails_with_nones(self):
        img1 = self.sequence.cameras('rolling')[1]
        img2 = self.sequence.cameras('rolling')[2]

        # Create a Ground truth pose which is somewhere between img1 & img2
        t = Interpolation.linear(img1.stamp, img2.stamp, .5)
        gt = GroundTruth.interpolate(self.sequence, t)

        stereo = gt.stereo('rolling', extrapolation='exact')
        self.assertIsNone(stereo)

    def test_marker_cam_offset_is_4x4_array(self):
        cam_marker = self.gt.marker << 'cam1'
        self.assertTupleEqual(cam_marker.shape, (4,4))

    def test_marker_cam_offset_is_correct(self):
        expected = np.array(([1.0, 0.0, 0.0, 0.5],
                             [0.0, 1.0, 0.0, 0.6],
                             [0.0, 0.0, 1.0, 0.7],
                             [0.0, 0.0, 0.0, 1.0]))
        actual1 = self.gt.marker << 'cam1'
        actual2 = self.gt.marker >> 'cam1'
        self.assertTrue(np.allclose(expected, actual1))
        self.assertTrue(np.allclose(np.linalg.inv(expected), actual2))

if __name__ == '__main__':
    unittest.main()
