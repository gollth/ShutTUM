import unittest
import os.path   as p
import numpy     as np
from StereoTUM.dataset import Dataset
from StereoTUM.values import Value

class TestValue(unittest.TestCase):

    def setUp(self):
        d = p.dirname(p.realpath(__file__))
        self.dataset = Dataset(p.join(d, 'valid'))

    def test_unknown_reference_raises_value_error(self):
        with self.assertRaises(ValueError):
            Value(self.dataset, 0, 'some unknown frame')

    def test_reference_set_correctly(self):
        value = Value(self.dataset, 14.145, 'cam1')
        self.assertEqual(value.reference, 'cam1')

    def test_stamp_set_correctly(self):
        value = Value(self.dataset, 14.145, 'cam1')
        self.assertEqual(value.stamp, 14.145)

    def test_wrong_reference_raises_value_error(self):
        value = Value(self.dataset, 14.145, 'cam1')
        with self.assertRaises(ValueError):
            value >> 'some unknown frame'

    def test_tf_lookup_with_int_type_raises_type_error(self):
        value = Value(self.dataset, 0, 'cam1')
        with self.assertRaises(TypeError):
            value >> 17
        with self.assertRaises(TypeError):
            value << 17

    def test_tf_lookup_with_list_type_raises_type_error(self):
        value = Value(self.dataset, 0, 'cam1')
        with self.assertRaises(TypeError):
            value >> ['hello', 'world']
        with self.assertRaises(TypeError):
            value << ['hello', 'world']

    def test_self_tf_lookup_is_correct(self):
        cam1 = Value(self.dataset, 0, 'cam1')
        identity = cam1 >> 'cam1'
        self.assertTrue(np.allclose(identity, np.eye(4)))

    def test_cam1_to_cam2_tf_lookup_is_correct(self):
        cam1 = Value(self.dataset, 0, 'cam1')
        cam2 = Value(self.dataset, 0, 'cam2')

        expected_transform = np.array([
            [0.0,-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0]]
        )
        actual_transform = cam1 >> cam2
        self.assertTrue(np.allclose(actual_transform, expected_transform))

    def test_cam2_to_imu_tf_lookup_is_correct(self):
        cam2 = Value(self.dataset, 0, 'cam2')
        imu  = Value(self.dataset, 0, 'imu')

        expected_transform = np.array([
            [ 0.0, 1.0, 0.0, 0.3],
            [-1.0, 0.0, 0.0,-0.2],
            [ 0.0, 0.0, 1.0,-0.1],
            [ 0.0, 0.0, 0.0, 1.0]]
        )
        actual_transform1 = cam2 >> imu
        actual_transform2 = imu << cam2
        self.assertTrue(np.allclose(actual_transform1, expected_transform))
        self.assertTrue(np.allclose(actual_transform2, expected_transform))

if __name__ == '__main__':
    unittest.main()
