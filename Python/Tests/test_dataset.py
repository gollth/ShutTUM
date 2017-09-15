import unittest
import yaml
import numpy     as np
import os.path   as p
import StereoTUM as api


class TestDataset (unittest.TestCase):
    def setUp(self):
        d = p.dirname(p.realpath(__file__))
        self._valid          = p.join(d, 'valid')
        self._missing_frames = p.join(d, 'missing_frames_folder')
        self._wrong_time     = p.join(d, 'wrong_time_param_file')

        self._time = {}
        with open(p.join(self._valid, 'params', 'time.yaml')) as stream:
            self._time = yaml.load(stream)

    def test_loads_valid_record_without_exception(self):
        try:
            api.Dataset(self._valid)
        except FileNotFoundError:
            self.fail('Loading with __init__ of a valid dataset failed')

    def test_loads_missing_frames_record_with_exception(self):
        with self.assertRaises(FileNotFoundError) as context:
            api.Dataset(self._missing_frames)

    def test_loads_wrong_time_param_record_with_exception(self):
        with self.assertRaises(ValueError) as context:
            api.Dataset(self._wrong_time)

    def test_raw_contains_correct_names(self):
        dataset = api.Dataset(self._valid)
        self.assertIsInstance(dataset.raw.imu,         np.ndarray)
        self.assertIsInstance(dataset.raw.groundtruth, np.ndarray)
        self.assertIsInstance(dataset.raw.frames,      np.ndarray)

        self.assertGreaterEqual(dataset.raw.frames.shape[1],      3, 'Dataset.Raw.Frames should have 3 or more columns')
        self.assertEqual       (dataset.raw.imu.shape[1],         7, 'Dataset.Raw.Imu should have 7 columns')
        self.assertEqual       (dataset.raw.groundtruth.shape[1], 8, 'Dataset.Raw.Groundtruth should have 8 columns')

    def test_times_matches_param_file(self):
        dataset = api.Dataset(self._valid)
        self.assertAlmostEqual(dataset.start,    self._time['time']['start'])
        self.assertAlmostEqual(dataset.end,      self._time['time']['end'])
        self.assertAlmostEqual(dataset.duration, self._time['time']['duration'])


    def test_gamma_lookup_matches_correct_value(self):
        dataset = api.Dataset(self._valid)
        black = dataset.gamma('cam1', 0)
        white = dataset.gamma('cam1', 255)
        grey  = dataset.gamma('cam1', 128)
        self.assertAlmostEqual(black,   0, delta=0.5)
        self.assertAlmostEqual(white, 255, delta=0.5)
        self.assertAlmostEqual(grey, 59.5, delta=0.5)

    def test_gamma_lookup_fails_for_wrong_cam_name(self):
        dataset = api.Dataset(self._valid)
        with self.assertRaises(ValueError):
            dataset.gamma('unknown cam', 100)

    def test_gamma_lookup_fails_for_outofbounds_inputs(self):
        dataset = api.Dataset(self._valid)
        with self.assertRaises(ValueError):
            dataset.gamma('cam1', -5)
        with self.assertRaises(ValueError):
            dataset.gamma('cam1', 260)

    def test_vignette_image_looks_correct(self):
        dataset = api.Dataset(self._valid)
        vig = dataset.vignette('cam1')
        self.assertTrue(vig is not None)
        self.assertIsInstance(vig, np.ndarray)

    def test_vignette_raises_on_wrong_camname(self):
        dataset = api.Dataset(self._valid)
        with self.assertRaises(ValueError):
            dataset.vignette('cam221')

    def test_rolling_shutter_speed_maches(self):
        dataset = api.Dataset(self._valid)
        self.assertAlmostEqual(dataset.rolling_shutter_speed, 0.01586, delta=0.001)

if __name__ == '__main__':
    unittest.main()
