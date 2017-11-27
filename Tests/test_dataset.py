import unittest
import yaml
import numpy     as np
import os.path   as p
from StereoTUM.dataset import Dataset
from collections import Iterable


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
            Dataset(self._valid)
        except IOError:
            self.fail('Loading with __init__ of a valid dataset failed')

    def test_loads_missing_frames_record_with_exception(self):
        with self.assertRaises(IOError) as context:
            Dataset(self._missing_frames)

    def test_loads_wrong_time_param_record_with_exception(self):
        with self.assertRaises(ValueError) as context:
            Dataset(self._wrong_time)

    def test_raw_contains_correct_names(self):
        dataset = Dataset(self._valid)
        self.assertIsInstance(dataset.raw.imu,         np.ndarray)
        self.assertIsInstance(dataset.raw.groundtruth, np.ndarray)
        self.assertIsInstance(dataset.raw.frames,      np.ndarray)

        self.assertGreaterEqual(dataset.raw.frames.shape[1],      3, 'Dataset.Raw.Frames should have 3 or more columns')
        self.assertEqual       (dataset.raw.imu.shape[1],         7, 'Dataset.Raw.Imu should have 7 columns')
        self.assertEqual       (dataset.raw.groundtruth.shape[1], 8, 'Dataset.Raw.Groundtruth should have 8 columns')

    def test_times_matches_param_file(self):
        dataset = Dataset(self._valid)
        self.assertAlmostEqual(dataset.start,    self._time['time']['start'])
        self.assertAlmostEqual(dataset.end,      self._time['time']['end'])
        self.assertAlmostEqual(dataset.duration, self._time['time']['duration'])


    def test_gamma_lookup_matches_correct_value(self):
        dataset = Dataset(self._valid)
        black = dataset.gamma('cam1', 0)
        white = dataset.gamma('cam1', 255)
        grey  = dataset.gamma('cam1', 128)
        self.assertAlmostEqual(black,   0, delta=0.5)
        self.assertAlmostEqual(white, 255, delta=0.5)
        self.assertAlmostEqual(grey, 59.5, delta=0.5)

    def test_gamma_lookup_fails_for_wrong_cam_name(self):
        dataset = Dataset(self._valid)
        with self.assertRaises(ValueError):
            dataset.gamma('unknown cam', 100)

    def test_gamma_lookup_fails_for_outofbounds_inputs(self):
        dataset = Dataset(self._valid)
        with self.assertRaises(ValueError):
            dataset.gamma('cam1', -5)
        with self.assertRaises(ValueError):
            dataset.gamma('cam1', 260)

    def test_vignette_image_looks_correct(self):
        dataset = Dataset(self._valid)
        vig = dataset.vignette('cam1')
        self.assertTrue(vig is not None)
        self.assertIsInstance(vig, np.ndarray)

    def test_vignette_raises_on_wrong_camname(self):
        dataset = Dataset(self._valid)
        with self.assertRaises(ValueError):
            dataset.vignette('cam221')

    def test_rolling_shutter_speed_maches(self):
        dataset = Dataset(self._valid)
        self.assertAlmostEqual(dataset.rolling_shutter_speed, 0.01586, delta=0.001)

    def test_camera_iteration_with_filter(self):
        dataset = Dataset(self._valid)
        cam = dataset.cameras('rolling')
        x = filter(lambda item: item.ID == 2, cam)
        self.assertEqual(list(x)[0].ID, 2)

    def test_camera_iteration_with_list_comprehension(self):
        dataset = Dataset(self._valid)
        wanted_id = 2
        x = [ image.stamp for image in dataset.cameras('rolling') if image.ID == wanted_id]
        self.assertEqual(x[0], dataset.cameras('rolling')[wanted_id].stamp)

    def test_times_contain_framestamps(self):
        dataset = Dataset(self._valid)
        g = set(img.stamp for img in dataset.cameras('global'))
        r = set(img.stamp for img in dataset.cameras('rolling'))
        frames = g | r

        self.assertTrue(frames < set(dataset.times))    # check if timestamp from frames is subset of all stamps

    def test_times_contain_imustamps(self):
        dataset = Dataset(self._valid)
        i = set(imu.stamp for imu in dataset.imu)
        self.assertTrue(i < set(dataset.times))

    def test_times_contain_groundtruthstamps(self):
        dataset = Dataset(self._valid)
        g = set(gt.stamp for gt in dataset.mocap)
        self.assertTrue(g < set(dataset.times))

    def test_large_time_is_not_in_times(self):
        dataset = Dataset(self._valid)
        x = { 7751621, 2951, -0.1 }
        self.assertFalse(x < set(dataset.times))

    def test_iteration_over_times_yields_times(self):
        dataset = Dataset(self._valid)
        times = dataset.times
        for i, time in enumerate(dataset.times):
            if time != times[i]: self.fail("Time %s (number %d) did not match iterated item %s" % (time, i, times[i]))

    def test_times_are_sorted(self):
        dataset = Dataset(self._valid)
        self.assertListEqual(dataset.times, sorted(dataset.times))

    def test_lookup_returns_image(self):
        dataset = Dataset(self._valid)
        A = dataset.cameras('global')[2]
        data = dataset[A.stamp]
        self.assertIsNotNone(data.global_)
        self.assertIsNotNone(data.rolling)
        self.assertIsNotNone(data.imu)
        self.assertIsNone(data.groundtruth)
        self.assertEqual(A, data.global_)
        self.assertNotEqual(A, data.rolling)

    def test_lookup_returns_imu(self):
        dataset = Dataset(self._valid)
        A = next(dataset.imu)   # get the first value
        data = dataset[A.stamp]
        self.assertIsNotNone(data.imu)
        self.assertIsNone(data.global_)
        self.assertIsNone(data.rolling)
        self.assertIsNone(data.groundtruth)
        self.assertEqual(A, data.imu)

    def test_lookup_returns_gt(self):
        dataset = Dataset(self._valid)
        A = dataset.mocap[1]
        data = dataset[A.stamp]
        self.assertIsNotNone(data.groundtruth)
        self.assertIsNone(data.global_)
        self.assertIsNone(data.rolling)
        self.assertIsNone(data.imu)
        self.assertEqual(A, data.groundtruth)

    def test_lookup_returns_only_nones_for_invalid_timestamp(self):
        dataset = Dataset(self._valid)
        data = dataset[11547]
        self.assertIsNone(data.global_)
        self.assertIsNone(data.rolling)
        self.assertIsNone(data.imu)
        self.assertIsNone(data.groundtruth)

    def test_slice_lookup_returns_more_then_one_element(self):
        dataset = Dataset(self._valid)
        datas = dataset[.1:.2]
        self.assertIsInstance(datas, Iterable)
        self.assertGreaterEqual(len([datas]), 1)

    def test_slicing_with_step_raises_valueerror(self):
        dataset = Dataset(self._valid)
        with self.assertRaises(ValueError) as context:
            x = dataset[1:2:.5]

    def test_inverse_slicing_returns_empty(self):
        dataset = Dataset(self._valid)
        datas = dataset[2:1]
        self.assertEqual(sum(1 for i in datas), 0)

    def test_none_end_slicing_will_yield_all_values_till_end(self):
        dataset = Dataset(self._valid)
        datas = dataset[.1:]
        E  = dataset[dataset.end]   # get the last element of the actual dataset
        A  = list(datas)[-1]        # get the last element of the generator

        self.assertEqual(E.groundtruth, A.groundtruth)


if __name__ == '__main__':
    unittest.main()
