import unittest
import yaml
import numpy     as np
import os.path   as p
from ShutTUM.sequence import Sequence
from collections import Iterable


class TestSequence (unittest.TestCase):
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
            Sequence(self._valid)
        except IOError:
            self.fail('Loading with __init__ of a valid sequence failed')

    def test_loads_missing_frames_record_with_exception(self):
        with self.assertRaises(IOError) as context:
            Sequence(self._missing_frames)

    def test_loads_wrong_time_param_record_with_exception(self):
        with self.assertRaises(ValueError) as context:
            Sequence(self._wrong_time)

    def test_raw_contains_correct_names(self):
        sequence = Sequence(self._valid)
        self.assertIsInstance(sequence.raw.imu,         np.ndarray)
        self.assertIsInstance(sequence.raw.groundtruth, np.ndarray)
        self.assertIsInstance(sequence.raw.frames,      np.ndarray)

        self.assertGreaterEqual(sequence.raw.frames.shape[1],      3, 'Sequence.Raw.Frames should have 3 or more columns')
        self.assertEqual       (sequence.raw.imu.shape[1],         7, 'Sequence.Raw.Imu should have 7 columns')
        self.assertEqual       (sequence.raw.groundtruth.shape[1], 8, 'Sequence.Raw.Groundtruth should have 8 columns')

    def test_times_matches_param_file(self):
        sequence = Sequence(self._valid)
        self.assertAlmostEqual(sequence.start,    self._time['time']['start'])
        self.assertAlmostEqual(sequence.end,      self._time['time']['end'])
        self.assertAlmostEqual(sequence.duration, self._time['time']['duration'])


    def test_gamma_lookup_matches_correct_value(self):
        sequence = Sequence(self._valid)
        black = sequence.gamma('cam1', 0)
        white = sequence.gamma('cam1', 255)
        grey  = sequence.gamma('cam1', 128)
        self.assertAlmostEqual(black,   0, delta=0.5)
        self.assertAlmostEqual(white, 255, delta=0.5)
        self.assertAlmostEqual(grey, 59.5, delta=0.5)

    def test_gamma_lookup_fails_for_wrong_cam_name(self):
        sequence = Sequence(self._valid)
        with self.assertRaises(ValueError):
            sequence.gamma('unknown cam', 100)

    def test_gamma_lookup_fails_for_outofbounds_inputs(self):
        sequence = Sequence(self._valid)
        with self.assertRaises(ValueError):
            sequence.gamma('cam1', -5)
        with self.assertRaises(ValueError):
            sequence.gamma('cam1', 260)

    def test_vignette_image_looks_correct(self):
        sequence = Sequence(self._valid)
        vig = sequence.vignette('cam1')
        self.assertTrue(vig is not None)
        self.assertIsInstance(vig, np.ndarray)

    def test_vignette_raises_on_wrong_camname(self):
        sequence = Sequence(self._valid)
        with self.assertRaises(ValueError):
            sequence.vignette('cam221')

    def test_rolling_shutter_speed_maches(self):
        sequence = Sequence(self._valid)
        self.assertAlmostEqual(sequence.rolling_shutter_speed, 0.01586, delta=0.001)

    def test_camera_iteration_with_filter(self):
        sequence = Sequence(self._valid)
        cam = sequence.cameras('rolling')
        x = filter(lambda item: item.ID == 2, cam)
        self.assertEqual(list(x)[0].ID, 2)

    def test_camera_iteration_with_list_comprehension(self):
        sequence = Sequence(self._valid)
        wanted_id = 2
        x = [ image.stamp for image in sequence.cameras('rolling') if image.ID == wanted_id]
        self.assertEqual(x[0], sequence.cameras('rolling')[wanted_id].stamp)

    def test_times_contain_framestamps(self):
        sequence = Sequence(self._valid)
        g = set(img.stamp for img in sequence.cameras('global'))
        r = set(img.stamp for img in sequence.cameras('rolling'))
        frames = g | r

        self.assertTrue(frames < set(sequence.times))    # check if timestamp from frames is subset of all stamps

    def test_times_contain_imustamps(self):
        sequence = Sequence(self._valid)
        i = set(imu.stamp for imu in sequence.imu)
        self.assertTrue(i < set(sequence.times))

    def test_times_contain_groundtruthstamps(self):
        sequence = Sequence(self._valid)
        g = set(gt.stamp for gt in sequence.mocap)
        self.assertTrue(g < set(sequence.times))

    def test_large_time_is_not_in_times(self):
        sequence = Sequence(self._valid)
        x = { 7751621, 2951, -0.1 }
        self.assertFalse(x < set(sequence.times))

    def test_iteration_over_times_yields_times(self):
        sequence = Sequence(self._valid)
        times = sequence.times
        for i, time in enumerate(sequence.times):
            if time != times[i]: self.fail("Time %s (number %d) did not match iterated item %s" % (time, i, times[i]))

    def test_times_are_sorted(self):
        sequence = Sequence(self._valid)
        self.assertListEqual(sequence.times, sorted(sequence.times))

    def test_lookup_returns_image(self):
        sequence = Sequence(self._valid)
        A = sequence.cameras('global')[2]
        data = sequence[A.stamp]
        self.assertIsNotNone(data.global_)
        self.assertIsNotNone(data.rolling)
        self.assertIsNotNone(data.imu)
        self.assertIsNone(data.groundtruth)
        self.assertEqual(A, data.global_)
        self.assertNotEqual(A, data.rolling)

    def test_lookup_returns_imu(self):
        sequence = Sequence(self._valid)
        A = next(sequence.imu)   # get the first value
        data = sequence[A.stamp]
        self.assertIsNotNone(data.imu)
        self.assertIsNone(data.global_)
        self.assertIsNone(data.rolling)
        self.assertIsNone(data.groundtruth)
        self.assertEqual(A, data.imu)

    def test_lookup_returns_gt(self):
        sequence = Sequence(self._valid)
        A = next(sequence.mocap)
        data = sequence[A.stamp]
        self.assertIsNotNone(data.groundtruth)
        self.assertIsNone(data.global_)
        self.assertIsNone(data.rolling)
        self.assertIsNone(data.imu)
        self.assertEqual(A, data.groundtruth)

    def test_lookup_returns_only_nones_for_invalid_timestamp(self):
        sequence = Sequence(self._valid)
        data = sequence[11547]
        self.assertIsNone(data.global_)
        self.assertIsNone(data.rolling)
        self.assertIsNone(data.imu)
        self.assertIsNone(data.groundtruth)

    def test_slice_lookup_returns_more_then_one_element(self):
        sequence = Sequence(self._valid)
        datas = sequence[.1:.2]
        self.assertIsInstance(datas, Iterable)
        self.assertGreaterEqual(len([datas]), 1)

    def test_slicing_with_step_raises_valueerror(self):
        sequence = Sequence(self._valid)
        with self.assertRaises(ValueError) as context:
            x = sequence[1:2:.5]

    def test_inverse_slicing_returns_empty(self):
        sequence = Sequence(self._valid)
        datas = sequence[2:1]
        self.assertEqual(sum(1 for i in datas), 0)

    def test_none_end_slicing_will_yield_all_values_till_end(self):
        sequence = Sequence(self._valid)
        datas = sequence[.1:]
        E  = sequence[sequence.duration]   # get the last element of the actual sequence
        A  = list(datas)[-1]             # get the last element of the generator

        self.assertEqual(E.stamp, A.stamp)

    def test_larger_end_will_slice_only_till_end(self):
        sequence = Sequence(self._valid)
        datas = sequence[:1000]

        E = sequence[sequence.duration]  # get the last element of the actual sequence
        A = list(datas)[-1]  # get the last element of the generator

        self.assertEqual(E.stamp, A.stamp)

    def test_shutter_types_matches_params_file(self):
        sequence = Sequence(self._valid)
        with open(p.join(self._valid, 'params', 'params.yaml')) as stream:
            refs = yaml.load(stream)

        shutters = sequence.shutter_types
        for cam in ["cam1", "cam2", "cam3", "cam4"]:
            if cam not in shutters:
                self.fail('Shutter types does not contain key "%s"' % cam)
            if shutters[cam] not in ['global', 'rolling']:
                self.fail('Shutter type for key "%s" unknown: %s, only "global" and "rolling" supported'
                          % (cam, shutters[cam]))

            self.assertEqual(refs[cam]['shutter']['type'], shutters[cam])

    def test_cam_name_lookup_works(self):
        sequence = Sequence(self._valid)
        self.assertEqual('cam1', sequence.lookup_cam_name('rolling', 'L'))
        self.assertEqual('cam2', sequence.lookup_cam_name('rolling', 'R'))
        self.assertEqual('cam3', sequence.lookup_cam_name('global', 'R'))
        self.assertEqual('cam4', sequence.lookup_cam_name('global', 'L'))

        with self.assertRaises(ValueError) as ctx:
            sequence.lookup_cam_name('invalid_shutter_name', 'L')

        with self.assertRaises(ValueError) as ctx:
            sequence.lookup_cam_name('global', 'invalid_side_name')

if __name__ == '__main__':
    unittest.main()
