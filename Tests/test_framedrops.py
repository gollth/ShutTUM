import unittest
import os.path as p

from StereoTUM.dataset import  Dataset

class TestFrameDrops (unittest.TestCase):

    def setUp(self):
        d = p.dirname(p.realpath(__file__))
        self.path = p.join(d, 'framedrop')
        self.dataset = Dataset(self.path)

    def test_index_matches_ids(self):
        global_ids = [1, 4]
        cam = self.dataset.cameras('rolling', sync=False)
        for id in global_ids:
            self.assertEqual(id, cam[id].ID)

        rolling_ids = [2, 3, 4]
        cam = self.dataset.cameras('global', sync=False)
        for id in rolling_ids:
            self.assertEqual(id, cam[id].ID)

    def test_sync_cameras_skip_single_frame_drops(self):
        global_ids = [2]
        cam = self.dataset.cameras('global')
        self.assertEqual(len(global_ids), len(cam))
        for id, stereo in zip(global_ids, cam):
            self.assertEqual(id, stereo.ID)

    def test_unsynced_cameras_will_not_drop_if_only_one_is_missing(self):
        expectation = [(2,2), (None,3), (4,None)]
        cam = self.dataset.cameras('global', sync=False)
        self.assertEqual(len(expectation), len(cam))
        for expect, stereo in zip(expectation, cam):
            if expect[0] is None and stereo.L is not None:
                self.fail('expected left image of %s to be none' % stereo)
            if expect[0] is not None and stereo.L is None:
                self.fail('expected left image of %s not to be none' % stereo)

            if expect[1] is None and stereo.R is not None:
                self.fail('expected right image of %s to be none' % stereo)
            if expect[1] is not None and stereo.R is None:
                self.fail('expected right image of %s not to be none' % stereo)

            if stereo.L is not None:
                self.assertEqual(expect[0], stereo.L.ID)
            if stereo.R is not None:
                self.assertEqual(expect[1], stereo.R.ID)


if __name__ == '__main__':
    unittest.main()
