import unittest
import numpy as np
import os.path as p

from StereoTUM.dataset import Dataset


class TestImageArchives (unittest.TestCase):

    def setUp(self):
        d = p.dirname(p.realpath(__file__))
        self.path = p.join(d, 'zipped')
        self.dataset = Dataset(self.path)

    def test_unzipped_sequence_is_seen_as_unzipped(self):
        d = p.dirname(p.realpath(__file__))
        dataset = Dataset(p.join(d, 'valid'))
        self.assertFalse(dataset.zipped)

    def test_zipped_sequence_is_seen_as_zipped(self):
        self.assertTrue(self.dataset.zipped)

    def test_cameras_contain_4_images(self):
        self.assertEqual(len(self.dataset.cameras('global')), 4)
        self.assertEqual(len(self.dataset.cameras('rolling')), 4)

    def test_image_paths_are_to_the_zip_archive(self):
        names = ['00001.jpeg', '00002.jpeg', '00003.jpeg']
        for name, img in zip(names, self.dataset.cameras('rolling')):
            targetL = p.join(self.path, 'frames', 'cam1.zip', name)
            targetR = p.join(self.path, 'frames', 'cam2.zip', name)
            self.assertEqual(targetL, img.L.path)
            self.assertEqual(targetR, img.R.path)

        for name, img in zip(names, self.dataset.cameras('global')):
            targetL = p.join(self.path, 'frames', 'cam4.zip', name)
            targetR = p.join(self.path, 'frames', 'cam3.zip', name)
            self.assertEqual(targetL, img.L.path)
            self.assertEqual(targetR, img.R.path)

    def test_image_loading_from_archive_returns_opencv_image(self):
        names = ['00001.jpeg', '00002.jpeg', '00003.jpeg']
        for name, img in zip(names, self.dataset.cameras('rolling')):
            image = img.L.load()
            self.assertIsInstance(image, np.ndarray)
            self.assertTupleEqual(image.shape, (1024, 1280))

        for name, img in zip(names, self.dataset.cameras('global')):
            image = img.R.load()
            self.assertIsInstance(image, np.ndarray)
            self.assertTupleEqual(image.shape, (1024, 1280))




if __name__ == '__main__':
    unittest.main()
