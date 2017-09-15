import StereoTUM as api

class StereoCamera:

    def __init__(self, dataset, shutter):
        self._dataset = dataset
        self._shutter = shutter
        self._data = self._dataset.raw.frames

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if len(self) <= self._index: raise StopIteration

        img = self[self._index]
        self._index += 1
        return img

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, item):
        return api.StereoImage(self._dataset, self._data[item], self._shutter)


class DuoStereoCamera:

    def __init__(self, dataset):
        self._rolling = StereoCamera(dataset, 'rolling')
        self._global  = StereoCamera(dataset, 'global')

    def __iter__(self):
        self._rolling.__iter__()
        self._global.__iter__()
        return self

    def __next__(self):
        g = self._global.__next__()
        r = self._rolling.__next__()
        return g, r

    def __len__(self):
        return min(len(self._global), len(self._rolling))

    def __getitem__(self, item):
        return self._global[item], self._rolling[item]


class Imu:

    def __init__(self, dataset):
        self._dataset = dataset
        self._data = self._dataset.raw.imu

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if len(self) >= self._index: raise StopIteration

        imu = self[self._index]
        self._index += 1
        return imu

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, item):
        return api.ImuValue(self._dataset, self._data[item])


class Mocap:

    def __init__(self, dataset):
        self._dataset = dataset
        self._data = self._dataset.raw.groundtruth

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if len(self) >= self._index: raise StopIteration

        gt = self[self._index]
        self._index += 1
        return gt

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, item):
        return api.GroundTruth(self._dataset, self._data[item])