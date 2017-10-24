import os.path as p
import StereoTUM.values


class StereoCamera:
    r"""
    A stereo camera consists of a left and a right camera facing both in the same direction. Both cameras have the same
    shutter method, so either rolling or global. If you iterate over a stereo camera you go through all the 
    :any:`StereoImage` in order::
        
        # You can use a "classic" for loop ... 
        for stereo in dataset.cameras('global')
            print(stereo.stamp)
        
        # Or filter/map functions ...
        stereo = filter(lambda item: item.ID == 2, dataset.cameras('rolling'))
        print(list(stereo))
        
        # Or even list & dict comprehensions
        stereo = [ image.stamp for image in dataset.cameras('rolling') if image.ID == 2]
        print(stereo)
    
    """
    def __init__(self, dataset, shutter):
        r"""
        Creates a new StereoCamera as iterable container for images. Usually you will not invoke this constructor
        directly but rather get a reference via :any:`cameras <StereoTUM.Dataset.cameras>`
        
        :param Dataset dataset: The reference to the dataset 
        :param str shutter: the name of the shutter this camera uses (usually "rolling" or "global")
        """
        self._dataset = dataset
        self._shutter = shutter
        self._data = self._dataset.raw.frames

    def __iter__(self):
        self._index = 0
        return self

    def next(self): # python2 support
        return self.__next__()

    def __next__(self):
        if len(self) <= self._index: raise StopIteration

        img = self[self._index]
        self._index += 1
        return img

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, item):
        try:
            return StereoTUM.values.StereoImage(self._dataset, self._data[item], self._shutter) 
        except ValueError:
            return None


class DuoStereoCamera:
    r"""
    A duo stereo camera consists of a two :any:`StereoCamera` s each with a left and a right one. Both stereo cameras
    use usally (but not necessarily) different shutter methods. If you iterate over a duo stereo camera you go get a 
    tuple with both :any:`StereoImage` s in order::

        # You can use a "classic" for loop ... 
        for stereo_1, stereo_2 in dataset.cameras()     # both is the default
            print(stereo_1.shutter)
            print(stereo_2.shutter)

    """
    def __init__(self, dataset):
        self._rolling = StereoCamera(dataset, 'rolling')
        self._global  = StereoCamera(dataset, 'global')

    def __iter__(self):
        self._rolling.__iter__()
        self._global.__iter__()
        return self

    def next(self): # python2 support
        return self.__next__()

    def __next__(self):
        g = self._global.__next__()
        r = self._rolling.__next__()
        return g, r

    def __len__(self):
        return min(len(self._global), len(self._rolling))

    def __getitem__(self, item):
        return self._global[item], self._rolling[item]


class Imu:
    r"""
    An IMU is an iterable container of all :any:`ImuValue` s. You can iterate over all of its observations like so::
    
        # Iteration via "classic" for loop
        for observation in dataset.imu:
            print(observation.acceleration)
    
        # Or use simple index based access
        print(dataset.imu[0].stamp)
        
    """
    def __init__(self, dataset):
        self._dataset = dataset
        self._data = self._dataset.raw.imu

    def __iter__(self):
        self._index = 0
        return self

    def next(self): # python2 support
        return self.__next__()

    def __next__(self):
        if len(self) <= self._index: raise StopIteration

        imu = self[self._index]
        self._index += 1
        return imu

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, item):
        v = StereoTUM.values.ImuValue(self._dataset, self._data[item])
        return v


class Mocap:
    r"""
    The Motion Capture system is an iterable container for all (recorded) :any:`GroundTruth` values in a dataset::
        
        # Iteration via "classic" for loop
        for gt in dataset.mocap:
            print(gt.pose)
    
        # Or use simple index based access
        T_world_2_cam1 = dataset.mocap[17] >> "cam1"
        print(T_world_2_cam1.pose)
        
    
    """
    def __init__(self, dataset):
        self._dataset = dataset
        self._data = self._dataset.raw.groundtruth

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if len(self) <= self._index: raise StopIteration

        gt = self[self._index]
        self._index += 1
        return gt

    def next(self): # python2 support
        return self.__next__()

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, item):
        return StereoTUM.values.GroundTruth(self._dataset, self._data[item])
