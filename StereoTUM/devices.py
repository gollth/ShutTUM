import StereoTUM.values


class StereoCamera(object):
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
        self._sync = True
        self._data = self._dataset.raw.frames

    def __iter__(self):
        self._id = int(self._data[0,1])   # take the first ID (column 1) in the first row (0th)
        self._max = int(self._data[-1,1]) # take the last ID (column 1) in the last row
        return self

    @property
    def sync(self):
        return self._sync

    @sync.setter
    def sync(self, value):
        self._sync = value

    def next(self): # python2 support
        return self.__next__()

    def __next__(self):
        if self._id > self._max:
            self._id = int(self._data[0, 1])  # reset the id for next iteration
            raise StopIteration

        img = None
        while img is None and self._id <= self._max:
            try:
                img = self[self._id]
            except ValueError as e:
                pass  # just try the next one
            finally:
                self._id += 1

        if self._id > self._max and img is None: raise StopIteration
        return img

    def __len__(self):
        return sum([1 for frame in self])

    def __getitem__(self, id):
        row = self._data[self._data[:,1] == id, :]
        if row.size == 0: raise ValueError('[%s] No Frame for stereo camera "%s" found with ID %d' % (self._dataset, self._shutter, id))
        return StereoTUM.values.StereoImage(self._dataset, row[0], self._shutter, self._sync)


class DuoStereoCamera:
    r"""
    A duo stereo camera consists of a two :any:`StereoCamera` s each with a left and a right one. Both stereo cameras
    use usually (but not necessarily) different shutter methods. If you iterate over a duo stereo camera you go get a 
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
