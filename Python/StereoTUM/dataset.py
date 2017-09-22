import yaml
import cv2
import os.path as p
import numpy   as np
import math
import StereoTUM as api
from collections import namedtuple


class Interpolation:
    """Interpolation lets you calculate intermediate values between two boarders, based on a given formula.

    .. image:: images/interpolation.svg       
    
    **StereoTUM**'s data is recorded at different frequencies. Images are timestamped at around 20 Hz, Imu measurements at 
    around 60 Hz (exactly three measurements per image), while Ground Truth data is clocked between 100 .. 120 Hz. To 
    get relateable measurements, you must sometimes interpolate between two values.
    
    StereoTUM supports currently three interpolation methods, namely :py:func:`~linear`, :py:func:`~cubic` and :py:func:`~slerp`.
    However it is possible to define your own function and hand it over to interpolating methods, such as  :py:func:`GroundTruth.interpolate` 
    or :py:func:`ImuValue.interpolate`::
        
        def some_crazy_interpolation(a,b,t, a0, b0):
            return a + b - 17*t**2
     
        # Then use it like so
        t = 1.5  # [s] Time stamp at which to interpolate
        x = dataset.imu[0].interpolate(dataset, t, accelaration_interpolation=some_crazy_interpolation)
     
        # Now x is interpolated between the imu values closest to 1.5s with the crazy function
    """

    @staticmethod
    def linear(a, b, t, *_):
        r"""
        :param `float/numpy.ndarray` a: the lower bound from which to interpolate
        :param `float/numpy.ndarray` b: the upper bound to which to interpolate 
        :param `float` t: the value between 0 .. 1 of the interpolation
        :param _: further args are ignored but required in the function definition (see :py:func:`~cubic`)
        :rtype: `float/numpy.ndarray`
        
        Interpolate linearly between two points
        
        .. image:: images/interpolation-linear.svg
        
        .. math:: x(t) = \left (1 - t \right ) \cdot a + t \cdot b
        
        Example::
            
            # Find the average between two 3D vectors
            v1 = np.array([0,0,1])
            v2 = np.array([5,0,0])
            v  = Interpolation.linear(v1, v2, .5)   # np.array([2.5,0,0.5])
        
        """
        return (1 - t) * a + t * b

    @staticmethod
    def cubic(a, b, t, a0=None, b0=None):
        r"""
        :param `float`/`numpy.ndarray` a: the lower bound from which to interpolate
        :param `float`/`numpy.ndarray` b: the upper bound to which to interpolate
        :param `float` t: the value between 0 .. 1 of the interpolation
        :param `float`/`numpy.ndarray` a0: the value before a, from which the tangent in a is calculated. If None, a0 = a
        :param `float`/`numpy.ndarray` b0: the value after b, from which the tangent in b is calculated. If None, b0 = b
        :rtype: `float`/`numpy.ndarray`
        
        Interpolate cubically between two points. Note that a, b, a0 and a0 must all have the same shape.
        
        .. image:: images/interpolation-cubic.svg
        
        .. math:: 
            x(t) = a + \frac{1}{2} t \cdot \left ( b - a_0 + t \cdot \left (2 a_0 - 5 a + 4 b - b_0 + t \cdot \left (3 \cdot \left (a - b \right ) + b_0 - a_0 \right ) \right ) \right ) 
        """
        if a0 is None: a0 = a
        if b0 is None: b0 = b
        return a + 0.5 * t * (b - a0 + t * (2.0 * a0 - 5.0 * a + 4.0 * b - b0 + t * (3.0 * (a - b) + b0 - a0)))

    @staticmethod
    def slerp(quat0, quat1, t, *_):
        r"""
        :param `numpy.ndarray` quat0: the lower rotational bound from which to interpolate (4x1 vector). Will be normalized
        :param `numpy.ndarray` quat1: the upper rotational bound to which to interpolate (4x1 vector). Will be normalized
        :param `float` t: the value between 0 .. 1 of the interpolation
        :param _: further args are ignored but required in the function definition (see :py:func:`~cubic`)
        :rtype: `numpy.ndarray`
        
        Interpolate spherically between two quaternions. This method is mostly used to interpolate rotations.
        Directly copied from `Transformation.py <http://www.lfd.uci.edu/~gohlke/code/transformations.py.html>`_
        """
        q0 = Interpolation._unit_vector(quat0[:4])
        q1 = Interpolation._unit_vector(quat1[:4])
        _EPS = np.finfo(float).eps * 4.0
        if t == 0.0:
            return q0
        elif t == 1.0:
            return q1
        d = np.dot(q0, q1)
        if abs(abs(d) - 1.0) < _EPS:
            return q0
        if d < 0.0:
            # invert rotation
            d = -d
            np.negative(q1, q1)
        angle = math.acos(d)
        if abs(angle) < _EPS:
            return q0
        isin = 1.0 / math.sin(angle)
        q0 *= math.sin((1.0 - t) * angle) * isin
        q1 *= math.sin(t * angle) * isin
        q0 += q1
        return q0

    @staticmethod
    def _unit_vector(data, axis=None, out=None):
        r"""
        Directly copied from `Transformation.py <http://www.lfd.uci.edu/~gohlke/code/transformations.py.html>`_
        """
        if out is None:
            data = np.array(data, dtype=np.float64, copy=True)
            if data.ndim == 1:
                data /= math.sqrt(np.dot(data, data))
                return data
        else:
            if out is not data:
                out[:] = np.array(data, copy=False)
            data = out
        length = np.atleast_1d(np.sum(data * data, axis))
        np.sqrt(length, length)
        if axis is not None:
            length = np.expand_dims(length, axis)
        data /= length
        if out is None:
            return data


class Dataset(object):
    r"""The base class representing one dataset record.
    
    In order for performent operations the dataset loads and checks lots of the data in its :func:`__init__` thus avoiding
    expensive checks in for loops or list comprehensions. In addition it holds the reference to list-like objects, such 
    as :func:`cameras` or :attr:`.imu`. You can iterate over each of these (depending
    on application) to get the corresponding :class:`Value`s in order.
    
    A typical application might look like::
    
        # Load a dataset
        dataset = Dataset('path/to/folder/0001')
        
        # Iterate over all imu values and find corresponding images
        for observation in dataset.imu:
            
            print(observation.acceleration)
            print(observation.angular_velocity)
            
            stereo = observation.stereo(shutter='global')
            if stereo is None: continue
            
            print(stereo.L.ID)
    
    """

    @staticmethod
    def _check_folder_exists(folder):
        r""" Checks if a folder exists and raises an exception otherwise
        :param str folder: the path to the folder to check for existance
        :raises FileNotFoundError
        """
        if not p.exists(p.join(folder)):
            raise FileNotFoundError("Could not find folder %s, record folder seems not to be valid!" % folder)

    @staticmethod
    def _check_file_exists(file):
        r""" Checks if a file exists and raises an exception otherwise
        :param str file: the path to the file to check for existance
        :raises FileNotFoundError
        """
        if not p.exists(file):
            raise FileNotFoundError("Could not find %s, record folder seems not to be valid!" % file)

    @staticmethod
    def _check_contains_key(file, dictionary, key):
        r"""
        :param str file: The reference file, which to give in the error message
        :param dict dictionary: the dict to check the key's existance
        :param str key: the key to check for 
        :raises ValueError
        
        Checks if a dict originating from a certain file contains a certain key and raises an exception otherwise
        """
        if key not in dictionary:
            raise ValueError("Could not find %s in %s, record folder seems not to be valid!" % (key, file))

    def __init__(self, path):
        r""" 
        :param str path: the path to *one* record of the dataset, such as ``~/StereoTUM/0001``
        :raises: ValueError: if anything goes wrong 
        Load the dataset into memory (except images) and do basic data consistency checks.
        
        1. It is checked that in path there exists a ``data``, ``frames`` and ``params`` folder
        2. It is checked that there exists the files ``data/frames.csv``, ``data/imu.csv`` and ``data/ground_truth.csv``
        3. The files from 2. are loaded into memory (see :py:attr:`raw`)
        4. The ``params/time.yaml`` is loaded
        5. The ``params/params.yaml`` is loaded
           
        """
        self._path = path

        # Consistency Check
        Dataset._check_folder_exists(p.join(path, 'data'))
        Dataset._check_folder_exists(p.join(path, 'frames'))
        Dataset._check_folder_exists(p.join(path, 'params'))

        f = p.join(path, 'data', 'frames.csv')
        i = p.join(path, 'data', 'imu.csv')
        g = p.join(path, 'data', 'ground_truth.csv')

        Dataset._check_file_exists(f)
        Dataset._check_file_exists(i)
        Dataset._check_file_exists(g)

        self._frames       = np.genfromtxt(f, delimiter='\t', skip_header=1)
        self._imu          = np.genfromtxt(i, delimiter='\t', skip_header=1)
        self._ground_truth = np.genfromtxt(g, delimiter='\t', skip_header=1)

        Raw = namedtuple('Raw', ['frames', 'imu', 'groundtruth'])
        self.raw = Raw(self._frames, self._imu, self._ground_truth)
        self._time = {}
        timefile = p.join(path, 'params', 'time.yaml')
        Dataset._check_file_exists(timefile)
        with open(timefile) as stream:
            self._time = yaml.load(stream)['time']
            Dataset._check_contains_key(timefile, self._time, 'start')
            Dataset._check_contains_key(timefile, self._time, 'end')
            Dataset._check_contains_key(timefile, self._time, 'duration')

        self._cams = {}
        paramfile = p.join(path, 'params', 'params.yaml')
        Dataset._check_file_exists(paramfile)
        with open(paramfile) as stream:
            self._refs = yaml.load(stream)

        self._gammas = {}
        self._cams = {}
        for ref in self._refs:
            if ref == 'world': continue  # world param must only be present, not more

            # Every other reference must have at least a transform parameter
            Dataset._check_contains_key(paramfile, self._refs[ref], 'transform')

            if 'shutter' not in self._refs[ref]: continue

            # If the reference contains a 'shutter' param, it is a camera
            cam = ref
            self._cams[cam] = self._refs[ref]

            folder   = p.join(path,   'params', cam)
            gamma    = p.join(folder, 'gamma.txt')
            vignette = p.join(folder, 'vignette.png')
            Dataset._check_folder_exists(folder)
            Dataset._check_file_exists(gamma)
            Dataset._check_file_exists(vignette)
            self._gammas[cam] = np.genfromtxt(gamma, delimiter=' ')

        self._cameras = api.DuoStereoCamera(self)
        self._imu     = api.Imu(self)
        self._mocap   = api.Mocap(self)

    def cameras(self, shutter='both'):
        r""" 
        :param str shutter: {both/global/rolling} the type of shutter you are interested in. 
        :return The reference of the cameras, which you can iterate either as :class:`~devices.StereoCamera` (global/rolling)or :class:`~devices.DuoStereoCamera` (both)
        :raise: ValueError: for anything other then both/global/rolling
        
        Get a reference to one or both of the two stereo cameras, to iterate over their images
        
        Note that often you want to directly iterate over those references, which you can do like::
        
            # Iterate over both rolling and global images
            for g, r in dataset.cameras(shutter='both'):
                print(g.L.ID)
                print(r.R.stamp)
        
        """
        if shutter == 'both': return self._cameras
        if shutter == 'global': return self._cameras._global
        if shutter == 'rolling': return self._cameras._rolling

        raise ValueError('Unknown shutter type: use either "global", "rolling", or "both" and not %s' % shutter)

    @property
    def imu(self):
        r""" 
        :return: The reference of the Imu for this dataset, which you can iterate as :class:`~devices.Imu`
        
        Get a reference to the IMU to iterate over its values.
        
        Note that often you want to directly iterate over the Imu like so::
            
            # Iterate over all imu values
            for observation in dataset.imu:
                print(imu.acceleration)
            
        
        """
        return self._imu

    @property
    def mocap(self):
        r"""
        :return: The reference to the Ground Truth list as :class:`~devices.Mocap`
       
        Get a reference to the Motion Capture system to iterate over the ground truth values.
         
        Note that often you want to iterate directly over the Ground Truth values like so::
            
            # Iterate over all ground truth values
            for gt in dataset.mocap:
                print(gt.stamp)
                print(gt.position)
            
        
        """
        return self._mocap

    @property
    def start(self):
        r"""
        The start time of the record as Unix Timestamp in seconds with decimal places
        """
        return self._time['start']

    @property
    def end(self):
        r"""
        The end time of the record as Unix Timestamp in seconds with decimal places
        """
        return self._time['end']

    @property
    def duration(self):
        r""" 
        The duration of the record in seconds as float, so basically:
        ``dataset.end - dataset.start``
        
        """
        return self._time['duration']

    @property
    def exposure_limits(self):
        r"""
        :return: a Namped Tuple with the fields ``min`` and ``max`` which each contain a float indicating the minimum
        and maximum exposure time in milliseconds
        
        The minimal & maximal exposure used for all cameras. Note that these values are the *limits*
        not the extrema of the record, so most of the time, these will not be reached, but if, clamped accordingly.::
        
            limits = dataset.exposure_limits
            print("Limits are %s .. %s ms" % limits.min, limits.max)
            
        
        """
        Limits = namedtuple('Limits', ['min', 'max'])
        for cam in self._cams:
            # take the first camera, since all limits are the same
            exp = self._cams[cam]['exposure']
            return Limits(min=exp['min'], max=exp['max'])

    def gamma(self, cam, input):
        r""" 
        :param str cam: the name of the camera (e.g. "cam1")
        :param float input: the position to lookup, i.e. X-axis on luminance plot. Between 0 .. 255, will be rounded to int
        :raises: ValueError: for unknown camera names or inputs below 0 or above 255
        
        Lookup a gamma value from ``params/<cam>/gamma.txt``
        """
        if cam not in self._cams:
            raise ValueError("Unknown camera name: %s" % cam)
        if input < 0 or input > 255:
            raise ValueError("Gamma function only defined for inputs from 0 .. 255 and not for %s" % input)
        return self._gammas[cam][round(input)]

    def vignette(self, cam):
        r"""
        :param str cam: the name of the camera to lookup its vignette (e.g. "cam1")  
        :return: the vignette image, read by ``cv2.imread()`` with dimensions [1280x1024] as grayscale
        :rtype: ndarray
        """
        if cam not in self._cams:
            raise ValueError("Unknown camera name: %s" % cam)
        file = p.join(self._path, 'params', cam, 'vignette.png')
        return cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    @property
    def rolling_shutter_speed(self):
        r"""
        How fast did the two rolling shutter cameras shuttered. Returns the time between the exposure of two consecutive 
        rows in milli seconds (approximate)
        
        """
        for cam in self._cams:
            shutter = self._cams[cam]['shutter']
            if shutter['type'] == 'rolling':
                return shutter['speed']
        raise ValueError('No cams in %s had rolling shutter enabled!' % self._path)
