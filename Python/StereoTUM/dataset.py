


import yaml
import cv2
import os.path as p
import numpy   as np
import math
import StereoTUM as api
from collections import namedtuple


class Interpolation:
    """
    Interpolation lets you calculate intermediate values between two boarders, based on a given formula.
    StereoTUM supports currently three interpolation methods, namely :func:`~Interpolation.linear`
    """
    @staticmethod
    def linear(a, b, t, *_):
        return (1 - t) * a + t * b

    @staticmethod
    def cubic(a, b, t, a0=None, b0=None):
        if a0 is None: a0 = a
        if b0 is None: b0 = b
        return a + 0.5 * t * (b - a0 + t * (2.0 * a0 - 5.0 * a + 4.0 * b - b0 + t * (3.0 * (a - b) + b0 - a0)))


    @staticmethod
    def slerp(quat0, quat1, t, *_):
        """Directly copied from http://www.lfd.uci.edu/~gohlke/code/transformations.py.html"""
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
        """Directly copied from http://www.lfd.uci.edu/~gohlke/code/transformations.py.html"""

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

    @staticmethod
    def _check_folder_exists(folder):
        if not p.exists(p.join(folder)):
            raise FileNotFoundError("Could not find folder %s, record folder seems not to be valid!" % folder)

    @staticmethod
    def _check_file_exists(file):
        if not p.exists(file):
            raise FileNotFoundError("Could not find %s, record folder seems not to be valid!" % file)

    @staticmethod
    def _check_contains_key(file, dictionary, key):
        if key not in dictionary:
            raise ValueError("Could not find %s in %s, record folder seems not to be valid!" % (key, file))

    def __init__(self, path):
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
        if shutter == 'both': return self._cameras
        if shutter == 'global': return self._cameras._global
        if shutter == 'rolling': return self._cameras._rolling

        raise ValueError('Unknown shutter type: use either "global", "rolling", or "both" and not %s' % shutter)

    @property
    def imu(self):
        return self._imu

    @property
    def mocap(self):
        return self._mocap

    @property
    def start(self):
        return self._time['start']

    @property
    def end(self):
        return self._time['end']

    @property
    def duration(self):
        return self._time['duration']

    @property
    def exposure_limits(self):
        Limits = namedtuple('Limits', ['min', 'max'])
        for cam in self._cams:
            # take the first camera, since all limits are the same
            exp = self._cams[cam]['exposure']
            return Limits(min=exp['min'], max=exp['max'])

    def gamma(self, cam, input):
        if cam not in self._cams:
            raise ValueError("Unknown camera name: %s" % cam)
        if input < 0 or input > 255:
            raise ValueError("Gamma function only defined for inputs from 0 .. 255 and not for %s" % input)
        return self._gammas[cam][input]

    def vignette(self, cam):
        if cam not in self._cams:
            raise ValueError("Unknown camera name: %s" % cam)
        file = p.join(self._path, 'params', cam, 'vignette.png')
        return cv2.imread(file)

    @property
    def rolling_shutter_speed(self):
        for cam in self._cams:
            shutter = self._cams[cam]['shutter']
            if shutter['type'] == 'rolling':
                return shutter['speed']
        raise ValueError('No cams in %s had rolling shutter enabled!' % self._path)
    