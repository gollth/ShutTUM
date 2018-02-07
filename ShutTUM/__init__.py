"""
.. module:: ShutTUM
   :platform: Unix, Windows
   :synopsis: A utility API to easily interact with the ShutTUM dataset

.. moduleauthor:: Thore Goll <thore.goll@tum.de>

"""
import math
import numpy   as np


class Interpolation:
    """Interpolation lets you calculate intermediate values between two boarders, based on a given formula.

    .. image:: images/interpolation.svg       

    **ShutTUM**'s data is recorded at different frequencies. Images are timestamped at around 20 Hz, Imu measurements  
    at 160 Hz (exactly multiples per image), while Ground Truth data is clocked at 120 Hz. To 
    get relateable measurements, you must sometimes interpolate between two values.

    ShutTUM supports currently three interpolation methods, namely :any:`linear`, :any:`cubic` 
    and :any:`slerp`. However it is possible to define your own function and hand it over to interpolating 
    methods, such as  :any:`GroundTruth.interpolate` or :any:`Imu.interpolate`::

        def some_crazy_interpolation(a,b,t, a0, b0):
            return a + b - 17*t**2

        # Then use it like so
        t = 1.5  # [s] Time stamp at which to interpolate
        x = sequence.imu[0].interpolate(sequence, t, accelaration_interpolation=some_crazy_interpolation)

        # Now x is interpolated between the imu values closest to 1.5s with the crazy function
        
    """

    @staticmethod
    def linear(a, b, t, *_):
        r"""
        :param a: (float/`ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_) the lower bound from which to interpolate
        :param b: (float/`ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_) the upper bound to which to interpolate 
        :param t: (float) the value between 0 .. 1 of the interpolation
        :param _: further args are ignored but required in the function definition (see :any:`cubic`)
        :rtype: float/`ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_

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
        :param a: (float/`ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_) the lower bound from which to interpolate
        :param b: (float/`ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_) the upper bound to which to interpolate
        :param t: (float) t: the value between 0 .. 1 of the interpolation
        :param a0: (float/`ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_) the value before a, from which the tangent in a is calculated. If None, a0 = a
        :param b0: float/`ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_) the value after b, from which the tangent in b is calculated. If None, b0 = b
        :rtype: float/`ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_

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
        :param quat0: (`ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_) the lower rotational bound from which to interpolate (4x1 vector). Will be normalized
        :param quat1: (`ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_) the upper rotational bound to which to interpolate (4x1 vector). Will be normalized
        :param t: (`float`) the value between 0 .. 1 of the interpolation
        :param _: further args are ignored but required in the function definition (see :any:`cubic`)
        :rtype: `ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_

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

