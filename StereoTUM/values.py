import cv2
import numpy as np
import os.path as p
import transforms3d as tf
import StereoTUM
from collections import namedtuple


class Value(object):
    r"""
    A Value represents every measurement of a sensor taken in a specific time instance.
    A camera might capture an :any:`Image`, an IMU will record :any:`Imu` or the motion capture system 
    :any:`GroundTruth`.
     
    .. image:: images/frames.png 
     
    All these values have some common properties in the StereoTUM:
    
    1) They are single, time-discrete values (see :any:`stamp <StereoTUM.Value.stamp>`)
    2) They are related to a certain reference frame (see :any:`reference <StereoTUM.Value.reference>`)
    3) They all have a transformation from its reference frame towards that of ``"cam1"``
    
    
    Since every value has these three properties, you can achieve easily get the transformations between different  values. 
    Therefore the leftshift ``<<`` and rightshift ``>>`` operator has been overloaded. Both accept as their right parameter either:
    
    * a string indicating the desired reference to or from which to transform (e.g. ``"cam1"``, ``"cam2"``, ``"world"``, ``"imu"`` ...)
    * or another value, whose :any:`reference <StereoTUM.Value.reference>` property is used to determine the transform
    
    The direction of the "shift" means "How is the transformation from reference x to y?"::
    
        # assume we have an image and a ground truth value
        image = dataset.cameras('global')[0]
        gt    = dataset.groundtruth[0]
        
        # Since both Image and GroundTruth derive from Value, they have a reference ...
        print("Image %s is associated with reference %s" % (image, image.reference))
        print("Ground Truth %s is associated with reference %s" % (gt, gt.reference))
        
        # ... and also a transform from the systems origin, which is "cam1"
        print("T_cam1_2_image: %s" % image._transform)
        print("T_cam1_2_gt:    %s" % gt._transform)
        # BUT since they are "private" functions from value you should rather use something like this:
         
         
        # Since we know both transforms to "cam1", we can compute the transformation between the two
        T_image_2_gt = image >> gt
        T_image_2_gt = gt << image        # same as line above
        # T_image_2_gt = gt.reference << image # This fails, since the shift operators cannot be overloaded for strings in the first place
        
        T_gt_2_image = gt >> image
        T_gt_2_image = image << gt
        T_gt_2_image = gt >> "cam2"       # for example, if image was taken by "cam2"
        # T_gt_2_image = "cam2" << gt.reference  # obviously this will also not work ...
    
    """
    def __init__(self, dataset, stamp, reference):
        self._dataset = dataset
        self._stamp = stamp
        if reference not in dataset._refs:
            raise ValueError("Cannot find the reference %s" % reference)
        self._reference = reference

    @property
    def _previous(self):
        return None

    def __eq__(self, other):
        if not isinstance(other, Value): return False
        return self._stamp == other._stamp and self.reference == other.reference

    @property
    def stamp(self):
        r"""
        The timestamp in seconds with decimal places. This stamp describes the exact moment this
        values has been recorded.
        """
        return self._stamp

    @property
    def reference(self):
        r"""
        The name as ``str`` of the reference frame, this value is associated with. 
        """
        return self._reference

    def dt(self, ifunknown=.001):
        r"""
        The time delta elapsed since the last :any:`Value` of this type.
        Note that this value varies depending on the concrete type of value. That means dt will be about 1/20 s
        for an :any:`Image` while around 1/100 s for a :any:`GroundTruth`. If a previous value is not present
        this will return time specified by `ifunknown`.
        """
        previous = self._previous       # cache for performance
        if previous is None: return ifunknown
        else: return self.stamp - previous.stamp

    @property
    def _transform(self):
        r"""
        The transformation from ``"cam1"`` to this value`s :any:`reference <StereoTUM.Value.reference>` as 4x4 `ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_ homogenous matrix 
        """
        return np.array(self._dataset._refs[self._reference]['transform'])

    def __str__(self):
        return "%s (%s/%.2f)" % (type(self).__name__, self.reference, self.stamp)

    def __lshift__(self, parent):
        if isinstance(parent, str):
            if parent not in self._dataset._refs:
                raise ValueError("Cannot find the (_static) parent reference %s" % parent)

            tparent = np.array(self._dataset._refs[parent]['transform'])

        elif isinstance(parent, Value):
            tparent = parent._transform
        else:
            raise TypeError("Cannot only lookup transforms for type string or StereoTUM.Value")

        tchild = self._transform
        return np.dot(np.linalg.inv(tparent), tchild)

    def __rshift__(self, child):
        if isinstance(child, str):
            if child not in self._dataset._refs:
                raise ValueError("Cannot find the (_static) parent reference %s" % child)
            tchild = np.array(self._dataset._refs[child]['transform'])
        elif isinstance(child, Value):
            tchild = child._transform
        else:
            raise TypeError("Cannot only lookup transforms for type string or StereoTUM.Value")

        tparent = self._transform
        return np.dot(np.linalg.inv(tparent), tchild)


class Image(Value):
    r"""
    An image is a :any:`Value` with its reference set to one of ``"cam1"`` ... ``"cam4"``.

    Each image is recorded by a :any:`StereoCamera`, which has a shutter type, so as its images. Note, though,
    that each camera might differ its shutter methods between dataset records, to achieve statistical independence. 
    That means, that you cannot rely on e.g. ``"cam1"`` always having ``"global"`` or ``"rolling"`` shutter, nor as the other cams.

    The cameras record data at approximately **20 FPS**, but sometimes their might exist frame drops. 

    You can query a lot of information from an image such as its :any:`shutter <StereoTUM.Image.shutter>`, :any:`exposure <StereoTUM.Image.exposure>` time and :any:`ID <StereoTUM.Image.ID>`.
    Since it is a :any:`Value` all transform shenanigans apply.
    """

    _Vector2 = namedtuple('Vector2', 'x y')
    
    def __init__(self, stereo, shutter, left):
        self._left = left
        self._stereo = stereo
        self._shutter = shutter

        if shutter not in ['global', 'rolling']:
            raise ValueError('Shutter type can only be "global" or "rolling"')

        for cam in stereo._dataset._cams:
            # Check if the shutter matches
            if stereo._dataset._cams[cam]['shutter']['type'] != shutter: continue

            # Check if the camera position (left/right) matches the camera name
            if left and cam not in ['cam1', 'cam3']: continue
            if not left and cam not in ['cam2', 'cam4']: continue

            # Now we have found a camera matching the wanted shutter type and position
            super(Image, self).__init__(stereo._dataset, stereo._data[0], cam)
            break  # from any further for loop iteration

        if not p.exists(self.path): raise ValueError('[%s] Image "%s" does not exist' % (self._dataset, self.path))

    @property
    def _previous(self):
        return self._stereo._previous

    def __eq__(self, other):
        if not isinstance(other, Image): return False
        return (super(Image, self).__eq__(other)
                and self._left == other._left
                and self._shutter == other._shutter)

    def __str__(self):
        return "Image (%s/#%05d/%.2f)" % (self.reference, self.ID, self.stamp)

    @property
    def resolution(self):
        r""" 
        Returns the resolution of the cameras as a named tuple ``(width, height)``
        
        .. seealso:: :any:`Dataset.resolution <StereoTUM.Dataset.resolution>`
        """
        return self._dataset.resolution

    @property
    def shutter(self):
        r""" The shutter method with which this image was captured as string, either ``"rolling"`` or ``"global"`` """
        return self._shutter

    @property
    def ID(self):
        r"""The frame ID as int of this image. This number, prepended to 5 digits is also the name of the JPEG file"""
        return self._stereo.ID

    @property
    def opposite(self):
        r"""This holds the reference of the opposite image, that is the image taken from the other camera with the 
        same shutter."""
        return self._stereo._opposite(self)

    @property
    def exposure(self):
        r"""The exposure time as float in milli seconds, that this image was illuminated"""
        return self._stereo.exposure

    @property
    def illuminance(self):
        r"""
        The image's illuminance in lux.
        
        .. seealso:: :any:`StereoImage.illuminance <StereoTUM.StereoImage.illuminance>`
        """
        return self._stereo.illuminance

    @property
    def path(self):
        r"""The path to the JPEG file of this image, relative to the the construction parameter of :any:`Dataset(...) <StereoTUM.Dataset.__init__>`"""
        return p.join(self._stereo._dataset._path, 'frames', self.reference, '%05d.jpeg' % self.ID)

    @property
    def imu(self):
        r"""The matching :any:`Imu` for this image. Since the capture of an image is synchronized with the IMU,
        no interpolation is needed."""
        return self._stereo.imu

    def groundtruth(self, position_interpolation=StereoTUM.Interpolation.linear,
                    orientation_interpolation=StereoTUM.Interpolation.slerp):
        r"""
        Find the matching :any:`GroundTruth` value for this image. Since the motion capture system and the cameras
        are not synced, we need to interpolate between ground truths by image's time stamp.

        :param position_interpolation: a predefined or custom interpolation function to interpolate positions 
        :param orientation_interpolation: a predefined or custom interpolation function to interpolate quaternions
        :return: the matching interpolated ground truth

        .. seealso:: :any:`StereoTUM.GroundTruth.interpolate`
        """
        # The ground truth value is from world to our reference
        # Therefore we first get the gt from world -> cam1 ...
        gt1 = GroundTruth.interpolate(self._dataset, self.stamp, position_interpolation, orientation_interpolation)

        # ... then from cam1 -> us
        t1 = self << 'cam1'

        # which delivers from world -> us
        A = gt1.pose.dot(t1)
        T, R, _, _ = tf.affines.decompose44(A)
        q = tf.quaternions.mat2quat(R)
        p = np.array(T)
        return GroundTruth(self._dataset, np.concatenate(([gt1.stamp], p, q)))

    def load(self):
        r"""
        Loads the JPEG itself into memory::

            image = ...
            pixels = image.load()   # uses cv2.imread()
            print(pixels.shape)     # (1024, 1280)
            print(type(pixels))     # numpy.ndarray

        """
        return cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)

    @property
    def distortion(self):
        r"""The FOV distortion parameter as ``float`` for the provided camera"""
        return self._dataset._refs[self.reference]['distortion']
    
    @property
    def focal(self):
        r""" The camera's focal length as named tuple ``(x, y)`` in pixels """
        return Image._Vector2(x=self._dataset._refs[self.reference]['intrinsics'][0],
                              y=self._dataset._refs[self.reference]['intrinsics'][1])

    @property
    def principle(self):
        r""" The camera's principle point as named tuple ``(x, y)`` in pixels """
        return Image._Vector2(x=self._dataset._refs[self.reference]['intrinsics'][2],
                              y=self._dataset._refs[self.reference]['intrinsics'][3])

    @property
    def P(self):
        r"""The projection or intrinsic camera matrix as 4x3 `ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_"""
        f = self.focal
        p = self.principle
        return np.array(((f.x,   0, p.x, 0),
                         (  0, f.y, p.y, 0),
                         (  0,   0,   1, 0)))


class StereoImage(Value):
    r"""A stereo image contains of two individual :any:`Image` s, a left and a right one.
    It is more or less a container for the two.

    Note that for internal reasons a stereo image derives from :any:`Value`. However, you should
    not use the transform functions (``<<`` and ``>>``) with this, since a stereo image contains two 
    reference frames, one for each camera::

        stereo = dataset.cameras('rolling')[0]

        # The following is ambiguous
        notok = stereo << "imu"   # which camera of the two do you mean?

        # Better would be
        ok = stereo.L << "imu"


    """

    @staticmethod
    def extrapolate(value, shutter, method='closest'):
        r"""
        Find a matching image for a certain :any:`Value` based on a extrapolation method

        :param value: The value for which to find a matching image
        :param shutter: The shutter type of the images to find (``"global"``, ``"rolling"``, **not** ``"both"``) 
        :param method: An optional extrapolation method to determine the rules for a "match":

            * ``"closest"``: the image with the least difference to value.stamp is chosen
            * ``"next"``: the image with the next larger time stamp than value.stamp is chosen
            * ``"prev"``: the image with the next smaller time stamp than value.stamp is chosen
            * ``"exact"``: the image where value.stamp == image.stamp holds is chosen, None otherwise

        :return: The matching stereo image or None if no was found
        
        .. seealso:: :any:`Dataset.cameras <StereoTUM.Dataset.cameras>`
        
        """
        if method == 'closest':
            f = value._dataset.raw.frames
            i = np.abs(f[:, 0] - value.stamp).argmin()
            return StereoImage(value._dataset, f[i, :], shutter)

        if method == 'next':
            f = value._dataset.raw.frames
            frame = f[f[:, 0] > value.stamp, :]
            if frame.size == 0: return None
            return StereoImage(value._dataset, frame[0], shutter)

        if method == 'prev':
            f = value._dataset.raw.frames
            frame = f[f[:, 0] < value.stamp, :]
            if frame.size == 0: return None
            return StereoImage(value._dataset, frame[-1], shutter)

        if method == 'exact':
            f = value._dataset.raw.frames
            frame = f[f[:, 0] == value.stamp, :]
            if frame.size == 0: return None
            return StereoImage(value._dataset, frame[0], shutter)

        raise ValueError('[%s] Unknown extrapolation method: %s (supported are "closest", "next", "prev" and "exact")'
                         % (value._dataset, method))

    def __init__(self, dataset, data, shutter):
        self._data = data
        self._dataset = dataset
        L, R = None, None
        frames = dataset.raw.frames
        id = self._data[1]
        while True:
            try: L = Image(self, shutter, left=True)
            except ValueError: L = None
            try: R = Image(self, shutter, left=False)
            except ValueError: R = None

            # If stereosync is required, the stereo image is only valid, if we found both L/R images
            if dataset.stereosync and L is not None and R is not None: break

            # If no stereosync is required, it suffices if we find only one
            if not dataset.stereosync and (L is not None or R is not None): break

            # Otherwise we look for the next data in frames.csv and try again ...
            id += 1
            self._data = frames[frames[:,1] == id]
            if not self._data.size == 0:
                self._data = self._data[0]
                continue

            raise ValueError('[%s] Cannot find no more stereo images due to frame drops for camera "%s"'
                             % (self._dataset, shutter))

        # Timestamp is in second column
        cam = L if L is not None else R
        self._left, self._right = L, R
        super(StereoImage, self).__init__(dataset, self._data[0], cam.reference)

    @property
    def _previous(self):
        return StereoImage.extrapolate(self, self.L.shutter, method='prev')

    def __eq__(self, other):
        if not isinstance(other, StereoImage): return False
        return super(StereoImage, self).__eq__(other) and np.allclose(self._data, other._data)

    def __str__(self):
        l = self._left.reference if self._left is not None else 'None'
        r = self._right.reference if self._right is not None else 'None'
        return "StereoImage ({%s|%s}/%05d" % (l, r, self.ID)

    def _opposite(self, img):
        if img is self._left: return self._right
        if img is self._right: return self._left
        raise ValueError("Image %s unknown, cannot find opposite")

    @property
    def resolution(self):
        r""" 
        Returns the resolution of the cameras as a named tuple ``(width, height)``
        
        .. seealso:: :any:`Dataset.resolution <StereoTUM.Dataset.resolution>`
        """
        return self._dataset.resolution

    @property
    def ID(self):
        r"""The frame ID as int of this image. This number, prepended to 5 digits is also the name of the JPEG file"""
        return int(self._data[1])

    @property
    def exposure(self):
        r"""
        The exposure time as float in milli seconds, that this image was illuminated. This is constant for 
        :any:`L <StereoTUM.StereoImage.L>` and :any:`R <StereoTUM.StereoImage.R>`
        """
        return self._data[2]

    @property
    def illuminance(self):
        r"""
        The estimated illumnience measured by the `TSL2561 Lux-sensor <https://cdn-shop.adafruit.com/datasheets/TSL2561.pdf>`_. 
        This float is measured in lx but is only an approximation. This value is constant for both 
        :any:`L <StereoTUM.StereoImage.L>` and :any:`R <StereoTUM.StereoImage.R>`. Based on this value, the estimated 
        :any:`exposure` time can be computed with the following hand fitted formula:

        .. image:: images/autoexposure.svg

        .. math:: t_{exposure} \left ( E_v \right ) = 198.1527 \cdot E_v^{-0.8003}
        
        This will return ``float('nan')`` if no measurement was taken, such as during an I2C error.
        """
        return self._data[3]

    @property
    def L(self):
        r""" The reference to the left :any:`Image`"""
        return self._left

    @property
    def R(self):
        r""" The reference to the right :any:`Image`"""
        return self._right

    @property
    def imu(self):
        r"""The matching :any:`Imu` for this image. Since the capture of an image is synchronized with the IMU,
                no interpolation is needed."""
        i = self._dataset.raw.imu
        match = i[i[:, 0] == self.stamp]
        if match.size > 0: return Imu(self._dataset, match[0])

        raise ValueError("It seems that %s has no matching IMU value" % self)


class Imu(Value):
    r""" An imu value represents the measurement of IMU sensor at a specific time.

    Since it is a :any:`Value` you can use it to calculate transforms with it. Also 
    the IMU is synchronized in a way, that it measures exactly seven times 
    per image. Any :any:`Imu` value consist of three acceleration measurements in X, Y and Z 
    and three angular velocity measurements around the X, Y, and Z axis.

    .. seealso:: :any:`Interpolation` for the data frequency & :any:`Value` for the reference frame

    """

    @staticmethod
    def extrapolate(value, method='closest'):
        r"""
        Find a matching :any:`Imu` value for a certain :any:`Value` based on a extrapolation method

        :param value: The value for which to find a matching :any:`Imu` value
        :param method: An optional extrapolation method to determine the rules for a "match":

            * ``"closest"``: the image with the least difference to value.stamp is chosen
            * ``"next"``: the image with the next larger time stamp than value.stamp is chosen
            * ``"prev"``: the image with the next smaller time stamp than value.stamp is chosen
            * ``"exact"``: the image where value.stamp == image.stamp holds is chosen, None otherwise

        :return: The matching stereo image or None if no was found
        """
        if method == 'closest':
            f = value._dataset.raw.imu
            i = np.abs(f[:, 0] - value.stamp).argmin()
            return Imu(value._dataset, f[i, :])

        if method == 'next':
            f = value._dataset.raw.imu
            imu = f[f[:, 0] > value.stamp, :]
            if imu.size == 0: return None
            return Imu(value._dataset, imu[0])

        if method == 'prev':
            f = value._dataset.raw.imu
            imu = f[f[:, 0] < value.stamp, :]
            if imu.size == 0: return None
            return Imu(value._dataset, imu[-1])

        if method == 'exact':
            f = value._dataset.raw.imu
            imu = f[f[:, 0] == value.stamp, :]
            if imu.size == 0: return None
            return Imu(value._dataset, imu[0])

        raise ValueError(
            'Unknown extrapolation method: %s (supported are "closest", "next", "prev" and "exact")' % method)

    @staticmethod
    def interpolate(dataset, stamp, accelaration_interpolation=StereoTUM.Interpolation.linear,
                    angvelo_interpolation=StereoTUM.Interpolation.linear):
        r"""
        This function enables you to find the interpolated :any:`Imu` values of a record given a certain timestamp.

        :param dataset: the dataset which holds all imu values to interpolate over 
        :param float stamp: the time at which to interpolate (in seconds, with decimal places) 
        :param accelaration_interpolation: A predefined or custom interpolation function
        :param angvelo_interpolation: A predefined or custom interpolation function
        :return: A :any:`Imu`

        .. seealso:: :any:`Interpolation`
        """

        imu = dataset.raw.imu
        idx = np.searchsorted(imu[:, 0], stamp)
        acc = np.ones((1, 3)) * np.nan
        gyr = np.ones((1, 4)) * np.nan
        if idx != 0 and idx != imu.shape[0]:
            ta = accelaration_interpolation(imu[idx - 1, 0], imu[idx, 0], stamp)
            acc = (1 - ta) * imu[idx - 1, 1:4] + ta * imu[idx, 1:4]
            tg = angvelo_interpolation(imu[idx - 1, 0], imu[idx, 0], stamp)
            gyr = (1 - tg) * imu[idx - 1, 1:4] + tg * imu[idx, 1:4]

        return Imu(dataset, np.concatenate(([stamp], acc, gyr)))

    def __init__(self, dataset, data):
        if len(data) < 7: raise ValueError(
            "[%s] Data must have at least 7 entries [time1, acc3, gyro3] but has %d" % (self._dataset, len(data)))
        super(Imu, self).__init__(dataset, stamp=data[0], reference='imu')
        self._acc = np.array(data[1:4])
        self._gyro = np.array(data[4:7])

    @property
    def _previous(self):
        return Imu.extrapolate(self, method='prev')

    def __eq__(self, other):
        if not isinstance(other, Imu): return False
        return (super(Imu, self).__eq__(other)
                and np.allclose(self._acc, other._acc)
                and np.allclose(self._gyro, other._gyro))

    @property
    def acceleration(self):
        r"""
        The acceleration 3D vector [x,y,z] of this measurement as 
        `ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_ 
        in :math:`\frac{m}{s^2}`
        """
        return self._acc

    @property
    def angular_velocity(self):
        r"""
        The angular velocity 3D vector around [x,y,z] of this measurement as 
        `ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_
        in :math:`\frac{rad}{s}`
        """
        return self._gyro

    def stereo(self, shutter, extrapolation='closest'):
        r"""
        The matching stereo image for this :any:`Imu` measurement
        :param shutter: The shutter type of the images to find (``"global"``, ``"rolling"``, **not** ``"both"``) 
        :param extrapolation: An optional extrapolation method to determine the rules for a "match":

            * ``"closest"``: the image with the least difference to value.stamp is chosen
            * ``"next"``: the image with the next larger time stamp than value.stamp is chosen
            * ``"prev"``: the image with the next smaller time stamp than value.stamp is chosen
            * ``"exact"``: the image where ``value.stamp == image.stamp`` holds is chosen, None otherwise
        :return: The matching stereo image or None if no was found

        .. seealso:: :any:`StereoTUM.StereoImage.extrapolate`
        """
        return StereoImage.extrapolate(self, shutter, method=extrapolation)

    def groundtruth(self, position_interpolation=StereoTUM.Interpolation.linear,
                    orientation_interpolation=StereoTUM.Interpolation.slerp):
        r"""
        Find the matching :any:`GroundTruth` value for this :any:`Imu` value. Since the motion capture system and the 
        IMU sensor are not synced, we need to interpolate between ground truths by time stamp of this :any:`Imu` value.

        :param position_interpolation: a predefined or custom interpolation function to interpolate positions 
        :param orientation_interpolation: a predefined or custom interpolation function to interpolate quaternions
        :return: the matching interpolated ground truth

        .. seealso:: :any:`StereoTUM.GroundTruth.interpolate`
        """
        return GroundTruth.interpolate(self._dataset, self.stamp, position_interpolation, orientation_interpolation)


class GroundTruth(Value):
    r"""
    A ground truth is a :any:`Value` with the reference ``"world"``.
    The ground truth is taken with a higher frequency than the images (around 120 Hz), but slower than the `:any:Imu`. 
    Since the :any:`mocap` system is stationary in one room only, it might not cover the whole duration of the dataset 
    (depending on the sequence).
    """

    @staticmethod
    def interpolate(dataset, stamp, position_interpolation=StereoTUM.Interpolation.linear,
                    orientation_interpolation=StereoTUM.Interpolation.slerp):
        r"""
        This function enables you to find the interpolated groundtruth of a record given a certain timestamp.

        :param dataset: the dataset which holds all ground truth values to interpolate over 
        :param float stamp: the time at which to interpolate (in seconds, with decimal places) 
        :param position_interpolation: A predefined or custom interpolation function
        :param orientation_interpolation: A predefined or custom interpolation function
        :return: A :any:`GroundTruth` -Value

        .. seealso:: :any:`Interpolation`
        """
        poses = dataset.raw.groundtruth
        idx = np.searchsorted(poses[:, 0], stamp)
        p = np.ones((1, 3)) * np.nan
        q = np.ones((1, 4)) * np.nan
        if idx != 0 and idx != poses.shape[0]:
            t = (stamp - poses[idx - 1, 0]) / (poses[idx, 0] - poses[idx - 1, 0])

            # Compute previous values for interpolation method, if existant
            a0, b0 = np.zeros((3, 1)), np.zeros((3, 1))
            qa0, qb0 = np.array((1, 0, 0, 0)), np.array((1, 0, 0, 0))
            if idx - 1 >= 0:             a0, qa0 = poses[idx - 2, 1:4], poses[idx - 2, 4:8]
            if idx + 1 < poses.shape[0]: b0, qb0 = poses[idx + 1, 1:4], poses[idx + 1, 4:8]

            p = position_interpolation(poses[idx - 1, 1:4], poses[idx, 1:4], t, a0, b0)
            q = orientation_interpolation(poses[idx - 1, 4:8], poses[idx, 4:8], t, qa0, qb0)
        return GroundTruth(dataset, np.concatenate(([stamp], p, q)))

    @staticmethod
    def extrapolate(value, method='closest'):
        r"""
        Find a matching ground truth for a certain :any:`Value` based on a extrapolation method

        :param value: The value for which to find a matching imuvalue
        :param method: An optional extrapolation method to determine the rules for a "match":

            * ``"closest"``: the image with the least difference to value.stamp is chosen
            * ``"next"``: the image with the next larger time stamp than value.stamp is chosen
            * ``"prev"``: the image with the next smaller time stamp than value.stamp is chosen
            * ``"exact"``: the image where value.stamp == image.stamp holds is chosen, None otherwise

        :return: The matching stereo image or None if no was found
        """
        if method == 'closest':
            f = value._dataset.raw.groundtruth
            i = np.abs(f[:, 0] - value.stamp).argmin()
            return GroundTruth(value._dataset, f[i, :])

        if method == 'next':
            f = value._dataset.raw.groundtruth
            gt = f[f[:, 0] > value.stamp, :]
            if gt.size == 0: return None
            return GroundTruth(value._dataset, gt[0])

        if method == 'prev':
            f = value._dataset.raw.groundtruth
            gt = f[f[:, 0] < value.stamp, :]
            if gt.size == 0: return None
            return GroundTruth(value._dataset, gt[-1])

        if method == 'exact':
            f = value._dataset.raw.groundtruth
            gt = f[f[:, 0] == value.stamp, :]
            if gt.size == 0: return None
            return GroundTruth(value._dataset, gt[0])

        raise ValueError(
            'Unknown extrapolation method: %s (supported are "closest", "next", "prev" and "exact")' % method)

    def __init__(self, dataset, data):
        if len(data) < 8:
            raise ValueError(
                "Data must have at least 8 entries [time1, position3, orientation4] but has %d" % len(data))
        self._data = data
        super(GroundTruth, self).__init__(dataset, stamp=data[0], reference='world')

    @property
    def _previous(self):
        return GroundTruth.extrapolate(self, method='prev')

    def __eq__(self, other):
        if not isinstance(other, GroundTruth): return False
        return super(GroundTruth, self).__eq__(other) and np.allclose(self._data, other._data)

    @property
    def position(self):
        r"""The position of this ground truth as 3D `ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_"""
        return self._data[1:4]

    @property
    def quaternion(self):
        r"""The orientation in quaternion representation with the scalar (w) component as first element in a 4D `ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_"""
        return self._data[4:8]

    @property
    def rotation(self):
        r"""The rotation matrix only **WITHOUT** the translational part as 4x4 `ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_ """
        t = np.eye(4)
        t[0:3, 0:3] = tf.quaternions.quat2mat(self.quaternion)
        return t

    @property
    def translation(self):
        r"""The translation matrix only **WITHOUT** the rotational part as 4x4 `ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_"""
        t = np.eye(4)
        t[0:3, 3] = self.position
        return t

    @property
    def pose(self):
        r"""The complete pose of the ground truth including both translation and orientation"""
        return tf.affines.compose(
            T=self.position,
            R=tf.quaternions.quat2mat(self.quaternion),
            Z=np.ones(3),
            S=np.zeros(3)
        )

    @property
    def _transform(self):
        return np.linalg.inv(self.pose)

    def stereo(self, shutter, extrapolation='closest'):
        r"""
        Find a matching stereo image pair for this ground truth value.

        :param shutter: The shutter type of the images to find (``"global"``, ``"rolling"``, **not** ``"both"``) 
        :param extrapolation: An optional extrapolation method to determine the rules for a "match" (one of ``"closest"``, ``"next"``, ``"prev"``, ``"exact"``)
        :return: The matching stereo image or None if no was found

        .. seealso:: :any:`StereoTUM.StereoImage.extrapolate`
        """
        return StereoImage.extrapolate(self, shutter, method=extrapolation)
