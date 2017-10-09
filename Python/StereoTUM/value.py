#!/usr/bin/env python3

import numpy as np


class Value(object):
    r"""
    A Value represents every measurement of a sensor taken in a specific time instance.
    A camera might capture an :any:`Image`, an IMU will record :any:`ImuValue` or the motion capture system 
    :any:`GroundTruth`.
     
    All these values have some common properties in the StereoTUM:
    
    1) They are single, time-discrete values (see :any:`stamp <StereoTUM.Value.stamp>`)
    2) They are related to a certain reference frame (see :any:`reference <StereoTUM.Value.reference>`)
    3) They all have a transformation from its reference frame towards that of ``"cam1"`` (see :any:`transform <StereoTUM.Value.transform>`)
    
    
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
        print("T_cam1_2_image: %s" % image.transform)
        print("T_cam1_2_gt:    %s" % gt.transform)
        # BUT since they are "private" functions from value you should rather use something like this:
         
         
        # Since we know both transforms to "cam1", we can compute the transformation between the two
        T_image_2_gt = image >> gt
        T_image_2_gt = gt << image        # same as line above
        T_image_2_gt = image >> "world"   # also the same
        # T_image_2_gt = "world" << image # This fails, since the shift operators cannot be overloaded for strings in the first place
        
        T_gt_2_image = gt >> image
        T_gt_2_image = image << gt
        T_gt_2_image = gt >> "cam2"       # for example, if image was taken by "cam2"
        T_gt_2_image = "cam2" << "world"  # obviously this will also not work ...
    
    """
    def __init__(self, dataset, stamp, reference):
        self._dataset = dataset
        self._stamp = stamp
        if reference not in dataset._refs:
            raise ValueError("Cannot find the reference %s" % reference)

        self._reference = reference

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

    @property
    def transform(self):
        r"""
        The transformation from ``"cam1"`` to this value`s :any:`reference <StereoTUM.Value.reference>` as 4x4 `ndarray <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html>`_ homogenous matrix 
        """
        return np.array(self._dataset._refs[self._reference]['transform'])

    def __str__(self):
        return "%s (%s/%.2f)" % (type(self).__name__, self.reference, self.stamp)

    def __lshift__(self, parent) -> np.ndarray:
        if isinstance(parent, str):
            if parent not in self._dataset._refs:
                raise ValueError("Cannot find the (static) parent reference %s" % parent)
            tparent = np.array(self._dataset._refs[parent]['transform'])

        elif isinstance(parent, Value):
            tparent = parent.transform
        else:
            raise TypeError("Cannot only lookup transforms for type string or StereoTUM.Value")

        tchild = self.transform
        return np.dot(np.linalg.inv(tparent), tchild)

    def __rshift__(self, child) -> np.ndarray:
        if isinstance(child, str):
            if child not in self._dataset._refs:
                raise ValueError("Cannot find the (static) parent reference %s" % child)
            tchild = np.array(self._dataset._refs[child]['transform'])
        elif isinstance(child, Value):
            tchild = child.transform
        else:
            raise TypeError("Cannot only lookup transforms for type string or StereoTUM.Value")

        tparent = self.transform
        return np.dot(np.linalg.inv(tparent), tchild)


