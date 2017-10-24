=======
Dataset
=======
All classes broadly associated with the dataset creation and manipulation

:any:`Interpolation`: Summary

.. autosummary::

   ~StereoTUM..Interpolation.cubic
   ~StereoTUM..Interpolation.linear
   ~StereoTUM..Interpolation.slerp

:any:`Dataset`: Summary

.. autosummary::

    ~StereoTUM.dataset.Dataset.__init__
    ~StereoTUM.dataset.Dataset.__getitem__
    ~StereoTUM.dataset.Dataset.cameras
    ~StereoTUM.dataset.Dataset.duration
    ~StereoTUM.dataset.Dataset.end
    ~StereoTUM.dataset.Dataset.exposure_limits
    ~StereoTUM.dataset.Dataset.gamma
    ~StereoTUM.dataset.Dataset.imu
    ~StereoTUM.dataset.Dataset.mocap
    ~StereoTUM.dataset.Dataset.raw
    ~StereoTUM.dataset.Dataset.resolution
    ~StereoTUM.dataset.Dataset.rolling_shutter_speed
    ~StereoTUM.dataset.Dataset.start
    ~StereoTUM.dataset.Dataset.times
    ~StereoTUM.dataset.Dataset.vignette



.. autoclass:: StereoTUM.Interpolation
   :members:


.. autoclass:: StereoTUM.dataset.Dataset
    :members:
    :undoc-members:

.. automethod:: StereoTUM.dataset.Dataset.__init__
.. automethod:: StereoTUM.dataset.Dataset.__getitem__
