=======
Sequence
=======
All classes broadly associated with the dataset creation and manipulation

:any:`Interpolation`: Summary

.. autosummary::

   ~StereoTUM.Interpolation.cubic
   ~StereoTUM.Interpolation.linear
   ~StereoTUM.Interpolation.slerp

:any:`Sequence`: Summary

.. autosummary::

    ~StereoTUM.dataset.Sequence.__init__
    ~StereoTUM.dataset.Sequence.__getitem__
    ~StereoTUM.dataset.Sequence.cameras
    ~StereoTUM.dataset.Sequence.duration
    ~StereoTUM.dataset.Sequence.end
    ~StereoTUM.dataset.Sequence.exposure_limits
    ~StereoTUM.dataset.Sequence.gamma
    ~StereoTUM.dataset.Sequence.imu
    ~StereoTUM.dataset.Sequence.mocap
    ~StereoTUM.dataset.Sequence.raw
    ~StereoTUM.dataset.Sequence.resolution
    ~StereoTUM.dataset.Sequence.rolling_shutter_speed
    ~StereoTUM.dataset.Sequence.stereosync
    ~StereoTUM.dataset.Sequence.start
    ~StereoTUM.dataset.Sequence.times
    ~StereoTUM.dataset.Sequence.vignette



.. autoclass:: StereoTUM.Interpolation
   :members:


.. autoclass:: StereoTUM.dataset.Sequence
    :members:
    :undoc-members:

.. automethod:: StereoTUM.dataset.Sequence.__init__
.. automethod:: StereoTUM.dataset.Sequence.__getitem__
