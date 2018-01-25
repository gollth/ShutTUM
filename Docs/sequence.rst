=======
Sequence
=======
All classes broadly associated with the dataset creation and manipulation

:any:`Interpolation`: Summary

.. autosummary::

   ~ShutTUM.Interpolation.cubic
   ~ShutTUM.Interpolation.linear
   ~ShutTUM.Interpolation.slerp

:any:`Sequence`: Summary

.. autosummary::

    ~ShutTUM.sequence.Sequence.__init__
    ~ShutTUM.sequence.Sequence.__getitem__
    ~ShutTUM.sequence.Sequence.cameras
    ~ShutTUM.sequence.Sequence.duration
    ~ShutTUM.sequence.Sequence.end
    ~ShutTUM.sequence.Sequence.exposure_limits
    ~ShutTUM.sequence.Sequence.gamma
    ~ShutTUM.sequence.Sequence.imu
    ~ShutTUM.sequence.Sequence.mocap
    ~ShutTUM.sequence.Sequence.raw
    ~ShutTUM.sequence.Sequence.resolution
    ~ShutTUM.sequence.Sequence.rolling_shutter_speed
    ~ShutTUM.sequence.Sequence.stereosync
    ~ShutTUM.sequence.Sequence.start
    ~ShutTUM.sequence.Sequence.times
    ~ShutTUM.sequence.Sequence.vignette



.. autoclass:: ShutTUM.Interpolation
   :members:


.. autoclass:: ShutTUM.sequence.Sequence
    :members:
    :undoc-members:

.. automethod:: ShutTUM.sequence.Sequence.__init__
.. automethod:: ShutTUM.sequence.Sequence.__getitem__
