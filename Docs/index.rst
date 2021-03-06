
.. image:: images/logo.png

A utility API to easily interact with the ShutTUM dataset by `Thore Goll <thore.goll@tum.de>`_.

.. toctree::
    :maxdepth: 2
    :hidden:

    sequence
    devices
    values
    downloads

---------
Structure
---------
.. image:: images/api.svg

-------
Classes
-------
.. autosummary::
     :toctree:

     ~ShutTUM.Interpolation
     ~ShutTUM.sequence.Sequence

     ~ShutTUM.devices.StereoCamera
     ~ShutTUM.devices.DuoStereoCamera

     ~ShutTUM.values.Value
     ~ShutTUM.values.GroundTruth
     ~ShutTUM.values.Image
     ~ShutTUM.values.StereoImage
     ~ShutTUM.values.Imu
