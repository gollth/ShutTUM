# ShutTUM

A photometrically calibrated multi-shutter stereo dataset for benchmarking visual odometry algorithms.

* 40 Sequences
* 2x stereo cameras 20 FPS using both global and rolling shutter
* Extrinsic, intrinsic and photometric calibration* IMU with accelerometer and gyroscope
* Hardware synchronized cameras and IMU
* greyscale 1.3 MP images (JPEG)
* Full 6D ground truth by motion capture system provided at start and end of every sequence


## Installation
Clone this repo and install it via the python setup file.
```
git clone https://github.com/gollth/ShutTUM.git
cd ShutTUM
python setup.py install
python
>>> import ShutTUM
>>> help(ShutTUM)
```

## Examples
### Python
```python
import cv2
from ShutTUM.sequence import Sequence

# Load a dataset's sequence
sequence = Sequence('path/to/folder/01')

# Iterate over all images captured by the global shutter cameras
for stereo in sequence.cameras('global'):
	if stereo.L is None: continue
    if stereo.R is None: continue
    
	print(stereo.ID)
    img = stereo.L.load()
    cv2.imshow('Left Image', img)
    cv2.waitKey(int(stereo.dt() * 1000))
```
See full python documentation [here](https://gollth.github.io/ShutTUM/)

### ROS
Source the ShutTUM as a catkin workspace
```
source ShutTUM/Examples/ros/devel/setup.bash
```
Visualize a sequence in RViZ:
```
roslaunch shuttum play.launch sequence:=/path/to/sequences/01 
```
Now rviz should open and playback the images. This launch file takes the additional arguments:
* ```loop``` (bool, false) to restart the sequence automatically when it is finsihed
* ```start```/```end``` (float, 0, inf) from which to which time stamps to play the sequence
* ```is_calib_seq``` (bool, false) if the sequence is a calibration sequence
* ```rviz``` (bool, false) should rviz be started?