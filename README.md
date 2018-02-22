![ShutTUM](https://raw.githubusercontent.com/gollth/ShutTUM/master/Docs/images/logo.png)

[![Build Status](https://travis-ci.org/gollth/ShutTUM.svg?branch=master)](https://travis-ci.org/gollth/ShutTUM)

A photometrically calibrated multi-shutter stereo dataset for benchmarking visual odometry algorithms.

* 40 Sequences ≥ 10 km of trajectories
* 2x stereo cameras 20 FPS using both global and rolling shutter
* Extrinsic, intrinsic and photometric calibration
* IMU with accelerometer and gyroscope
* Hardware synchronized cameras and IMU
* greyscale 1.3 MP images (JPEG)
* Full 6D ground truth by motion capture system provided at start and end of every sequence


## Installation
Clone this repo and install it via the python setup file.
```
git clone https://github.com/gollth/ShutTUM.git
cd ShutTUM
python setup.py install [--user]
python
>>> import ShutTUM
>>> help(ShutTUM)
```

## Examples
### Python
A simple example shows the following script. It will load a sequence and show all global images in an OpenCV window

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


### Analzyer
```
python ShutTUM/Examples/python/analyze.py --help
```

Use this script to analyze a sequence. It comes with a lot of options (see help). For some aspects ROS (such as ```groundtruth```) is needed.
You can:
* visualize frames
* print/plot frame drops
* show IMU values
* recreate ground truth data
* interpolate ground truth data to frames

### DSO runner
```
python ShutTUM/Examples/python/dsorunner.py --side L --options nogui=0 --shutter global one /path/to/sequence /path/to/results 
```

Use this script to run [DSO](https://github.com/JakobEngel/dso) on one or multiple sequences.
See the help options for more info. This script converts the files into the correct format for DSO, 
(in a .temp directory) and copies the results in the end to the desired output. Use ```--debug``` 
switch to see the converted files.

### ROS
Source the ShutTUM as a catkin workspace
```
source ShutTUM/Examples/ros/devel/setup.bash
```

#### Visualize a sequence in RViZ:
```
roslaunch shuttum play.launch sequence:=/path/to/sequences/01 
```
Now rviz should open and playback the images. This launch file takes the additional arguments:
* ```loop``` (bool, false) to restart the sequence automatically when it is finsihed
* ```start```/```end``` (float, 0, inf) from which to which time stamps to play the sequence
* ```is_calib_seq``` (bool, false) if the sequence is a calibration sequence
* ```rviz``` (bool, false) should rviz be started?

#### Create a bag file from a sequence:
```
roslaunch shuttum bagcreator.launch sequence:=/path/to/sequences/01 bag:=/path/to/result.bag
```
This will playback the sequence and record the data. Note, that buffer size of rosbag record 
is set to zero. When the playback node finishes, do not cancel this launch file immediately!
Wait until ```/path/to/result.bag``` does not grow anymore in size, then CTRL-C.
* ```rviz``` (bool, false) should rviz be started?
* ```decimating``` (bool, false) should the images be decimated? Useful for e.g. camera calibration. 
* ```decimation``` (int, 10) If decimating, by which factor should the images be throttled? Topic is ```.../image_raw/decimated```
* ```start```/```end``` (float, 0, inf) from which to which time stamps to play the sequence

### Camera Calibration
Calibration with [Kalibr](https://github.com/ethz-asl/kalibr):
```
roslaunch shuttum kalibr_cameras.launch bag:=/path/to/result.bag model:=fov
```
Launches camera calibration on the images ```/cam/{global|rolling}/{left|right}/image_raw/decimated```
in the bag file you provide. The distortion model parameter can be either ```fov```(default), ```radtan```
or ```equi``` (see [supported formate](https://github.com/ethz-asl/kalibr/wiki/supported-models)). When Kalibr 
finishes, it will its result files into ```~/.ros/```

### IMU Calibration
```
roslaunch stuttum kalibr_imu.launch bag:=/path/to/result.bag lens:=fisheye start:=5 end:=45
```
This launches the [Camera-IMU-Calibration](https://github.com/ethz-asl/kalibr/wiki/camera-imu-calibration)
* ```lens``` (either ```fisheye``` or ```standard```): The lens used by that sequence
* ```start```/```end``` (float, 0, 1000): beginning and end offset for picking up of the sequence in seconds
* ```imu``` (path, ```$(find shuttum)/params/imu.yaml```): custom [IMU configuration file](https://github.com/ethz-asl/kalibr/wiki/yaml-formats)
* ```target``` (path ```$(find shuttum)/params/aprilA0.yaml" />```): custom [Grid configureation](https://github.com/ethz-asl/kalibr/wiki/calibration-targets)

### Matlab
The matlab interface is very minimalistic and just puts the files in ```data/``` into arrays.
```
>> addpath /path/to/ShutTUM/Examples/matlab
>> help(shuttum)
>> sequence = shuttum('/path/to/sequences/01')
   
sequence = 

  struct with fields:

         frames: [5335×4 double]
            imu: [48031×7 double]
    groundtruth: [6812×8 double]
```


## Known Issues
The ground truth data and frames/IMU value are not precisely time
synchronized. When you run a SLAM or VO algorithm (such as [DSO](https://github.com/JakobEngel/dso))
you can use the trajectory to estimate the time shift.

With the help of the extrinsic calibration in ```data/params.yaml```
of each sequence you can correlate the odometry with the motion capture
trajectory and optimize for a time shift. Note that the ground truth poses
are in the **marker** frame. Then you can run
```
python ShutTUM/Examples/python/analyze.py /path/to/sequences/01 groundtruth extract --time-offset XXX
```
where _t_new = t_raw + time_offset_

This will recreate the ```data/ground_truth.csv``` file with the time offset applied.
The raw ground truth data file is in ```.logs/XXXXXXXX.bag```


Since this API assumes that the four cameras use rolling/global shutter
the namespaces are the same. For calibration sequences with four times the same
shutter, cam1/2 names are set to ```global``` and cam3/4 are set to ```rolling```
This is however only a naming convention and does not affect the data in any way.