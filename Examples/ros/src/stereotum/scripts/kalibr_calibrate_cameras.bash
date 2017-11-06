#!/bin/bash
rosrun kalibr kalibr_calibrate_cameras \
	--bag $1 \
	--topics /cam/global/L/image/decimated /cam/global/R/image/decimated /cam/rolling/L/image/decimated /cam/rolling/R/image/decimated \
	--models pinhole-fov pinhole-fov pinhole-fov pinhole-fov \
	--target $2