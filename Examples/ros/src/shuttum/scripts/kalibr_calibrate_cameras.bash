#!/bin/bash
rosrun kalibr kalibr_calibrate_cameras \
	--bag $1 \
	--topics /cam/global/left/image_raw/decimated /cam/global/right/image_raw/decimated /cam/rolling/left/image_raw/decimated /cam/rolling/right/image_raw/decimated \
	--models pinhole-$3 pinhole-$3 pinhole-$3 pinhole-$3 \
	--target $2