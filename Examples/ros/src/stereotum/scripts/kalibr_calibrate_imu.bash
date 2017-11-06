#!/bin/bash
rosrun kalibr kalibr_calibrate_imu_camera \
	--bag $1 \
	--target $2 \
	--cams $3 \
	--imu $4 \
    --bag-from-to $5 $6
