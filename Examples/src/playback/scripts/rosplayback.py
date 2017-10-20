#!/usr/bin/env python

import tf
import sys
import time
import os.path as p
import numpy   as np
import rospy   as ros

from tf.transformations import translation_from_matrix, quaternion_from_matrix
from collections       import namedtuple
from std_msgs.msg      import Header, Float32
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg   import Image, Imu, CameraInfo
from cv_bridge         import CvBridge, CvBridgeError
sys.path.append('/usr/stud/gollt/StereoTUM/Python')
from StereoTUM.dataset import Dataset

# Intialize
ros.init_node('rosplayback')
ros.set_param('/use_sim_time', True)
bridge    = CvBridge()
dataset   = Dataset('~/0245')

# Create Publishers
clock = ros.Publisher('/clock', Clock, queue_size=10)
tffer = tf.TransformBroadcaster()
imu   = ros.Publisher('/imu', Imu, queue_size=10)
CamPub= namedtuple('CamPub', 'L R caml camr expl expr')
camg  = CamPub(L=ros.Publisher('/cam/global/L/image', Image, queue_size=1), 
	           R=ros.Publisher('/cam/global/R/image', Image, queue_size=1),
	           caml=ros.Publisher('/cam/global/L/camera_info', CameraInfo, queue_size=10),
	           camr=ros.Publisher('/cam/global/R/camera_info', CameraInfo, queue_size=10),
	           expl=ros.Publisher('/cam/global/L/exposure', Float32, queue_size=10),
	           expr=ros.Publisher('/cam/global/R/exposure', Float32, queue_size=10))
camr  = CamPub(L=ros.Publisher('/cam/rolling/L/image', Image, queue_size=1),
	           R=ros.Publisher('/cam/rolling/R/image', Image, queue_size=1),
	           caml=ros.Publisher('/cam/rolling/L/camera_info', CameraInfo, queue_size=10),
	           camr=ros.Publisher('/cam/rolling/R/camera_info', CameraInfo, queue_size=10),
	           expl=ros.Publisher('/cam/rolling/L/exposure', Float32, queue_size=10),
	           expr=ros.Publisher('/cam/rolling/R/exposure', Float32, queue_size=10))


def createheader(value): 
	return Header(stamp=ros.Time.from_sec(value.stamp), frame_id=value.reference)

def publishtf(value, fixed='world'):
	if value.reference == fixed: return
	pose = value << fixed
	tffer.sendTransform(
		translation_from_matrix(pose), quaternion_from_matrix(pose), 
		ros.Time.from_sec(value.stamp), value.reference, fixed)

def publishimage(img, cam, exp, value):
	msg = bridge.cv2_to_imgmsg(value.load(), 'mono8')
	msg.header = createheader(value)
	img.publish(msg)
	publishtf(value, 'cam1')
	msg = CameraInfo(width=res.width, height=res.height, 
		             distortion_model='fov', D=[value.distortion],
		             P=value.P.flatten().tolist())
	msg.header = createheader(value)
	cam.publish(msg)
	exp.publish(Float32(data=value.exposure))


res = dataset.resolution
laststamp = 0
for stamp in dataset.times:
	if ros.is_shutdown(): break

	g, r, i, t = dataset.lookup(stamp)

	# Times
	clock.publish(ros.Time.from_sec(stamp))
	
	# Images
	if g is not None:
		if g.L is not None: publishimage(camg.L, camg.caml, camg.expl, g.L)
		if g.R is not None: publishimage(camg.R, camg.camr, camg.expr, g.R)
	if r is not None:
		if r.L is not None: publishimage(camr.L, camr.caml, camr.expl, r.L)
		if r.R is not None: publishimage(camr.R, camr.camr, camr.expr, r.R)

	# IMU
	if i is not None:
		imu.publish(Imu(
			header=createheader(i),
			linear_acceleration_covariance=np.diag(i.acceleration).flatten().tolist(),
			angular_velocity_covariance   =np.diag(i.angular_velocity).flatten().tolist(),
			orientation_covariance=np.diag((-1,0,0)).flatten().tolist()	# according to message desc. we set first element to -1 since we dont have orientation
		))
		publishtf(i, 'cam1')


	# Spinning
	dt = stamp - laststamp
	laststamp = stamp
	if dt > 0: time.sleep(dt)


