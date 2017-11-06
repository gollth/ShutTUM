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
ros.init_node('bagcreator')
ros.set_param('/use_sim_time', True)
bridge    = CvBridge()
record    = ros.get_param('~record')
dataset   = Dataset()
loop      = ros.get_param('~loop', False) 
start     = ros.get_param('~start', None)
end       = ros.get_param('~end',   None)

msg = 'Playback started [%s]' % record
if loop:  msg += ' [LOOP]'
if start: msg += ' [START: %ss]' % start
if end:   msg += ' [END: %ss]' % end
ros.loginfo(msg)

# Create Publishers
clock = ros.Publisher('/clock', Clock, queue_size=10)
tffer = tf.TransformBroadcaster()
imu   = ros.Publisher('/imu', Imu, queue_size=10)
CamPub= namedtuple('CamPub', 'img cam exp')

def create_camera_publisher(prefix):
	return (prefix, CamPub(img=ros.Publisher(prefix + '/image', Image, queue_size=1),
		          cam=ros.Publisher(prefix + '/camera_info', CameraInfo, queue_size=10),
		          exp=ros.Publisher(prefix + '/exposure', Float32, queue_size=10)
		   ))

# Create four publishers, one for each camera
cams = dict(map(create_camera_publisher, ['/cam/global/L', '/cam/global/R', '/cam/rolling/L', '/cam/rolling/R']))

def createheader(value): 
	return Header(stamp=ros.Time.from_sec(value.stamp), frame_id=value.reference)

def publishtf(value, fixed='world'):
	if value.reference == fixed: return
	pose = value << fixed
	tffer.sendTransform(
		translation_from_matrix(pose), quaternion_from_matrix(pose), 
		ros.Time.from_sec(value.stamp), value.reference, fixed)

def publishimage(pub, value):
	msg = bridge.cv2_to_imgmsg(value.load(), 'mono8')
	msg.header = createheader(value)
	pub.img.publish(msg)
	publishtf(value, 'cam1')
	msg = CameraInfo(width=res.width, height=res.height, 
		             distortion_model='fov', D=[value.distortion],
		             P=value.P.flatten().tolist())
	msg.header = createheader(value)
	pub.cam.publish(msg)
	pub.exp.publish(Float32(data=value.exposure))


res = dataset.resolution
laststamp = 0

while not ros.is_shutdown():
	for data in dataset[start:end]:
		if ros.is_shutdown(): break

		# Times
		clock.publish(ros.Time.from_sec(data.stamp))
		
		# Images
		if data.global_ is not None:
			if data.global_.L is not None: publishimage(cams['/cam/global/L'], data.global_.L)
			if data.global_.R is not None: publishimage(cams['/cam/global/R'], data.global_.R)
		if data.rolling is not None:
			if data.rolling.L is not None: publishimage(cams['/cam/rolling/L'], data.rolling.L)
			if data.rolling.R is not None: publishimage(cams['/cam/rolling/R'], data.rolling.R)

		# IMU
		if data.imu is not None:
			imu.publish(Imu(
				header=createheader(data.imu),
				linear_acceleration_covariance=np.diag(data.imu.acceleration).flatten().tolist(),
				angular_velocity_covariance   =np.diag(data.imu.angular_velocity).flatten().tolist(),
				orientation_covariance=np.diag((-1,0,0)).flatten().tolist()	# according to message desc. we set first element to -1 since we dont have orientation
			))
			publishtf(data.imu, 'cam1')


		# Spinning
		dt = data.stamp - laststamp
		laststamp = data.stamp
		if dt > 0: time.sleep(dt)

	# if no looping was chosen, we break out the while loop 
	if not loop: break