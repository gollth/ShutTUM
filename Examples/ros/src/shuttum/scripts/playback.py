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
from geometry_msgs.msg import Vector3
from sensor_msgs.msg   import Image, Imu, CameraInfo
from cv_bridge         import CvBridge, CvBridgeError
sys.path.append('/usr/stud/gollt/ShutTUM/')
from ShutTUM.sequence import Sequence

# Intialize
ros.init_node('playback')
ros.set_param('/use_sim_time', True)
bridge    = CvBridge()
sequence    = ros.get_param('~sequence')
dataset   = Sequence(sequence)
loop      = ros.get_param('~loop', False) 
start     = ros.get_param('~start', None)
end       = ros.get_param('~end',   None)
speed     = ros.get_param('~speed', 1)

msg = 'Playback started [%s]' % sequence
if loop:  msg += ' [LOOP]'
if start: msg += ' [START: %s s]' % start
else: start = None
if end:   msg += ' [END: %s s]' % end
else: end = None
if speed != 1: msg += ' [%.2fx]' % speed
if speed == 0: ros.logwarning('Speed is zero, results might not be as expected')
ros.loginfo(msg)


# Create Publishers
clock = ros.Publisher('/clock', Clock, queue_size=10)
tffer = tf.TransformBroadcaster()
imu   = ros.Publisher('/imu', Imu, queue_size=10)
CamPub= namedtuple('CamPub', 'img cam exp')

def create_camera_publisher(prefix):
	return (prefix, CamPub(img=ros.Publisher(prefix + '/image_raw', Image, queue_size=1),
				  cam=ros.Publisher(prefix + '/camera_info', CameraInfo, queue_size=10),
				  exp=ros.Publisher(prefix + '/exposure', Float32, queue_size=10)
		   ))

# Create four publishers, one for each camera
cams = dict(map(create_camera_publisher, ['/cam/global/left', '/cam/global/right', '/cam/rolling/left', '/cam/rolling/right']))

def createheader(value): 
	return Header(stamp=ros.Time.from_sec(value.stamp), frame_id=value.reference)

def publishtf(value, fixed, invert=False):
	if value.reference == fixed: return
	pose = value << fixed if not invert else value >> fixed
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
			if data.global_.L is not None: publishimage(cams['/cam/global/left'], data.global_.L)
			if data.global_.R is not None: publishimage(cams['/cam/global/right'], data.global_.R)
		if data.rolling is not None:
			if data.rolling.L is not None: publishimage(cams['/cam/rolling/left'], data.rolling.L)
			if data.rolling.R is not None: publishimage(cams['/cam/rolling/right'], data.rolling.R)

		# IMU
		if data.imu is not None:
			unknown = np.diag((-1,0,0)).flatten().tolist()	# according to message desc. we set first element to -1
			imu.publish(Imu(
				header=createheader(data.imu),
				linear_acceleration=Vector3(*data.imu.acceleration),
				linear_acceleration_covariance=unknown,
				angular_velocity=Vector3(*data.imu.angular_velocity),
				angular_velocity_covariance=unknown,
				orientation_covariance=unknown
			))
			publishtf(data.imu, 'cam1')

		# Ground truth
		if data.groundtruth is not None:
			publishtf(data.groundtruth, 'cam1', invert=True)
			publishtf(data.groundtruth.marker, 'cam1')


		# Spinning
		dt = data.stamp - laststamp
		laststamp = data.stamp
		if dt > 0: time.sleep(dt / speed)

	# if no looping was chosen, we break out the while loop 
	if not loop: break
	ros.loginfo('Playback restarting from beginning')
