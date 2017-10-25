#!/usr/bin/env python

import rospy   as ros
from sensor_msgs.msg   import Image

# Intialize
ros.init_node('decimator')
topic = ros.get_param('~topic')
decimation = ros.get_param('~decimation')
pub = ros.Publisher(topic + '/decimated', Image, queue_size=1)

counter = 0
def callback(msg):
	global counter
	counter += 1
	if counter < decimation: return

	pub.publish(msg)
	counter = 0

ros.Subscriber(topic, Image, callback)
ros.spin()