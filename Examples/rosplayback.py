#!/usr/bin/env python

from StereoTUM.dataset import Dataset

dataset = Dataset('../Python/Tests/valid')

# Iterate over all imu values and find corresponding images
for observation in dataset.cameras('global'):

    print(observation.stamp)
    print(observation.angular_velocity)
