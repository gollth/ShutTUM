#!/usr/bin/env python3

import cv2
import numpy as np
from StereoTUM.dataset import Dataset

# Parameters
path    = '../Tests/valid'  # The folder to the dataset to visualize
shutter = 'global'          # The shutter you want to show
speed   = .2                # The playback speed (0 for stills .. 1 for realtime)

# Create a dataset
dataset = Dataset(path)


# Iterate over all stereo images for the given shutter method
for stereo in dataset.cameras(shutter):

    # Create a combined stereo image
    display = np.concatenate((stereo.L.load(), stereo.R.load()), axis=1)

    # Write some information on it
    text = 'Frame %s | Stamp %s | L' % (stereo.ID, stereo.stamp)
    cv2.putText(display, text, org=(50, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 0))

    text = 'Frame %s | Stamp %s | R' % (stereo.ID, stereo.stamp)
    cv2.putText(display, text, org=(display.shape[1]//2, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 0))


    # Resize the display to fit the screen
    display = cv2.resize(display, (0,0), fx=.5, fy=.5)
    cv2.imshow('StereoTUM', display)

    # Do the spinning
    if speed == 0: cv2.waitKey(0)
    else:          cv2.waitKey(int(stereo.dt(ifunknown=.1) * 1000 / speed))

cv2.waitKey(0)  # wait until user terminates
