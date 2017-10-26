#!/usr/bin/env python3
import sys
import cv2
import numpy as np
from argparse import ArgumentParser

sys.path.append('/usr/stud/gollt/StereoTUM/')
from StereoTUM.dataset import Dataset

# Parameters
# Parameters
parser = ArgumentParser()
parser.add_argument('record', help="The record to visualize of the StereoTUM dataset")
parser.add_argument('shutter', choices=['global', 'rolling'], help="The shutter type you want to see")
parser.add_argument('--speed', type=float, default=1, help='Real time factor for the playback [default 1.0]')
args = parser.parse_args()

# Create a dataset
dataset = Dataset(args.record)


# Iterate over all stereo images for the given shutter method
for stereo in dataset.cameras(args.shutter):

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
    if args.speed == 0: cv2.waitKey(0)
    else:               cv2.waitKey(int(stereo.dt(ifunknown=.1) * 1000 / args.speed))

cv2.waitKey(0)  # wait until user terminates
