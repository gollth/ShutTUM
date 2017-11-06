import sys
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

sys.path.append('/usr/stud/gollt/StereoTUM/')
from StereoTUM.dataset import Dataset

# Parameters
parser = ArgumentParser()
parser.add_argument('record', help="The record to visualize of the StereoTUM dataset")
parser.add_argument('--animation', action='store_true', help="If the plot shall be animated or shown as still")
args = parser.parse_args()

plt.ion()
f, ax = plt.subplots(2, sharex=True)
ax[0].set_title('Acceleration')

X   = np.full([3], np.nan)
ACC = np.full([3], np.nan)
GYR = np.full([3], np.nan)
names = ['X Y Z'.split(), 'pitch roll yaw'.split()]

def plot():
    for i in range(3):
        ax[0].plot(X[:,i], ACC[:,i], label=names[0][i])
        ax[1].plot(X[:,i], GYR[:,i], label=names[1][i])

    ax[0].set_title('Acceleration [m/s^2]')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].set_title('Angular Velocity [rad/s]')
    ax[1].legend()
    ax[1].grid(True)
    plt.xlabel('Time [s]')

# Create a dataset
dataset = Dataset(args.record)
for imu in dataset.imu:

    X   = np.vstack((X,   3*[imu.stamp]))
    ACC = np.vstack((ACC, imu.acceleration))
    GYR = np.vstack((GYR, imu.angular_velocity))

    if not args.animation: continue

    ax[0].cla()
    ax[1].cla()
    plot()
    plt.pause(imu.dt(.001))


if not args.animation:
    plot()
    plt.show(block=True)
