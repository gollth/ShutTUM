import numpy as np
import matplotlib.pyplot as plt
from StereoTUM.dataset import Dataset

# Parameters
path    = '../Tests/valid'  # The folder to the dataset to visualize
speed   = .2                # The playback speed (0 for stills .. 1 for realtime)

graph, = plt.plot([], [])
plt.autoscale(enable=True)

# Create a dataset
dataset = Dataset(path)
for imu in dataset.imu:
    # Update the plot
    graph.set_xdata(np.append(graph.get_xdata(), [imu.stamp, imu.stamp, imu.stamp]))
    graph.set_ydata(np.append(graph.get_ydata(), imu.acceleration))
    plt.draw()
    plt.pause(imu.dt(ifunknown=.1))



