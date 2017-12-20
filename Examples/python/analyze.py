#!/usr/bin/env python
import math
import numpy as np
import os.path as p
import sys
sys.path.append('/usr/stud/gollt/StereoTUM/')
from StereoTUM.dataset import Dataset
from argparse import ArgumentParser


def analyze_frames(path, shutter, speed=1.):

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
		else:          cv2.waitKey(int(stereo.dt(ifunknown=.1) * 1000 / args.speed))

	cv2.waitKey(0)  # wait until user terminates


def analyse_frame_drops(path, include_log=False):
	r""" Returns an 2D array with the frames recorded by the cams

	The four columns correspond to cam1 .. cam4.
	The N rows correspond to the N triggered frames. The value of each
	cell shows, if the camera has captured this frame (by putting its caputured ID 
	into the cell) or if it dropped this frame (by leaving the cell at zero).

	When you use ``include_log`` a tuple will be returned, with the array as first part
	and a string as formatted table as second part.
	"""
	sequence = Dataset(args.sequence, stereosync=False)
	shutters = dict(map(lambda (k,v): (k, v[0].upper()), sequence.shutter_types.items()))

	symbol = lambda (x): '%05d' % x if x > 0 else '-----'
	def limit(msg, n):
		if len(msg) < n: return msg.ljust(n)
		return '%s...' % msg[0:n-3]

	frames = np.zeros((sequence.raw.frames.shape[0]+1, 4))	# skip zeroth frame
	gids = [0,1] if shutters['cam1'] == 'G' else [3,2]
	rids = [0,1] if shutters['cam1'] == 'R' else [3,2]
	for img in sequence.cameras('global'):
		if img.L is not None: frames[img.ID, gids[0]] = img.ID
		if img.R is not None: frames[img.ID, gids[1]] = img.ID
		
	for img in sequence.cameras('rolling'):
		if img.L is not None: frames[img.ID, rids[0]] = img.ID
		if img.R is not None: frames[img.ID, rids[1]] = img.ID

	if not include_log:
		return frames

	else:
		msg = ""
		msg += '=====================================================================\n'
		msg += '| StereoTUM Dataset Sequence: %s |\n' % limit(args.sequence, 37)
		msg += '+===================================================================+\n'
		msg += '|   Frames (%s triggered), dropped frames:                       |\n' % symbol(frames.shape[0]-1)
		msg += '+----------------+----------------+----------------+----------------+\n'
		msg += '|     cam1 (%s)   |    cam2 (%s)    |    cam3 (%s)    |    cam4 (%s)    |\n' % (
			shutters['cam1'], shutters['cam2'], shutters['cam3'], shutters['cam4'])
		msg += '+----------------+----------------+----------------+----------------+\n'

		for idx, line in enumerate(frames):
			if idx == 0: continue
			if np.all(line): continue

			msg += '|'
			for id in line: msg += '     %s      |' % symbol(id)
			if not np.any(line): msg += '  (frame %05d)\n' % idx
			else: 	msg += '\n'

		drops = (frames > 0).sum(axis=0)
		msg += '+----------------+----------------+----------------+----------------+\n'
		msg += '|'
		for drop in drops:
			msg += ' %05s (%02.1f%%)  |' % (drop, 100. * drop / (frames.shape[0]-1))

		msg += '\n+----------------+----------------+----------------+----------------+\n'

		return frames, msg


def analyze_imu(path):
	r""" Plots both angular velocities and linear accelarations over time from this sequence"""

	f, ax = plt.subplots(2, sharex=True)
	ax[0].set_title('Acceleration')

	X   = np.full([3], np.nan)
	ACC = np.full([3], np.nan)
	GYR = np.full([3], np.nan)
	names = ['X Y Z'.split(), 'pitch roll yaw'.split()]

	# Create a dataset
	dataset = Dataset(path)
	for imu in dataset.imu:
		X   = np.vstack((X,   3*[imu.stamp]))
		ACC = np.vstack((ACC, imu.acceleration))
		GYR = np.vstack((GYR, imu.angular_velocity))


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
	plt.show()


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. eucledian norm, along axis.
    >>> v0 = numpy.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    True
    >>> v0 = numpy.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float64)
    >>> unit_vector(v0, axis=1, out=v1)
    >>> numpy.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1.0]))
    [1.0]

    see: https://github.com/ros/geometry/blob/hydro-devel/tf/src/tf/transformations.py
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quaternions.
    >>> q0 = random_quaternion()
    >>> q1 = random_quaternion()
    >>> q = quaternion_slerp(q0, q1, 0.0)
    >>> numpy.allclose(q, q0)
    True
    >>> q = quaternion_slerp(q0, q1, 1.0, 1)
    >>> numpy.allclose(q, q1)
    True
    >>> q = quaternion_slerp(q0, q1, 0.5)
    >>> angle = math.acos(numpy.dot(q0, q))
    >>> numpy.allclose(2.0, math.acos(numpy.dot(q0, q1)) / angle) or \
        numpy.allclose(2.0, math.acos(-numpy.dot(q0, q1)) / angle)
    True

    see: https://github.com/ros/geometry/blob/hydro-devel/tf/src/tf/transformations.py
    """
    _EPS = np.finfo(float).eps * 4.0
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0

def interpolate_poses(times, gt_poses):
	new_gt_poses = np.zeros((times.shape[0],8))
	for i in range(times.shape[0]):
		timestamp = times[i]
		idx = np.searchsorted(gt_poses[:, 0], timestamp)
		if idx==0 or idx==gt_poses.shape[0]:
			new_gt_poses[i,0] = timestamp
			new_gt_poses[i,1:4] = np.nan
			new_gt_poses[i,4:8] = np.nan
		else:
			t = (timestamp - gt_poses[idx-1, 0])/(gt_poses[idx, 0] - gt_poses[idx-1, 0])
			p = (1-t)*gt_poses[idx-1, 1:4] + t*gt_poses[idx, 1:4]
			q = quaternion_slerp(gt_poses[idx-1, 4:8], gt_poses[idx, 4:8], t)
			new_gt_poses[i,0] = times[i]
			new_gt_poses[i,1:4] = p
			new_gt_poses[i,4:8] = q

	return new_gt_poses


if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('sequence', help="The sequence number/identifier")
	part = parser.add_subparsers(help='Which part of the sequence do you want to analyze?', dest='part')
	frames = part.add_parser('frame', help="Cameras")
	imu    = part.add_parser('imu',    help="IMU")
	gt     = part.add_parser('groundtruth', help="Ground truth")

	# Frames Subparser
	frames_command = frames.add_subparsers(help='Analyze the frame captured by each camera', dest='command')
	dropper = frames_command.add_parser('drops', help='Analyze the missing frames from each camera')
	dropper.add_argument('--plot', action='store_true', help="Show a sparsity plot of the frame drops")
	dropper.add_argument('--file', default=None, type=str, help="Log file where to print to (default stdout)")	
	player  = frames_command.add_parser('play', help="Play a stereo image from this sequence")
	player.add_argument('shutter', choices=['global', 'rolling'], help="The shutter type you want to see")
	player.add_argument('--speed', type=float, default=1, help='Real time factor for the playback [default 1.0]')

	# Ground Truth Subparser
	gt_command = gt.add_subparsers(help='Analyze the ground truth data')
	interpoler = gt_command.add_parser('interpolate', help="Interpolate ground truth poses to frame stamps")
	interpoler.add_argument('--std', type=float, default=3, help="How much std is seen as break in the timestamps [3]?")
	interpoler.add_argument('--no', type=int,   default=200, help="How many entries should be ignored around the time stamp gap [200 each]?")
	interpoler.add_argument('--plot', action='store_true', help="Plot the timestamps?")
	interpoler.add_argument('--nan',  action='store_true', help="Shall non-existant ground truth values be included as non or be omitted?")
	interpoler.add_argument('--result', default=None, help="Where to put the results [defaults to <sequence>/data/ground_truth_interpolated.csv]")

	args = parser.parse_args()	
	
	if args.part == 'frames':

		if args.command == 'drops':
			drops, log = analyse_frame_drops(args.sequence, include_log=True)
			if args.file is None: print(log)

			else: 
				with open(args.file) as file:
					file.write(log)

			if args.plot:
				import matplotlib.pyplot as plt
				from matplotlib.ticker import FuncFormatter

				def labeller(val,pos):
					return 'cam%d' % (val+1)

				fig = plt.figure()
				ax = fig.add_subplot(111)
				ax.spy(drops, aspect='auto')
				ax.xaxis.set_major_formatter(FuncFormatter(labeller))
				plt.ylabel('Frame')
				plt.show()

		if args.command == 'play':
			import cv2
			analyze_frames(args.sequence, args.shutter, args.speed)

	if args.part == 'imu':
		import matplotlib.pyplot as plt
		analyze_imu(args.sequence)

	if args.part == 'groundtruth':
		gt = p.join(args.sequence, 'data', 'ground_truth.csv')
		gt_poses = np.loadtxt(gt, skiprows=1)

		diff = np.diff(gt_poses[:,0])
		m    = np.mean(diff)
		sig  = np.std(diff)
		print('Timestamp: Mean: %s, Std: %s' % (m, sig))

		# Find the outlier in the deriv, indicating the time break
		ibreak = np.where(diff > m + args.std * sig)[0][0]
		limits = gt_poses[ibreak:ibreak+2, 0];
		
		# Delete all outliers as well as +- 20 entries around the first outlier
		deletes = range(ibreak - args.no, ibreak + args.no)
		print("Will delete +/-%d entries around timestamp[%d]:  %s" % (args.no, ibreak, gt_poses[ibreak,0]))
		gt_poses = np.delete(gt_poses, deletes, axis=0)

		timescsv = np.loadtxt(p.join(args.sequence, 'data', 'frames.csv'), skiprows=1)	    
		times = timescsv[:,0]
		
		result = interpolate_poses(times, gt_poses)
		outside = (limits[0] < result[:,0]) & (result[:,0] < limits[1])

		if args.nan:
			nanpose = np.zeros((np.sum(outside), 3+4))   # 3D Postion 4D Quaternion
			nanpose.fill(np.nan)

			print("Replacing all interpolated poses between %s ... %s with NaNs" % (limits[0], limits[1]))
			result[outside,1:] = nanpose
		else:
			print("Deleting all interpolated poses between %s ... %s " % (limits[0], limits[1]))
			result = result[~outside,:]
			result = np.delete(result, outside, axis=0)

		if args.result is None: args.result = p.join(args.sequence, 'data', 'ground_truth_interpolated.csv')
		print("Saving interpolated ground truth poses to %s" % args.result)
		np.savetxt(args.result, result, fmt='%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f', comments='',
			header='Timestamp [s]	Translation X [m]	Translation Y [m]	Translation Z [m]	Orientation W	Orientation X	Orientation Y	Orientation Z') 

		if args.plot:
			from matplotlib import pyplot as plt
			
			plt.plot(result[:,0], result[:,1], label='Translation X')
			plt.plot(result[:,0], result[:,2], label='Translation Y')
			plt.plot(result[:,0], result[:,3], label='Translation Z')
			plt.grid()
			plt.legend(loc='center')
			plt.show()
