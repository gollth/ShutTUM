#!/usr/bin/env python

import numpy as np
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
    shutters = dict(map(lambda kv: (kv[0], kv[1][0].upper()), sequence.shutter_types.items()))

    symbol = lambda x: '%05d' % x if x > 0 else '-----'
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


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('sequence', help="The sequence number/identifier")
    part = parser.add_subparsers(help='Which part of the sequence do you want to analyze?', dest='part')
    frames = part.add_parser('frames', help="Cameras")
    imu    = part.add_parser('imu',    help="IMU")
    gt     = part.add_parser('groundtruth', help="Ground truth")

    # Frames Subparser
    frames_command = frames.add_subparsers(help='Analyze the frame drops made by each camera', dest='command')

    plotter = frames_command.add_parser('plot', help="Show a sparsity plot of the frame drops")

    printer = frames_command.add_parser('print', help="Print the frame drops to a file")
    printer.add_argument('--file', default=None, type=str, help="Log file where to print to (default stdout)")

    player  = frames_command.add_parser('play', help="Play a stereo image from this sequence")
    player.add_argument('shutter', choices=['global', 'rolling'], help="The shutter type you want to see")
    player.add_argument('--speed', type=float, default=1, help='Real time factor for the playback [default 1.0]')

    args = parser.parse_args()

    if args.part == 'frames':

        if args.command == 'print':
            drops, log = analyse_frame_drops(args.sequence, include_log=True)
            if args.file is None: print(log)

            else:
                with open(args.file) as file:
                    file.write(log)

        if args.command == 'plot':
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter

            drops = analyse_frame_drops(args.sequence)

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
        raise ValueError("Not yet implemented")
