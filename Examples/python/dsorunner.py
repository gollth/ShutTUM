#!/usr/bin/env python3

import os
import sys
import shutil
import numpy as np
import os.path as p

sys.path.append('/usr/stud/gollt/StereoTUM/')
from StereoTUM.dataset import Dataset
from argparse import ArgumentParser
from subprocess import STDOUT, check_call, CalledProcessError


def play(sequence_path, shutter, side, debug=False, options=[], dso_prefix=''):
	r""" Let's DSO run for a specific camera on a sequence from the StereoTUM dataset

	:param sequence_path: (str) the path to the StereoTUM sequence
	:param shutter: {'global', 'rolling'} which shutter type of the sequence shall be used
	:param side: {'L', 'R'} to work with the left or right camera
	:param debug: (bool) prints the executing command & does not remove the .temp directory
	:param options: (list(str)) additional arguments for dso in the format [name]=[value] (e.g. quiet, nogui, nolog...). 
	:param dso_prefix: (str) a path prefix to where the 'dso_dataset' executable lies (default '')
	Caution with nogui=0 this function will not terminate until you close Pangolin

	:return: a 2D numpy array (shape Nx8 with N=number of images) with the columns [Time X Y Z W X Y Z]
	"""
	sequence = Dataset(sequence_path)

	temp = p.join(sequence_path, '.temp')
	if p.exists(temp): shutil.rmtree(temp)
	os.mkdir(temp)

	cams = sequence.cameras(shutter)
	
	# Create the camera.txt file for DSO
	calib = p.join(temp, 'camera.txt')
	sequence.stereosync = True
	firstframe = next(iter(cams))
	firstframe = firstframe.L if side == 'L' else firstframe.R
	with open (calib, 'w') as file:
		file.write('%.6f\t%.6f\t%.6f\t%.6f\t%.6f' % (
			firstframe.focal.x / sequence.resolution.width,
			firstframe.focal.y / sequence.resolution.height,
			firstframe.principle.x / sequence.resolution.width,
			firstframe.principle.y / sequence.resolution.height,
			firstframe.distortion
		))
		
		file.write(os.linesep)
		file.write('%d\t%d' % sequence.resolution)
		file.write(os.linesep)
		file.write('0.4\t0.53\t0.5\t0.5\t0')	# post-rectification equivilant pinhole model [fx fy px py omega]
		file.write(os.linesep)
		file.write('640\t480')


	with open (p.join(sequence_path, 'frames', 'times.txt'), 'w') as file:
		i = 0
		sequence.stereosync = False
		for image in sequence.cameras(shutter):
			i += 1
			cam = image.L if side == 'L' else image.R
			if cam is None: 
				if debug: print(" -> Frame drop %d, skipping..." % image.ID)
				continue
			file.write('%d\t%.6f\t%.3f' % (
				cam.ID,
				cam.stamp,
				cam.exposure
			))
			file.write(os.linesep)

	
	images = p.join(sequence_path, 'frames', firstframe.reference)
	gamma  = p.join(sequence_path, 'params', firstframe.reference, 'gamma.txt')
	vignette = p.join(sequence_path, 'params', firstframe.reference, 'vignette.png')
	try:
		if not filter(lambda o: o.startswith('nogui='), options): options.append('nogui=1')
		cmd = [
			p.join(dso_prefix, 'dso_dataset'),
			'files=%s' % images,
			'calib=%s' % calib,
			'gamma=%s' % gamma,
			'vignette=%s' % vignette,
			' '.join(options)
		]
		if debug: print(' '.join(cmd))
		DEVNULL = open(os.devnull, 'w')
		check_call(cmd, stdout=DEVNULL, stderr=STDOUT)

	except CalledProcessError:
		pass  # CTRL-C

	else:
		if not p.exists('result.txt'): return None

		results = np.genfromtxt('result.txt')
		results[:,[4,5,6,7]] = results[:,[7,4,5,6]]	# Quaternion xyzw -> wxyz
		return results

	finally:
		if not debug: shutil.rmtree(temp)

		
		

if __name__ == '__main__':


	parser = ArgumentParser()
	# required
	parser.add_argument('sequence', help="The sequence number/identifier")
	parser.add_argument('shutter', choices=['global', 'rolling'],
						help="Which shutter type of the sequence shall be used")
	parser.add_argument('side', choices=['L', 'R'],
						help="To work with the left or right camera")
	parser.add_argument('--options', default=[], nargs='*',
						help="Additional arguments for dso in the format [name]=[value] (e.g. quiet, nogui, nolog...). Caution with nogui=0 this script will not terminate until you close Pangolin")

	# optional
	parser.add_argument('--result', default=None, 
						help="The path for the csv file containing the DSO poses [defaults to <sequence>/data/dso-{global,rolling}-{L,R}(-<repeat>).csv]")
	parser.add_argument('--debug', action='store_true', 
						help="Prints the executing command & does not remove the .temp directory")
	parser.add_argument('--dsoprefix', default='',
						help="A path prefix to where the 'dso_dataset' executable lies (default '')")
	parser.add_argument('--repeats', default=1, type=int,
						help="How often to repeat the run")

	args = parser.parse_args()

	for rep in range(args.repeats):
		result = args.result
		if result is None: 
			repstr = ''
			if args.repeats > 1: repstr = '-%02d' % (rep+1)
			result = p.join(args.sequence, 'data', 'dso-%s-%s%s.csv' % (args.shutter, args.side, repstr))
		
		print('[DSO runner] Starting sequence %s for shutter "%s" (repetition %d)' % (args.sequence, args.shutter, rep+1))
		odometry = play(args.sequence, args.shutter, args.side, debug=args.debug, options=args.options, dso_prefix=args.dsoprefix)
		if odometry is None:
			print("[DSO runner] no results.txt has generated, skipping this run =(")
			continue

		title = 'Timestamp [s]\tPosition X [m]\tPosition Y [m]\tPosition Z[m]\tOrientation W\tOrientation X\tOrientation Y\tOrientation Z'
		print("[DSO runner] Saving odometry to %s" % result)
		np.savetxt(result, odometry, fmt='%.6f', delimiter='\t', header=title, comments='')
		
	print("[DSO runner] Finished")
		
		

