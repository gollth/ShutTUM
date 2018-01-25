function dataset = shuttum(path)
%SHUTTUM Loads a record from the ShutTUM dataset from path
%   record = shuttum('path/to/record')
%   loads the record into a Struct with the fields 'frames', 'imu', and
%   'groundtruth', which correspond to their respective .csv files in the
%   record. 
%   
%   The frames array contains the columns frame ID, the timestamp 
%   in seconds, the frame's exposure in milliseconds and the estimated 
%   illuminence in lux. 
%
%   The imu array contains the columns timestamp in seconds, linear
%   acceleration in x, y & z axis of the IMU in Meter/seconds^2 followed by
%   angular velocities around the x (pitch), y (yaw) and z (roll) axis of 
%   the IMU in Meter/second.
%
%   Finally the groundtruth array contains the columns timestamp in
%   seconds, translational offset from world fixed coord system to the left
%   camera in Meters, followed by a quaternion specifing the rotational
%   transformation from world fixed coord system to the left camera.

    if ~exist(path, 'dir'), error(['Record ' path ' does not exist!']); end

    frames = dlmread(fullfile(path, 'data', 'frames.csv'), '\t', 2, 0);
    imu    = dlmread(fullfile(path, 'data', 'imu.csv'), '\t', 1, 0);
    gt     = dlmread(fullfile(path, 'data', 'ground_truth.csv'), '\t', 1, 0);


    dataset = struct('frames', frames, 'imu', imu, 'groundtruth', gt);

end

