"""
Odometry is defined as the use of data
from various kinds of sensors, such as IMUs,
cameras and LiDar point clouds, to estimate
relative motion between 2 poses over time.
As such, all logic relevant to odometry is
stored in this module.

When we say "Pose", we are concerned with the robot/
sensor's position and orientation w.r.t some coordinate
frame with a defined origin. We are usually interested
in the 3D case, but this module also allows use in 2D
SLAM problems.
"""
from .PoseTransformSource import PoseTransformSource
