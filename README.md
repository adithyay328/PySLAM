# PySLAM
This project implements approximate implementations
of quite a few common SLAM algorithms, all in Python and with GTSAM's python bindings.

Namely, the goal of this project is to reproduce, albeit not perfectly, the following common
SLAM algorithms:
- ORB-SLAMv1
- KimeraSLAM(ignoring the IMU pre-integration; this project assumes monocular cams with no IMU, atleast for the time being)