import cv2

from pyslam.sensors.StandardCameraCapture import StandardCameraCapture
from pyslam.sensors.UncalibratedFileCameraRawSensor import UncalibratedFileCameraRawSensor
from pyslam.core.uid import generateUID

cam = UncalibratedFileCameraRawSensor("/dev/video0")
cam.activateSensor()

cv2.imshow("frame", cam.capture(generateUID()).cvImgMatrix)
cv2.waitKey(2000)