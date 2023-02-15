# Contains definitions for a standard, uncalibrated raw camera sensor.

import cv2

from pyslam.capture.RawSensor import (
    RawSensor,
)
from pyslam.sensors.StandardCameraCapture import (
    StandardCameraCapture,
)


class UncalibratedFileCameraRawSensor(RawSensor):
    """Implements logic to capture frames from a standard camera i.e.
    a webcam on a laptop. Expects cameras to be available as a
    file like /dev/video0, and uses opencv to get image frames.
    Expects cameras to be uncalibrated
    """

    def __init__(
        self,
        cameraFName,
    ):
        # OpenCV capture object to use internally; will
        # be initialized in the activation function
        self.__cvCapture: cv2.VideoCapture = None

        # Store camera fName internally
        self.cameraFName = cameraFName

    def activateSensor(
        self,
    ):
        # Set internal capture object to the capture object;
        # starts recording
        self.__cvCapture = cv2.VideoCapture(self.cameraFName)

    def teardownSensor(
        self,
    ):
        # Release the internal capture object
        self.__cvCapture.release()

    def capture(
        self,
        sensorWrapperUID,
    ):
        # Get a frame from the cv2 capture object, then construct
        # a capture object
        # and send it out
        (
            ret,
            frame,
        ) = self.__cvCapture.read()

        return StandardCameraCapture(
            sensorWrapperUID,
            frame,
        )
