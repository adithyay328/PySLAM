from typing import Optional
from datetime import datetime

from PIL.Image import Image
import cv2

from pyslam.capture import Measurement, Sensor
from pyslam.uid import UID
from pyslam.image_processing.cv_pillow import (
    PillowColorFormat,
    arrayToPillowImage,
)


"""
This module contains Measurement Types and Sensor types
related to capturing from a monocular, uncalibrated
camera.
"""


class MonocularUncalibratedCameraMeasurement(Measurement):
    """
    A measurement representing an image from an uncalibrated,
    monocular camera.

    :param uid: If already known, represents the UID of this
        measurement
    :param sourceUID: The UID of the MeasurementSource that
        yielded this measurement.
    :param image: The pillow image object representing the image for
        this measurement.
    :param timestamp: If already known, represents the timestamp
        of this measurement.
    """

    def __init__(
        self,
        uid: Optional[UID],
        sourceUID: UID,
        image: Image,
        timestamp: Optional[datetime] = None,
    ) -> None:
        super().__init__(uid, sourceUID, timestamp)

        self.image: Image = image


class MonocularUncalibratedFileCamera(
    Sensor[MonocularUncalibratedCameraMeasurement]
):
    """
    A sensor that can read from a monocular camera
    that's available as a file(for example, a camera
    at /dev/video0). Internally uses opencv for all
    capture logic, but converts to Pillow afterwards,
    as that is the image format we default to for
    storage.

    :param fName: The file name of the camera we want
      to capture from.
    :param pillowTargetColor: The target color format we will use to store the
      captured images as Pillow images. This also determins the cv color conversion
      code you must provide.
    :param openCVColorCode: The opencv color conversion code that, when used
      with cv2.cvtColor, takes the mat off the sensor and outputs a matrix of the correct
      color format. "Correct" is based on what you passed in for "pillowTargetColor";
      if you passed in "L", you need a conversion code to convert to grayscale,
      and if you passed in "RGB", you need a conversion code to convert to RGB.
      Pass in a -1 if nothing needs to be done, and the color is already correct.
    :param uid: An optional UID object to use as this Sensor's UID.
    """

    def __init__(
        self,
        fName: str,
        pillowTargetColor: PillowColorFormat,
        openCVColorCode: int,
        uid: Optional[UID] = None,
    ) -> None:
        super().__init__(uid)

        self.fName: str = fName
        self.pillowTargetColor: PillowColorFormat = (
            pillowTargetColor
        )
        self.openCVColorCode: int = openCVColorCode

        # This will contain a cv2 VideoCapture when the sensor
        # is capturing, otherwise it's a None type
        self.cv2VideoCap: Optional[cv2.VideoCapture] = None
        # When capturing, contains a frequency; when not, it's
        # a None type
        self.captureFrequency: Optional[float] = None

    def makeActive(self, captureFrequency: float) -> None:
        self.captureFrequency = captureFrequency

        # Start cv2 video capture
        self.cv2VideoCap = cv2.VideoCapture(self.fName)

    def getMeasurement(
        self,
    ) -> MonocularUncalibratedCameraMeasurement:
        # Make sure that our capture object
        # is instantiated correctly; we need to read
        # from it right now
        if type(self.cv2VideoCap) != cv2.VideoCapture:
            raise ValueError(
                f"Expected cv2VideoCap to have type cv2.VideoCapture, got {type(self.cv2VideoCap)}"
            )

        cv2VideoCap: cv2.VideoCapture = self.cv2VideoCap

        ret, frame = cv2VideoCap.read()

        if not ret:
            raise ValueError(
                f"CV2 failed to capture a frame from camera at {self.fName}"
            )
        else:
            colorCorrectArray = (
                cv2.cvtColor(frame, self.openCVColorCode)
                if self.openCVColorCode != -1
                else frame
            )
            image: Image = arrayToPillowImage(
                colorCorrectArray, self.pillowTargetColor
            )

            newMeasurement: MonocularUncalibratedCameraMeasurement = MonocularUncalibratedCameraMeasurement(
                None, self._uid, image, None
            )

            return newMeasurement

    def leaveActive(self) -> None:
        # Reset all internal variables to None
        if type(self.cv2VideoCap) != cv2.VideoCapture:
            raise ValueError(
                f"Expected cv2VideoCap to have type cv2.VideoCapture, got {type(self.cv2VideoCap)}"
            )

        cv2VideoCap: cv2.VideoCapture = self.cv2VideoCap
        cv2VideoCap.release()

        # Now, set capture rate to none
        self.captureFrequency = None
