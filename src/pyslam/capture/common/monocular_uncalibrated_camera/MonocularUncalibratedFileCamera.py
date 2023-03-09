from typing import Optional

import cv2

from pyslam.capture import Sensor
from pyslam.image_processing.Image import Image
from pyslam.capture.common.monocular_uncalibrated_camera.MonocularUncalibratedCameraMeasurement import (
    MonocularUncalibratedCameraMeasurement,
)
from pyslam.uid import UID


class MonocularUncalibratedFileCamera(
    Sensor[MonocularUncalibratedCameraMeasurement]
):
    """
    A sensor that can read from a monocular camera
    that's available as a file(for example, a camera
    at /dev/video0). Internally uses opencv for all
    capture logic.

    :param fName: The file name of the camera we want
      to capture from.
    :param openCVColorCode: The opencv color conversion code that, when used
      with cv2.cvtColor, takes the inputted mat and outputs the correct
      black and white image. An example would be cv.COLOR_BGR2GRAY, but
      obviously it varies based on which camera you're taking it from.
      Set to -1 if the source camera is black and white.
    :param uid: An optional UID object to use as this Sensor's UID.
    """

    def __init__(
        self,
        fName: str,
        openCVColorCode: int,
        uid: Optional[UID] = None,
    ):
        super().__init__(uid)

        self.fName: str = fName
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
            bwFrame = cv2.cvtColor(frame, self.openCVColorCode)
            image: Image = Image(
                self.openCVColorCode, frame, bwFrame
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
