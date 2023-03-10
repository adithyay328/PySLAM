from typing import Optional

import cv2
from PIL.Image import Image

from pyslam.capture import Sensor
from pyslam.capture.common.monocular_uncalibrated_camera.MonocularUncalibratedCameraMeasurement import (
    MonocularUncalibratedCameraMeasurement,
)
from pyslam.uid import UID
from pyslam.image_processing.cv_pillow import (
    PillowColorFormat,
    arrayToPillowImage,
)


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
            collorCorrectArray = (
                cv2.cvtColor(frame, self.openCVColorCode)
                if self.openCVColorCode != -1
                else frame
            )
            image: Image = arrayToPillowImage(
                collorCorrectArray, self.pillowTargetColor
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
