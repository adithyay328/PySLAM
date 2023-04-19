"""

This file implements
a MonocularVideoExtractor, which is a MeasurementSource that
extracts frames from a video file, but exposes
them as if they were camera captures.

"""
from typing import Optional
from collections.abc import Iterable

import cv2
import numpy as np
import numpy
from PIL.Image import Image
import imageio.v3 as iio

from pyslam.capture import Measurement, Sensor
from pyslam.common_capture.UncalibratedMonocularCamera import MonocularUncalibratedCameraMeasurement
from pyslam.uid import UID
from pyslam.image_processing.cv_pillow import (
    PillowColorFormat,
    arrayToPillowImage,
)

class MonocularVideoExtractor(Sensor):
    """
    This class implements a MeasurementSource that
    extracts frames from a video file, but exposes
    them as if they were camera captures. Internally
    it uses scikit-video to read from the video file,
    but opencv can be used for color conversion.

    :param fName: The file name of the video file we want
        to extract frames from.
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

        # When capturing, contains a frequency; when not, it's
        # a None type
        self.captureFrequency: Optional[float] = None

        self.imageIoIterator = None

    def makeActive(self, captureFrequency: float) -> None:
        # Sets the capture frequency
        self.captureFrequency = captureFrequency

        # Get a generator that will yield frames from the video
        self.imageIoIterator = iio.imiter(self.fName)

    def getMeasurement(
        self,
    ) -> MonocularUncalibratedCameraMeasurement:
        # Make sure that our capture object
        # is instantiated correctly; we need to read
        # from it right now
        if self.imageIoIterator is None:
            raise ValueError(
                f"Cannot get measurement from {self.fName} because it is not active"
            )

        # Get the next frame, catching a StopIteration errror
        # if we get it, indicating that we've reached the end
        # of the video
        assert(self.imageIoIterator is not None)
        try:
            frame: np.ndarray = next(self.imageIoIterator)

            # Convert the frame to the correct color format
            colorCorrectArray = (
                cv2.cvtColor(frame, self.openCVColorCode)
                if self.openCVColorCode != -1
                else frame
            )

            # Convert the frame to a Pillow image
            image: Image = arrayToPillowImage(
                colorCorrectArray, self.pillowTargetColor
            )

            # Create a new measurement
            newMeasurement: MonocularUncalibratedCameraMeasurement = MonocularUncalibratedCameraMeasurement(
                None, self._uid, image, None
            )

            return newMeasurement
        
        except StopIteration:
            raise ValueError(
                f"Reached end of video at {self.fName}"
            )
    
    def leaveActive(self) -> None:
        # Close the video capture
        self.scikitVideoReader = None

        # Set the capture frequency to None
        self.captureFrequency = None