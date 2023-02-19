# A capture subclass that contains an opencv matrix capture from a
# camera. Doesn't care if the camera is calibrated or not. Also can be
# easily used for virtual cameras for testing purposes

import cv2

from pyslam.capture.Capture import (
    Capture,
)

class MonocularCameraCapture(Capture):
    """A camera capture sub-class that contains opencv
    matrices representing images."""

    def __init__(
        self,
        sensorWrapperUID : str,
        colorFormat : str,
        sourceImgMatix: cv2.Mat,
        bwImgMat : cv2.Mat
    ):
        """
        :param sensorWrapperUID: The UID of the sensor wrapper this
          came from.
        :param colorFormat: The color format of the source image matrix
        :param sourceImgMatrix: The source image matrix(what came straigt from
          the camera)
        :param bwImgMat: The image matrix in black and white. Convenient since
          many of our computer vision procedures operate on BW images, and this
          allows us to ignore the source color format we got from the camera.
        """
        super().__init__(sensorWrapperUID)

        self.colorFormat = colorFormat
        self.sourceImgMatix = sourceImgMatix
        self.bwImgmat = bwImgMat