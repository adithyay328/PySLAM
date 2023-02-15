# A capture subclass that contains an opencv matrix capture from a
# camera. Doesn't care if the camera is calibrated or not. Also can be
# easily used for virtual cameras for testing purposes

from pyslam.capture.Capture import (
    Capture,
)


class StandardCameraCapture(Capture):
    """A camera capture sub-class that contains opencv
    matrices representing images."""

    def __init__(
        self,
        sensorWrapperUID,
        cvImgMatix,
    ):
        super().__init__(sensorWrapperUID)

        self.cvImgMatrix = cvImgMatix
