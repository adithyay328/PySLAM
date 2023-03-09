from typing import Optional
from datetime import datetime

from pyslam.capture import Measurement, MeasurementSource
from pyslam.image_processing.Image import Image
from pyslam.uid import UID


class MonocularUncalibratedCameraMeasurement(Measurement):
    """
    A measurement representing an image from an uncalibrated,
    monocular camera.

    :param uid: If already known, represents the UID of this
        measurement
    :param sourceUID: The UID of the MeasurementSource that
        yielded this measurement.
    :param image: The image object representing the image for
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
