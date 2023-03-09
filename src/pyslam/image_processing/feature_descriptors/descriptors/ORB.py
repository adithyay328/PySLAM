import numpy as np
import cv2

from pyslam.image_processing.feature_descriptors.Descriptor import (
    Descriptor,
)


class ORB(Descriptor):
    """
    A class representing an ORB descriptor. Distance
    is computed using hamming distance

    :param descriptorArr: A numpy array containing the
        contents/data for this descriptor.
    """

    def __init__(self, descriptorArr: np.ndarray) -> None:
        super().__init__(descriptorArr)

    def computeDistance(self, other: "ORB") -> float:
        """
        An abstract method that must be implemented that
        computes the distance between this descriptor's data
        and another descriptor's data.

        :param other: The descriptor to compute distance to.
        """
        return cv2.norm(self.data, other.data, cv2.NORM_HAMMING)
