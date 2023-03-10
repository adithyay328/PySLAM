from abc import ABC, abstractmethod
from typing import List

from PIL.Image import Image

from pyslam.image_processing.feature_descriptors.Keypoint import (
    Keypoint,
)


class KeypointExtractor(ABC):
    """
    An abstract base class that represents
    a procedure that can extract keypoints
    from an image.
    """

    @abstractmethod
    def getKeypoints(self, inImg: Image) -> List[Keypoint]:
        """
        An abstract method that subclasses must implement; takes in an
        image, and returns a list of Keypoints.

        :param inImg: The image to compute Keypoints for.

        :return: Returns a list of Keypoints.
        """
        pass
