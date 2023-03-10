from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic

from PIL.Image import Image

from pyslam.image_processing.feature_descriptors.Descriptor import (
    Descriptor,
)
from pyslam.image_processing.feature_descriptors.Keypoint import (
    Keypoint,
)

T = TypeVar("T", bound=Descriptor)


class DescriptorExtractor(Generic[T], ABC):
    """
    An abstract base class that represents
    a procedure that can extract descriptors
    from an image, given a set of features
    to extract on.

    While this seemingly implies that this scheme
    need to be separate to the keypoint/feature
    coordinate selection algorithm, no such
    requirement really exists; in the case
    of end-to-end feature and descriptor
    extractors, just do both internally,
    returning the features then the descriptors,
    one after the other.
    """

    @abstractmethod
    def getDescriptors(
        self, inImg: Image, inKeypoints: List[Keypoint]
    ) -> List[T]:
        """
        An abstract method that subclasses must implement; takes in an
        image and a list of Keypoints, and returns a list of Descriptors;
        the assumption is that the nth Descriptor is for the nth given
        Keypoint.

        :param inImg: The image to compute Descriptors for.
        :param inKeypoints: The list of Keypoints to compute
            descriptors for.

        :return: Returns a list of Descriptors. We expect it to have
            the same size as inKeypoints, with the Nth descriptor
            describing the Nth keypoint.
        """
        pass
