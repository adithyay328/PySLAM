from typing import Generic, TypeVar, List
import weakref

from pyslam.image_processing.Image import Image
from pyslam.image_processing.feature_descriptors.extractors.DescriptorExtractor import (
    DescriptorExtractor,
)
from pyslam.image_processing.feature_descriptors.Descriptor import (
    Descriptor,
)
from pyslam.image_processing.feature_descriptors.Keypoint import (
    Keypoint,
)
from pyslam.image_processing.feature_descriptors.extractors.KeypointExtractor import (
    KeypointExtractor,
)

T = TypeVar("T", bound=Descriptor)


class ImageFeatures(Generic[T]):
    """This class is responsible for all computations related to
    keypoints and descriptors for an image. It takes in extractors
    for keypoints and descriptors respectively, stores them internally,
    and provides common utilities such as keypoint normalization.

    :param inputImage: The image to run computations against.
    :param keypointExtractor: An extractor that can extract keypoints from our image
    :param descriptorExtractor: An extractor that can extract descriptors from our image,
      given the previously computed keypoints.
    """

    def __init__(
        self,
        inputImage: Image,
        keypointExtractor: KeypointExtractor,
        descriptorExtractor: DescriptorExtractor[T],
    ) -> None:
        # Storing our input image as a weakref; don't want to prevent garbage collection
        self.inputImage = weakref.ref(inputImage)

        self.keypointExtractor: KeypointExtractor = (
            keypointExtractor
        )
        self.descriptorExtractor: DescriptorExtractor = (
            descriptorExtractor
        )

        # Keypoints from the extractor
        self.keypoints: List[
            Keypoint
        ] = keypointExtractor.getKeypoints(inputImage)
        # Optional array to store normalized keypoints
        self.normalizedKeypoints: List[Keypoint] = []
        # Descriptor vectors
        self.descriptors: List[
            T
        ] = descriptorExtractor.getDescriptors(
            inputImage, self.keypoints
        )