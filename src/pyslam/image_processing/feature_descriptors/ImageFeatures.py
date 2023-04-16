from typing import Generic, TypeVar, List, Tuple

from PIL.Image import Image
import jax
import jax.numpy as jnp
import numpy as np


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


@jax.jit
def normalizeKeypointMatrix(
    inMat: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Given an input keypoint matrix, applies normalization, and returns a tuple of
    (normalized points, normalization matrix.)

    :param inMat: An input matrix of all keypoints, as heterogenous points.
        inMat[0] should be an array with 2 elements, which are the x and y coordinate
        of the first keypoint.

    :return: Returns a tuple of jnp arrays. The first is the normalized array containing all
        kepoint coordinates(same format as input paramater),
        and the second is the 3x3 homogenous matrix that you could use to normalize
        all 2D points in the way we did internally. Invert it to un-normalize.
    """

    # First, zero-mean
    meanPosition: jax.Array = jnp.mean(inMat, axis=0)

    inMatZeroMeaned: jax.Array = inMat - meanPosition

    # Now, compute average distance from origin
    averageMagnitude: jax.Array = jnp.mean(
        jnp.linalg.norm(inMatZeroMeaned, axis=1)
    )

    # Apply isotropic(equal on both dimensions) normalization, s.t. that average
    # magnitude is sqrt(2). Non-isotropic and isotropic seem to have the same performance,
    # but this is easier to do.
    scalingFactor: jax.Array = (2**0.5) / averageMagnitude
    inMatNormalized: jax.Array = inMatZeroMeaned * scalingFactor

    translationMatrix: jax.Array = jnp.array(
        [
            [1, 0, -1 * meanPosition[0]],
            [0, 1, -1 * meanPosition[1]],
            [0, 0, 1],
        ]
    )

    scalingMatrix: jax.Array = jnp.array(
        [[scalingFactor, 0, 0], [0, scalingFactor, 0], [0, 0, 1]]
    )

    normalizationMatrix: jax.Array = (
        scalingMatrix @ translationMatrix
    )

    return (inMatNormalized, normalizationMatrix)


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
        self.keypointExtractor: KeypointExtractor = (
            keypointExtractor
        )
        self.descriptorExtractor: DescriptorExtractor[
            T
        ] = descriptorExtractor

        # Keypoints from the extractor
        self.keypoints: List[
            Keypoint
        ] = keypointExtractor.getKeypoints(inputImage)
        # Descriptor vectors
        self.descriptors: List[
            T
        ] = descriptorExtractor.getDescriptors(
            inputImage, self.keypoints
        )

        # Optional array to store normalized keypoints
        self.normalizedKeypoints: List[Keypoint] = []
        # JNP array that goes from un-normalized to
        # normalized keypoints
        self.normalizeMat: jax.Array = jnp.array([])

    # TODO: Build a unit test that
    # verifies that the normalization
    # matrix is correct, by applying
    # it to a set of keypoints, and
    # verifying that the average distance
    # from the origin is sqrt(2) and zero-meaned.
    # Also check if its similar to the
    # return normalized point matrix.
    def buildNormalizedKeypoints(self) -> None:
        """
        Takes the current list of keypoints,
        and builds the corresponding
        set of normalizad keypoints.
        """
        if self.normalizedKeypoints != []:
            raise ValueError(
                "Normalized keypoints is already populated!"
            )

        # A JNP array of all our keypoints as hetergenous coordinate
        # vectors
        jnpKeypointArray: jax.Array = jnp.vstack(
            [kp.asHeterogenous() for kp in self.keypoints]
        )

        # Normalized keypoint array and normalization matrix
        (
            normalizedKeypointArr,
            jnpNormalMat,
        ) = normalizeKeypointMatrix(jnpKeypointArray)

        # Store normalization matrix internally
        self.normalizeMat = jnpNormalMat

        # Convert keypoint arr into a list of keypoints
        for i in range(normalizedKeypointArr.shape[0]):
            self.normalizedKeypoints.append(
                Keypoint(np.array(jnpKeypointArray[i]))
            )

        # Done
