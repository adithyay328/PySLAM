from typing import List

import cv2
import numpy as np
from PIL.Image import Image

from pyslam.image_processing.feature_descriptors.extractors.DescriptorExtractor import (
    DescriptorExtractor,
)
from pyslam.image_processing.feature_descriptors.extractors.KeypointExtractor import (
    KeypointExtractor,
)
from pyslam.image_processing.feature_descriptors.descriptors.ORB import (
    ORB,
)
from pyslam.image_processing.feature_descriptors.Keypoint import (
    Keypoint,
)
from pyslam.image_processing.cv_pillow import pillowToArray


class ORB_Detect_And_Compute(
    DescriptorExtractor[ORB], KeypointExtractor
):
    """
    An end-to-end ORB extractor that uses
    opencv's orb.detectAndCompute internally.

    :param numPoints: The number of keypoints to try and find
        and describe. No guarantee this number will actually be
        computed however.
    """

    def __init__(self, numPoints: int) -> None:
        self.numPoints: int = numPoints
        self.orb = cv2.ORB_create(nfeatures=numPoints)

        # Internal arrays to store keypoints and descriptors
        self.__keypoints: List[Keypoint] = []
        self.__orbDescs: List[ORB] = []

    def __compute(self, inImg: Image) -> None:
        """
        An internal function that computes and saves keypoints and
        their corresponding descriptors. Since this is an end-to-end
        extractor these are done together, but since the API treats these
        as separate steps, simply return them separately.
        """
        # Compute keypoints and descriptors with opencv
        kp: List[cv2.KeyPoint] = []
        des: np.ndarray = np.array([])

        # Convert image to a black and white matrix
        bwPIL: Image = inImg.convert("L")
        bwImgMat: cv2.Mat = pillowToArray(bwPIL)

        kp, des = self.orb.detectAndCompute(bwImgMat, None)

        # Convert both into PySLAM types
        for keyP in kp:
            self.__keypoints.append(
                Keypoint(np.array([keyP.pt[0], keyP.pt[1]]))
            )
        for idx in range(des.shape[0]):
            self.__orbDescs.append(ORB(des[idx]))

        # Done

    def getKeypoints(self, inImg: Image) -> List[Keypoint]:
        """
        Computes keypoints for the given image.

        :param inImg: The image to compute Keypoints for.

        :return: Returns a list of Keypoints.
        """
        # If we haven't yet computed keypoints and descriptors,
        # do that first
        if len(self.__keypoints) == 0:
            self.__compute(inImg)

        result = self.__keypoints

        # Clear internal array of keypoints
        self.__keypoints = []

        return result

    def getDescriptors(
        self, inImg: Image, inKeypoints: List[Keypoint]
    ) -> List[ORB]:
        """
        Returns all descriptors we computed internally.

        :param inImg: The image to compute Descriptors for.
        :param inKeypoints: The list of Keypoints to compute
            descriptors for.

        :return: Returns a list of Descriptors. We expect it to have
            the same size as inKeypoints, with the Nth descriptor
            describing the Nth keypoint.
        """
        if len(self.__orbDescs) != len(inKeypoints):
            raise ValueError(
                """Internal orb descriptors and given keypoints don't have same size; 
                these were not computed at the same time, and that is not a valid use
                of this end to end extractor."""
            )
        else:
            result = self.__orbDescs
            self.__orbDescs = []
            return result
