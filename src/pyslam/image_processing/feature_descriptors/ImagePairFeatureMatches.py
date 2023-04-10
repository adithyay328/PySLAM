"""
This module contains types and logic related to computing
matching pairs of Features accross Images.
"""
from typing import List, Tuple

import cv2
import numpy as np

from pyslam.image_processing.feature_descriptors.ImageFeatures import (
    ImageFeatures,
)
from pyslam.image_processing.feature_descriptors.descriptors.ORB import (
    ORB,
)

from pyslam.image_processing.feature_descriptors.Keypoint import Keypoint
from pyslam.optim.ransac.RANSACDataset import RANSACDataset


class Match:
    """
    A basic type representing a Match found between 2 images.
    One image needs to be determined as image one, and the other
    is image two. This object contains the indices in both images.

    :param imgOneIdx: The index of the matching descriptor in the first
        image.
    :param imgTwoIdx: The index of the matching descriptor in the second
        image.
    """

    def __init__(self, imgOneIdx: int, imgTwoIdx: int) -> None:
        self.imgOneIdx: int = imgOneIdx
        self.imgTwoIdx: int = imgTwoIdx


class ImagePairMatches:
    """
    A class that computes matches of Features across two images.
    Uses FLANN from OpenCV internally.

    :param imageFeaturesOne: The ImageFeatures object corresponding
        to the first image.
    :param imageFeaturesTwo: The ImageFeatures object corresponding
        to the second image.
    """

    def __init__(
        self,
        imageFeaturesOne: ImageFeatures,
        imageFeaturesTwo: ImageFeatures,
    ):
        self.imageFeaturesOne: ImageFeatures = imageFeaturesOne
        self.imageFeaturesTwo: ImageFeatures = imageFeaturesTwo

        self.matches: List[Match] = []

    def computeMatches(self) -> None:
        """
        Runs the internal matching logic to compute matches
        accross images.
        """

        # Before doing anything else, we need both sets of descriptors
        # as numpy arrays
        imageOneDescArray: np.ndarray = np.vstack(
            [
                descriptor.data
                for descriptor in self.imageFeaturesOne.descriptors
            ]
        )
        imageTwoDescArray: np.ndarray = np.vstack(
            [
                descriptor.data
                for descriptor in self.imageFeaturesTwo.descriptors
            ]
        )

        # TODO: Might be bad design to have all the matching logic as part of this class'
        # implementation; might be better to move this into configurable classes that
        # can do matching for us, but for now this isn't the worst offense.

        if self.matches != []:
            raise ValueError(
                "Matches have already been computed!"
            )
        elif type(self.imageFeaturesOne.descriptors[0]) == ORB:
            # Configure FLANN as OpenCV reccomends for ORB
            FLANN_INDEX_LSH: int = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,
                key_size=12,
                multi_probe_level=1,
            )
            flann = cv2.FlannBasedMatcher(index_params)

            # Get a bunch of OpenCV match objects; we want 2 matches
            # per query, so we can run Lowe's Ratio Test
            matches = flann.knnMatch(
                imageOneDescArray, imageTwoDescArray, k=2
            )

            for match in matches:
                if len(match) != 2:
                    continue
                matchOne, matchTwo = match
                # Lowe's Ratio Test here; check if
                # the first match has a distance that
                # is noticably smaller than matchTwo,
                # since that indicates that it is a
                # decisive match, rather than just another
                # bad match
                LOWES_DISTANCE_REQUIREMENT: float = 0.8
                if (
                    matchOne.distance
                    < LOWES_DISTANCE_REQUIREMENT
                    * matchTwo.distance
                ):
                    self.matches.append(
                        Match(
                            matchOne.queryIdx, matchOne.trainIdx
                        )
                    )
        else:
            raise NotImplementedError
    
    def toRANSACDataset(self, normalized : bool) -> RANSACDataset[Tuple[Keypoint, Keypoint]]:
        """
        Constructs a RANSAC dataset from the matches
        that can be used to compute an epipolar model.

        :param normalized: Whether or not to use
            normalized keypoints for the RANSAC dataset.

        :return: A RANSACDataset object containing the matched
            keypoints.
        """
        if normalized:
            assert len(self.imageFeaturesOne.normalizedKeypoints) > 0

        data : List[Tuple[Keypoint, Keypoint]] = []
        for match in self.matches:
            if normalized:
                data.append( (self.imageFeaturesOne.normalizedKeypoints[match.imgOneIdx], self.imageFeaturesTwo.normalizedKeypoints[match.imgTwoIdx]) )
            else:
                data.append( (self.imageFeaturesOne.keypoints[match.imgOneIdx], self.imageFeaturesTwo.keypoints[match.imgTwoIdx]) )
        dataset : RANSACDataset[Tuple[Keypoint, Keypoint]] = RANSACDataset(data, None)
        return dataset
