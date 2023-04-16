"""
This module implements a function that takes in 2 images,
their descriptors and a match object between them, and draws
the matches onto an openvc matrix that we return.
"""
from typing import List

from PIL.Image import Image
import numpy as np
import cv2

from pyslam.image_processing.feature_descriptors.ImagePairFeatureMatches import (
    Match,
    ImagePairMatches,
)
from pyslam.image_processing.cv_pillow import (
    pillowToArray,
    arrayToPillowImage,
    PillowColorFormat,
)
from pyslam.image_processing.feature_descriptors.ImageFeatures import (
    ImageFeatures,
)
from pyslam.image_processing.feature_descriptors.Keypoint import (
    Keypoint,
)


def drawStereoMatches(
    imgOne: Image,
    imgTwo: Image,
    imgOneFeatures: ImageFeatures,
    imgTwoFeatures: ImageFeatures,
    matches: ImagePairMatches,
) -> Image:
    """
    This function uses OpenCV's drawing functionality to
    draw matches between images, and returns the image as
    a Pillow image.

    :param imgOne: The first Image as a Pillow Image.
    :param imgTwo: The first Image as a Pillow Image.
    :param imgOneFeatures: Features from the first image.
    :param imgTwoFeatures: Features from the first image.
    :param matches: Object containing matches we predicted accross both images.

    :return: Returns a Pillow image with the matches drawn onto it.
    """
    # First, we need both input images as numpy arrays
    imgOneCV2: np.ndarray = pillowToArray(imgOne)
    imgTwoCV2: np.ndarray = pillowToArray(imgTwo)

    # Here are the other 3 lists we need to pass into cv2 to
    # draw our matches
    imgOneKeypoints: List[cv2.KeyPoint] = [
        cv2.KeyPoint(
            currImgOneKeypoint.coords[0],
            currImgOneKeypoint.coords[1],
            1,
        )
        for currImgOneKeypoint in imgOneFeatures.keypoints
    ]
    imgTwoKeypoints: List[cv2.KeyPoint] = [
        cv2.KeyPoint(
            currImgTwoKeypoint.coords[0],
            currImgTwoKeypoint.coords[1],
            1,
        )
        for currImgTwoKeypoint in imgTwoFeatures.keypoints
    ]
    cv2Matches: List[cv2.DMatch] = [
        cv2.DMatch(match.imgOneIdx, match.imgTwoIdx, 1)
        for match in matches.matches
    ]

    # Now, we need to iterate through our matches,
    # and populate the match array
    # for match in matches.matches:
    #     cv2Matches.append(cv2.DMatch(match.imgOneIdx, match.imgTwoIdx, 1))

    #     # Now, find the keypoints corresponding to this match; add
    #     # to the keypoints arrays
    #     currImgOneKeypoint : Keypoint = imgOneFeatures.keypoints[match.imgOneIdx]
    #     currImgTwoKeypoint : Keypoint = imgTwoFeatures.keypoints[match.imgTwoIdx]

    #     imgOneKeypoints.append(

    #     )
    #     imgTwoKeypoints.append(
    #         cv2.KeyPoint(currImgTwoKeypoint.coords[0], currImgTwoKeypoint.coords[1], 1)
    #     )

    # Now, we have everything we need. Draw it, and get the cv2 mat
    drawnMat: np.ndarray = cv2.drawMatches(
        imgOneCV2,
        imgOneKeypoints,
        imgTwoCV2,
        imgTwoKeypoints,
        cv2Matches,
        None,
    )

    # Convert drawnMat to RGB
    drawnMatRGB: np.ndarray = cv2.cvtColor(
        drawnMat, cv2.COLOR_BGR2RGB
    )

    return arrayToPillowImage(drawnMatRGB, PillowColorFormat.RGB)
