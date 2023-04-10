from typing import List, Tuple

import cv2
from PIL.Image import Image
import numpy as np

from pyslam.epipolar_core import (
    FundamentalMatrix,
    HomographyMatrix,
)
from pyslam.image_processing.feature_descriptors.ImageFeatures import (
    ImageFeatures,
)
from pyslam.image_processing.feature_descriptors.ImagePairFeatureMatches import (
    ImagePairMatches, Match
)
from pyslam.image_processing.cv_pillow import (
    pillowToArray,
    arrayToPillowImage,
    PillowColorFormat,
)

from pyslam.optim.ransac import (
    RANSACDataset,
    RANSACModel,
    RANSACEstimator,
)

from pyslam.image_processing.feature_descriptors.Keypoint import Keypoint

"""
Contains functions and logic for visualizations
related to epipolar models.
"""


# Lifted this from opencv tutorial
def drawlines(img1, img2, lines, pts1, pts2):
    """img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines"""
    r, c, _ = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 3, color, -1)
        img2 = cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 3, color, -1)
    return img1, img2


def drawEpipolarLines(
    fMat: FundamentalMatrix,
    imgOne: Image,
    imgOneFeatures: ImageFeatures,
    imgTwo: Image,
    imgTwoFeatures: ImageFeatures,
    matches : ImagePairMatches,
) -> Image:
    """
    Given two images and their respective features, draw the epipolar lines
    onto the images, horizontally concatenate them and return them.

    :param fMat: The Fundamental Matrix to use for drawing epipolar lines
    :param imgOne: The first image
    :param imgOneFeatures: The features for the first image. We assume these
        to have their normalization matrix already built, since you should
        have used that for estimating the fMatrix.
    :param imgTwo: The second image
    :param imgTwoFeatures: The features for the second image. We assume these
        to have their normalization matrix already built, since you should
        have used that for estimating the fMatrix.
    :param matches: The matches between the two images. We use these
        to only visualize inliers.

    :return: Returns a Pillow image with all epipolar lines drawn
        onto it. The left half is the first image, the right half
        is the second image.
    """
    cv2MatOne: cv2.Mat = pillowToArray(imgOne)
    cv2MatTwo: cv2.Mat = pillowToArray(imgTwo)

    # Extract color mode of the input image
    assert imgOne.mode == imgTwo.mode
    assert imgOne.mode in ["RGB", "L"]
    colorMode = (
        PillowColorFormat.RGB
        if imgOne.mode == "RGB"
        else PillowColorFormat.L
    )

    # Get the un-normalized fundamental matrix
    # fMatUnNorm: np.ndarray = fMat.unnormalize(
    #     np.array(imgOneFeatures.normalizeMat),
    #     np.array(imgTwoFeatures.normalizeMat),
    # )

    # Get all inlier points from the F-matrix
    ransacDataset : RANSACDataset[Tuple[Keypoint, Keypoint]] = matches.toRANSACDataset(True)
    inlierIndices : np.ndarray = fMat.findInlierIndices(ransacDataset)
    print(f"Found {len(inlierIndices)} inliers")

    inlierPtsOne : np.ndarray = np.vstack([imgOneFeatures.keypoints[i].asHomogenous() for i in inlierIndices])
    inlierPtsTwo : np.ndarray = np.vstack([imgTwoFeatures.keypoints[i].asHomogenous() for i in inlierIndices])

    # Draw epipolar lines using the opencv snippet
    # above; first, convert all keypoints to numpy arrays
    imageTwoEpipolarLines: np.ndarray = (
        fMat.matrix @ inlierPtsOne.T
    ).T

    imageOneEpipolarLines: np.ndarray = (
        fMat.matrix.T @ inlierPtsTwo.T
    ).T

    # Now, compute keypoints, but this time as
    # heterogeneous coordinates
    imageOneHeteterogeneousKeypoints: np.ndarray = np.array(
        [kp.asHeterogenous() for kp in imgOneFeatures.keypoints]
    )
    imageTwoHeteterogeneousKeypoints: np.ndarray = np.array(
        [kp.asHeterogenous() for kp in imgTwoFeatures.keypoints]
    )

    # Draw the epipolar lines and get resultant images
    imageOneWithLines, _ = drawlines(
        cv2MatOne,
        cv2MatTwo,
        imageOneEpipolarLines,
        imageOneHeteterogeneousKeypoints,
        imageTwoHeteterogeneousKeypoints,
    )
    imageTwoWithLines, _ = drawlines(
        cv2MatTwo,
        cv2MatOne,
        imageTwoEpipolarLines,
        imageTwoHeteterogeneousKeypoints,
        imageOneHeteterogeneousKeypoints,
    )

    # Concatenate the images horizontally
    concatenatedImage = np.hstack((imageOneWithLines, imageTwoWithLines))

    # Convert the concatenated image back to a Pillow image
    return arrayToPillowImage(concatenatedImage, colorMode)

def drawHomographyHypotheses(
    hMat: HomographyMatrix,
    imgOne: Image,
    imgOneFeatures: ImageFeatures,
    imgTwo: Image,
    imgTwoFeatures: ImageFeatures,
    matches : ImagePairMatches,
) -> Image:
    """
    Given two images and their respective features, draw the homography
    hypotheses onto the images, horizontally concatenate them and return them.

    :param hMat: The Homography Matrix to use for drawing homography
        hypotheses
    :param imgOne: The first image
    :param imgOneFeatures: The features for the first image. We assume these
        to have their normalization matrix already built, since you should
        have used that for estimating the fMatrix.
    :param imgTwo: The second image
    :param imgTwoFeatures: The features for the second image. We assume these
        to have their normalization matrix already built, since you should
        have used that for estimating the fMatrix.
    :param matches: The matches between the two images. We use these
        to only visualize inliers.

    :return: Returns a Pillow image with all homography hypotheses drawn
        onto it. The left half is the first image, the right half
        is the second image.
    """
    cv2MatOne: cv2.Mat = pillowToArray(imgOne)
    cv2MatTwo: cv2.Mat = pillowToArray(imgTwo)

    # Extract color mode of the input image
    assert imgOne.mode == imgTwo.mode
    assert imgOne.mode in ["RGB", "L"]
    colorMode = (
        PillowColorFormat.RGB
        if imgOne.mode == "RGB"
        else PillowColorFormat.L
    )

    #Get the un-normalized homography matrix
    # hMatUnNorm: np.ndarray = hMat.unnormalize(
    #     np.array(imgOneFeatures.normalizeMat),
    #     np.array(imgTwoFeatures.normalizeMat),
    # )

    # We go from the first image to the second
    # image, horizontally concatenating the
    # images so we can see predictions

    # Get all inlier points in the
    # first image
    ransacDataset : RANSACDataset[Tuple[Keypoint, Keypoint]] = matches.toRANSACDataset(True)
    inlierIndices : np.ndarray = hMat.findInlierIndices(ransacDataset)
    
    imageOneInlierKeypoints: np.ndarray = np.vstack([imgOneFeatures.keypoints[i].asHomogenous() for i in inlierIndices])
    
    # First, get all keypoints in the first image
    imageOneKeypoints: np.ndarray = np.array(
        [kp.asHomogenous() for kp in imgOneFeatures.keypoints]
    ).reshape(-1, 3)

    # Now, compute the predicted keypoints in the
    # second image
    imageTwoPredictedKeypoints: np.ndarray = (
        hMat.matrix @ imageOneInlierKeypoints.T
    ).T

    # Now, turn the predicted keypoints into
    # heterogeneous coordinates
    imageTwoPredicted_Heterogenous : np.ndarray = (
        imageTwoPredictedKeypoints[:, :2] / np.vstack((imageTwoPredictedKeypoints[:, -1], imageTwoPredictedKeypoints[:, -1])).T
    )

    # print(imageOneKeypoints)
    # print(imageTwoPredictedKeypoints)
    # print(imageTwoPredicted_Heterogenous)
    # print(hMat.matrix)
    # print(hMatUnNorm)

    # Now, draw the predicted keypoints onto the
    # second image
    imageTwoWithPredictions = cv2MatTwo.copy()
    for kp in imageTwoPredicted_Heterogenous:
        cv2.circle(
            imageTwoWithPredictions,
            tuple(kp.astype(int)),
            3,
            (0, 0, 255),
            -1,
        )
    
    # Now, draw onto the first image
    imageOneWithPredictions = cv2MatOne.copy()
    for kp in imageOneKeypoints:
        cv2.circle(
            imageOneWithPredictions,
            tuple(kp[:2].astype(int)),
            3,
            (0, 0, 255),
            -1,
        )
    
    # Concatenate the images horizontally
    concatenatedImage = np.hstack((imageOneWithPredictions, imageTwoWithPredictions))
    
    # Convert the concatenated image back to a Pillow image
    return arrayToPillowImage(concatenatedImage, colorMode)