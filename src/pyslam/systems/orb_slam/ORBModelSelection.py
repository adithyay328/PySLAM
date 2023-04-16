"""
Implements the model scoring and selection algorithm proposed by ORB-SLAMv1
in the section "Autonomous Map Initialization". Provides
quite a robust model selection approach. 
"""
from typing import Callable, List, Tuple, Union

from pyslam.epipolar_core import (
    FundamentalMatrix,
    HomographyMatrix,
)
from pyslam.image_processing.feature_descriptors.Keypoint import (
    Keypoint,
)

import numpy as np

"""
Some constants proposed by ORB-SLAM
for the model selection system.
"""

# Outlier rejection thresholds; T_H is for homography, T_F for fundamental matrix
T_H = 5.99
T_F = 3.84

# Minimum score for the better model to be considered accurate;
# if neither model is accurate, the system will skip
# this frame and try again with the next one
MIN_SCORE = 600

# This floating point indicates when we should chose the
# homogaphy over chosing the fundamental matrix
ORB_HEURISTIC_THRESHOLD = 0.45


def ORB_IsModelGood(modelScore: float) -> bool:
    """
    Given the score of a model, returns whether or not
    the model is good and should be used for the
    rest of the pipeline.

    :param modelScore: The score of the model

    :return: Whether or not the model is good
    """

    return modelScore > MIN_SCORE


def ORB_Model_Pick_Homography(
    homographyError: float, fundamentalError: float
) -> bool:
    """
    Given the error of the homography and fundamental matrix,
    returns the best model to use.

    :param homographyError: The error of the homography matrix
    :param fundamentalError: The error of the fundamental matrix

    :return: Whether or not we should pick the homography matrix
    """

    return (
        homographyError / (homographyError + fundamentalError)
        > ORB_HEURISTIC_THRESHOLD
    )


def ORB_Robust_Common_Scoring_Function(
    model: Union[HomographyMatrix, FundamentalMatrix],
    threshold: float,
) -> Callable[[List[Tuple[Keypoint, Keypoint]]], float]:
    """
    Implements the robust scoring function proposed
    by ORB-SLAMv1. This is used by both the homography
    and fundamental matrix scoring functions for
    the ORB model selection system.

    :param errors: A list of errors for all matches.

    :return: The scoring function
    """

    def returnFunc(
        kps: List[Tuple[Keypoint, Keypoint]]
    ) -> float:
        score = 0.0
        for kpOne, kpTwo in kps:
            symetricErrors = []
            if isinstance(model, HomographyMatrix):
                symetricErrors += list(
                    model.getSymetricTransferErrorPair(
                        model, kpOne, kpTwo
                    )
                )
            elif isinstance(model, FundamentalMatrix):
                symetricErrors += list(
                    model.getSymetricTransferErrorPair(
                        model, kpOne, kpTwo
                    )
                )

            # Ignore the error if larger than T_H
            if symetricErrors[0] < threshold:
                score += threshold - symetricErrors[0]
            if symetricErrors[1] < threshold:
                score += threshold - symetricErrors[1]

        return score

    return returnFunc


def ORB_Homograhy_Scoring_Function(
    homographyMat: HomographyMatrix,
) -> Callable[[List[Tuple[Keypoint, Keypoint]]], float]:
    """
    Returns a scoring function for the homography matrix.
    This uses the robust scoring function proposed
    by ORB-SLAMv1.

    :param homographyMat: The homography matrix to use

    :return: The scoring function
    """

    return ORB_Robust_Common_Scoring_Function(homographyMat, T_H)


def ORB_Fundamental_Scoring_Function(
    fundamentalMat: FundamentalMatrix,
) -> Callable[[List[Tuple[Keypoint, Keypoint]]], float]:
    """
    Returns a scoring function for the fundamental matrix.
    This uses the robust scoring function proposed
    by ORB-SLAMv1.

    :param fundamentalMat: The fundamental matrix to use

    :return: The scoring function
    """

    return ORB_Robust_Common_Scoring_Function(
        fundamentalMat, T_F
    )
