# This module defines a HomographyMatrix estimation routine.
# Internally, it only uses the SVD for homogenous least squares,
# using the least number of points we can get away with (4 points).
# Iterative refinement isn't used on the initial solution, as that
# is assumed to be the reponsibility of Bundle Adjustment/batch
# optimization

import numpy as np

from pyslam.odometry import PoseTransformSource


class HomographyMatrix(PoseTransformSource):
    def __init__(
        self,
        imgOneNormalizedFeatures: NormalizedImageFeatures,
        imgTwoNormalizedFeatures: NormalizedImageFeatures,
    ):
        """
        A class representing a HomographyMatrix that we
        estimate from 2 cameras, whose normalized
        image features are already computed.
        :param imgOneNormalizedFeatures: The image features
            we detected from the first image.
        :param imgTwoNormalizedFeatures: The image features
            we detected from the second image.

        :return: self, the Homography matrix we estimated
            that goes from image one to image two
        """
