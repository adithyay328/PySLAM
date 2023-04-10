"""
Implements core logic for
estimation related to the epipolar
geometry.

Implements estimation of fundamental
and homography matrices, triangulation
and relative pose estimation.
"""
from typing import List, Tuple

import numpy as np
import jax.numpy as jnp
import jax

from pyslam.camera_matrix import Camera
from pyslam.image_processing.feature_descriptors.Keypoint import (
    Keypoint,
)
from pyslam.optim.procedures import homogenousMatrixLeastSquares
from pyslam.optim.ransac import RANSACModel, RANSACDataset


def triangulatePoints(
    c1: Camera,
    c2: Camera,
    pointsOne: List[Keypoint],
    pointsTwo: List[Keypoint],
) -> List[np.ndarray]:
    """
    Implements logic related to triangulating points
    seen in two cameras.

    :param c1: The first camera involved.
    :param c2: The second camera involved.
    :param pointsOne: A list of Keypoints seen in camera one.
    :param pointsTwo: A list of Keypoints seen in camera two. These should
        be ordered such that the nth Keypoint in points one corresponds
        to the nth Keypoint in pointsTwo

    :return: Returns a list of 3D world points following triangulation; these
        are in heterogenous coordinates
    """
    results: List[np.ndarray] = []

    assert len(pointsOne) == len(pointsTwo)

    for idx in range(len(pointsOne)):
        # Build out matrix to perform
        # homogenous LLS on

        # For each camera, we need the same
        # set of constraints, which we can
        # then h-stack and get the solution
        # for

        # First, camera one:

        # Get row's and transpose them
        camOnep1T = c1.cameraMat[0, :].T
        camOnep2T = c1.cameraMat[1, :].T
        camOnep3T = c1.cameraMat[2, :].T

        # Now, get x and y for the point
        # in image one
        x1 = pointsOne[idx].asHeterogenous()[0]
        y1 = pointsOne[idx].asHeterogenous()[1]

        # Now, build out the constraint matrix
        camOneRowOne: np.ndarray = y1 * camOnep3T - camOnep2T
        camOneRowTwo: np.ndarray = camOnep1T - x1 * camOnep3T

        # Now, repeat all of the above, but
        # for camera two
        camTwop1T = c2.cameraMat[0, :].T
        camTwop2T = c2.cameraMat[1, :].T
        camTwop3T = c2.cameraMat[2, :].T

        x2 = pointsTwo[idx].asHeterogenous()[0]
        y2 = pointsTwo[idx].asHeterogenous()[1]

        camTwoRowOne: np.ndarray = y2 * camTwop3T - camTwop2T
        camTwoRowTwo: np.ndarray = camTwop1T - x2 * camTwop3T

        # Now, we can h-stack these two
        # matrices and get the solution
        # for the point
        constraintMatrix: np.ndarray = np.hstack(
            (
                camOneRowOne,
                camOneRowTwo,
                camTwoRowOne,
                camTwoRowTwo,
            )
        )
        point: np.ndarray = np.array(homogenousMatrixLeastSquares(
            jnp.array(constraintMatrix)
        ))

        # Now, we can normalize the point
        # and add it to the results
        results.append(point[0:3] / point[3])

    return results


class FundamentalMatrix(RANSACModel[Tuple[Keypoint, Keypoint]]):
    def __init__(self, threshold: float) -> None:
        """
        Implements a model for estimating a fundamental
        matrix from a set of points.

        :param threshold: The threshold to use for determining
            whether a point is an inlier or not.
        """
        super().__init__()
        self.matrix: np.ndarray = np.eye(3)
        self.threshold: float = threshold

    def fit(
        self, data: RANSACDataset[Tuple[Keypoint, Keypoint]]
    ) -> None:
        """
        Fits the fundamental matrix to the specified
        RANSACDataset. Always use normalized coordinates
        as the input to this; these can be extracted
        from ImageFeatues instances easily.

        :param data: The dataset to fit the fundamental
            matrix to.
        """
        # First, we need to get the points
        # from the dataset
        pointsOne: List[Keypoint] = []
        pointsTwo: List[Keypoint] = []

        for pointPair in data.data:
            pointsOne.append(pointPair[0])
            pointsTwo.append(pointPair[1])

        # Now, we can build out the constraint
        # matrix
        constraintMatrix: np.ndarray = np.zeros(
            (len(pointsOne), 9)
        )

        for idx in range(len(pointsOne)):
            # Get the point in image one
            x1 = pointsOne[idx].asHeterogenous()[0]
            y1 = pointsOne[idx].asHeterogenous()[1]

            # Get the point in image two
            x2 = pointsTwo[idx].asHeterogenous()[0]
            y2 = pointsTwo[idx].asHeterogenous()[1]

            # Build out the constraint matrix
            constraintMatrix[idx, :] = np.array(
                [
                    x1 * x2,
                    x1 * y2,
                    x1,
                    y1 * x2,
                    y1 * y2,
                    y1,
                    x2,
                    y2,
                    1,
                ]
            )

        # Now, we can get the solution
        # for the fundamental matrix
        solMat : np.ndarray = np.array(homogenousMatrixLeastSquares(
            jnp.array(constraintMatrix)
        ).reshape(3, 3))

        # Now, we can get the SVD of the
        # solution matrix to
        # enforce the rank 2 constraint
        u, s, v = np.linalg.svd(solMat)
        s[-1] = 0

        # Now, we can reassemble the matrix
        # and set it as the matrix
        self.matrix = u @ np.diag(s) @ v

    def symetricTransferError(
        self, pointOne: Keypoint, pointTwo: Keypoint
    ) -> float:
        """
        Computes the symetric transfer error between
        two points. This is actually the symetric
        epipolar error, but in literature such as ORB-SLAM
        this name is abused as the symetric transfer error.

        :param pointOne: The first point.
        :param pointTwo: The second point.

        :return: The symetric epipolar error for
            this pair of points.
        """
        # Make sure our matrix is
        # initialized
        assert not ((self.matrix == np.eye(3)).all())

        # First, we need to get the
        # point in image one
        x1 = pointOne.asHeterogenous()[0]
        y1 = pointOne.asHeterogenous()[1]

        # Now, get the point in image two
        x2 = pointTwo.asHeterogenous()[0]
        y2 = pointTwo.asHeterogenous()[1]

        x1_H = np.array([x1, y1, 1])
        x2_H = np.array([x2, y2, 1])

        # Get epipolar lines
        epipolarLineOne: np.ndarray = (
            self.matrix @ pointOne.asHomogenous()
        )
        epipolarLineTwo: np.ndarray = (
            self.matrix.T @ pointTwo.asHomogenous()
        )

        # Now, we can compute the transfer
        # error for each point, which is the
        # magnitude of the distance between
        # the point and the epipolar line
        transferError = (
            (np.dot(x2_H, self.matrix @ x1_H) ** 2)
            * ( 
            (1 / ( np.linalg.norm(self.matrix @ x1_H, ord=1) ** 2 + np.linalg.norm(self.matrix.T @ x1_H, ord=2) ** 2 )) 
            + ( 1 / ( np.linalg.norm(self.matrix.T @ x2_H, ord=1) ** 2 + np.linalg.norm(self.matrix.T @ x2_H, ord=2) ** 2 ))
            )
        )

        return transferError

        

    def findInlierIndices(
        self,
        data: RANSACDataset[Tuple[Keypoint, Keypoint]],
    ) -> np.ndarray:
        """
        Finds the indices of the inliers in the dataset.
        This is done by computing the symetric transfer error
        for points, and restrict to some threshold.

        :param data: The dataset to find inliers for.

        :return: Root indices that are determined to be inliers.
        """
        rootIndices: List[int] = []

        for i in range(len(data.indices)):
            error: float = self.symetricTransferError(
                data.data[i][0], data.data[i][1]
            )
            if error <= self.threshold:
                rootIndices.append(int(data.indices[i]))

        return np.array(rootIndices)

    def unnormalize(
        self,
        imageOneScalingMat: np.ndarray,
        imageTwoScalingMat: np.ndarray,
    ) -> np.ndarray:
        """
        Unnormalizes the fundamental matrix. This is done by
        right multiplying the fundamental matrix by the
        the scaling matrix for the first image, and then left
        multiplying by the transpose of the scaling
        matrix for the second image.

        :param imageOneScalingMat: The scaling matrix for the
            first image.
        :param imageTwoScalingMat: The scaling matrix for the
            second image.

        :return: The unnormalized fundamental matrix.
        """

        return (
            imageTwoScalingMat.T
            @ self.matrix
            @ imageOneScalingMat
        )


class HomographyMatrix(RANSACModel[Tuple[Keypoint, Keypoint]]):
    def __init__(self, threshold: float) -> None:
        """
        Implements a model for estimating a homography
        matrix from a set of points.

        :param threshold: The threshold to use for determining
            whether a point is an inlier or not.
        """
        super().__init__()
        self.matrix: np.ndarray = np.eye(3)
        self.threshold: float = threshold

    def fit(
        self, data: RANSACDataset[Tuple[Keypoint, Keypoint]]
    ) -> None:
        """
        Fits the homography matrix to the specified
        RANSACDataset. Always use normalized coordinates
        as the input to this; these can be extacted
        from ImageFeatues instances easily.

        :param data: The dataset to fit the homography
            matrix to.
        """
        # First, we need to get the points
        # from the dataset
        pointsOne: List[Keypoint] = []
        pointsTwo: List[Keypoint] = []

        for pointPair in data.data:
            pointsOne.append(pointPair[0])
            pointsTwo.append(pointPair[1])

        # Now, we can build out the constraint
        # matrix
        constraintMatrix: np.ndarray = np.zeros(
            (len(pointsOne) * 2, 9)
        )

        for idx in range(len(pointsOne)):
            # Get the point in image one
            x1 = pointsOne[idx].asHeterogenous()[0]
            y1 = pointsOne[idx].asHeterogenous()[1]

            # Get the point in image two
            x2 = pointsTwo[idx].asHeterogenous()[0]
            y2 = pointsTwo[idx].asHeterogenous()[1]

            # Build out the constraint matrix
            constraintMatrix[2 * idx, :] = np.array(
                [
                    -x1,
                    -y1,
                    -1,
                    0,
                    0,
                    0,
                    x1 * x2,
                    y1 * x2,
                    x2,
                ]
            )
            constraintMatrix[2 * idx + 1, :] = np.array(
                [
                    0,
                    0,
                    0,
                    -x1,
                    -y1,
                    -1,
                    x1 * y2,
                    y1 * y2,
                    y2,
                ]
            )

        # Now, we can get the solution
        # for the homography matrix
        solMat = homogenousMatrixLeastSquares(
            jnp.array(constraintMatrix)
        ).reshape(3, 3)

        self.matrix = np.array(solMat)

    def symetricTransferError(
        self, pointOne: Keypoint, pointTwo: Keypoint
    ) -> float:
        """
        Computes the symetric transfer error between
        two points. This is defined in Zisserman,
        but basically it's just the sum of
        differences between the point and the
        predicted point in the other image.

        :param pointOne: The first point.
        :param pointTwo: The second point.

        :return: The symetric transfer error.
        """
        # Make sure our matrix is
        # initialized
        # assert not ((self.matrix == np.eye(3)).all())

        # Predict point in each image first.
        predictedPoint_imgTwo: np.ndarray = (
            self.matrix @ pointOne.asHomogenous()
        )
        predictedPoint_imgOne: np.ndarray = (
            np.linalg.inv(self.matrix) @ pointTwo.asHomogenous()
        )

        # Convert to heterogenous
        predictedPoint_imgTwo = (
            predictedPoint_imgTwo[0:2] / np.hstack((predictedPoint_imgTwo[-1], predictedPoint_imgTwo[-1]))
        )
        predictedPoint_imgOne = (
            predictedPoint_imgOne[0:2] / np.hstack((predictedPoint_imgOne[-1], predictedPoint_imgOne[-1]))
        )

        # Now, we can compute the transfer
        # error for each point, which is the
        # magnitude of the distance between
        # the given point and the predicted point
        transferErrorOne: float = float(
            np.linalg.norm(
                pointOne.asHeterogenous() - predictedPoint_imgOne
            ) ** 2
        )

        transferErrorTwo: float = float(
            np.linalg.norm(
                pointTwo.asHeterogenous() - predictedPoint_imgTwo
            ) ** 2
        )

        return transferErrorOne + transferErrorTwo

    def findInlierIndices(
        self,
        data: RANSACDataset[Tuple[Keypoint, Keypoint]],
    ) -> np.ndarray:
        """
        Finds the indices of the inliers in the dataset.
        This is done by computing the symetric transfer error
        for points, and restrict to some threshold.

        :param data: The dataset to find inliers for.

        :return: Root indices that are determined to be inliers.
        """
        rootIndices: List[int] = []

        for i in range(len(data.indices)):
            error: float = self.symetricTransferError(
                data.data[i][0], data.data[i][1]
            )
            if error <= self.threshold:
                rootIndices.append(int(data.indices[i]))

        return np.array(rootIndices)

    def unnormalize(
        self,
        imageOneScalingMat: np.ndarray,
        imageTwoScalingMat: np.ndarray,
    ) -> np.ndarray:
        """
        Unnormalizes the homography matrix. This is done by
        right multiplying the homography matrix by the
        the scaling matrix for the first image, and then left
        multiplying by the transpose of the scaling
        matrix for the second image.

        :param imageOneScalingMat: The scaling matrix for the
            first image.
        :param imageTwoScalingMat: The scaling matrix for the
            second image.

        :return: The unnormalized homography matrix.
        """
        return (
            imageTwoScalingMat.T
            @ self.matrix
            @ imageOneScalingMat
        )
