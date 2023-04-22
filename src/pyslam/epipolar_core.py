"""
Implements core logic for
estimation related to the epipolar
geometry.

Implements estimation of fundamental
and homography matrices, triangulation
and relative pose estimation.

There are some assumptions we make
about how epipolar models are scored.
The main thing is that we are going
to compute a score per match hypothesis,
and inliers are simply points whose score
is below a certain threshold.
"""
from typing import List, Tuple, Callable, Optional
import logging

import numpy as np
import jax.numpy as jnp
import jax

from pyslam.camera_matrix import Camera
from pyslam.image_processing.feature_descriptors.Keypoint import (
    Keypoint,
)
from pyslam.optim.procedures import homogenousMatrixLeastSquares
from pyslam.optim.ransac import RANSACModel, RANSACDataset


@jax.jit
def _jaxTriangulatePoints(
    c1: jax.Array,
    c2: jax.Array,
    pointsOne: jax.Array,
    pointsTwo: jax.Array,
) -> jax.Array:
    """
    An inner function that uses
    JAX for fast trangulation.
    Use the triangulatePoints
    function as a nice interface to this

    :param c1: The first camera involved.
    :param c2: The second camera involved.
    :param pointsOne: A list of Keypoints seen in camera one.
    :param pointsTwo: A list of Keypoints seen in camera two. These should
        be ordered such that the nth Keypoint in points one corresponds
        to the nth Keypoint in pointsTwo

    :return: Returns a jax array of size(n, 1, 4), where n is the number
        of world coordinates. The 4 is because this procedure yields
        homogenous coordinates.
    """
    # First, extract both camera rows
    camOnep1T = c1[0, :].T
    camOnep2T = c1[1, :].T
    camOnep3T = c1[2, :].T

    camTwop1T = c2[0, :].T
    camTwop2T = c2[1, :].T
    camTwop3T = c2[2, :].T

    # Now, build all the matrices
    # we're going to run homogenous
    # least squares on. Each matrix
    # provides constraints for triangulation
    # of a single point
    triangulationMats: List[jax.Array] = []

    for i in range(len(pointsOne)):
        # Extract all point x's and y's
        x1 = pointsOne[i][0]
        y1 = pointsOne[i][1]

        x2 = pointsTwo[i][0]
        y2 = pointsTwo[i][1]

        # Build out the constraint rows
        # for each camera
        camOneRowOne = y1 * camOnep3T - camOnep2T
        camOneRowTwo = camOnep1T - x1 * camOnep3T
        camTwoRowOne = y2 * camTwop3T - camTwop2T
        camTwoRowTwo = camTwop1T - x2 * camTwop3T

        # Now, we can h-stack these four
        # rows to get the constraint matrix
        # for this point
        triangulationMats.append(
            jnp.vstack(
                (
                    camOneRowOne,
                    camOneRowTwo,
                    camTwoRowOne,
                    camTwoRowTwo,
                )
            )
        )

    # Now, we can stack all of them into
    # a single matrix and run homogenous
    # least squares on it
    return homogenousMatrixLeastSquares(
        jnp.array(triangulationMats)
    )


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

    assert len(pointsOne) == len(pointsTwo)

    # We're going to use JAX to speed up
    # the triangulation process. To use
    # that logic, convert all
    # inputs to pure jax arrays
    c1Jax = jnp.array(c1.cameraMat)
    c2Jax = jnp.array(c2.cameraMat)
    pointsOne_Jax_Heterogenous = jnp.array(
        [p.asHeterogenous() for p in pointsOne]
    )
    pointsTwo_Jax_Heterogenous = jnp.array(
        [p.asHeterogenous() for p in pointsTwo]
    )

    # Now, run the JAX triangulation
    # function
    triangulatedPoints = _jaxTriangulatePoints(
        c1Jax,
        c2Jax,
        pointsOne_Jax_Heterogenous,
        pointsTwo_Jax_Heterogenous,
    )

    # First, reshape into (nx4)
    triangulatedPointsReshaped = np.array(
        triangulatedPoints.reshape(-1, 4)
    )
    # Now, divide by last dimension and convert to heterogenous
    triangulatedPointsHet = triangulatedPointsReshaped[:, :3] / (
        np.hstack(
            [triangulatedPointsReshaped[:, -1].reshape(-1, 1)]
            * 3
        )
    )

    # Now, convert the results back
    # to numpy arrays, and split to make it a list
    return np.vsplit(triangulatedPointsHet, len(pointsOne))


# TODO: Clean up the interface for
# epipolar models; really messy
# right now


class FundamentalMatrix(RANSACModel[Tuple[Keypoint, Keypoint]]):
    def __init__(
        self,
        threshold: float,
        errorFunction: Callable[
            ["FundamentalMatrix"],
            Callable[[Keypoint, Keypoint], float],
        ],
        scoreFunction: Callable[
            ["FundamentalMatrix"],
            Callable[[List[Tuple[Keypoint, Keypoint]]], float],
        ],
    ) -> None:
        """
        Implements a model for estimating a fundamental
        matrix from a set of points.

        :param threshold: The threshold to use for determining
            whether a point is an inlier or not.
        :param errorFunction: The function to use for computing
            the error between a single point and the model.
        :param scoreFunction: A function that takes in
            a list of errors for different match hypotheses,
            and outputs a single score for the model.
        """
        super().__init__()
        self.matrix: np.ndarray = np.eye(3)
        self.threshold: float = threshold
        self.errorFunction: Callable[
            ["FundamentalMatrix"],
            Callable[[Keypoint, Keypoint], float],
        ] = errorFunction
        self.scoreFunction: Callable[
            ["FundamentalMatrix"],
            Callable[[List[Tuple[Keypoint, Keypoint]]], float],
        ] = scoreFunction

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

        # Add one outer dimension, since the homog. LLS
        # operates on batches
        constraintMatrix = np.expand_dims(
            constraintMatrix, axis=0
        )

        # Now, we can get the solution
        # for the fundamental matrix
        solMat: np.ndarray = np.array(
            homogenousMatrixLeastSquares(
                jnp.array(constraintMatrix)
            ).reshape(3, 3)
        )

        # Now, we can get the SVD of the
        # solution matrix to
        # enforce the rank 2 constraint
        u, s, v = np.linalg.svd(solMat)
        s[-1] = 0

        # Now, we can reassemble the matrix
        # and set it as the matrix
        self.matrix = u @ np.diag(s) @ v

    @staticmethod
    def getSymetricTransferErrorPair(
        fMat: "FundamentalMatrix",
        pointOne: Keypoint,
        pointTwo: Keypoint,
    ) -> Tuple[float, float]:
        """
        Computes the symetric transfer error
        between two points. This is actually the symetric
        epipolar error, but in literature such as ORB-SLAM
        this name is abused as the symetric transfer error.

        :param fMat: The fundamental matrix to use.
        :param pointOne: The first point.
        :param pointTwo: The second point.

        :return: The symetric epipolar error for
            this pair of points, as a tuple of
            (errorOne, errorTwo). The first error
            is the error for the first point, and
            the second error is the error for the
            second point.
        """
        # First, we need to get the
        # point in image one
        x1 = pointOne.asHeterogenous()[0]
        y1 = pointOne.asHeterogenous()[1]

        # Now, get the point in image two
        x2 = pointTwo.asHeterogenous()[0]
        y2 = pointTwo.asHeterogenous()[1]

        x1_H = np.array([x1, y1, 1])
        x2_H = np.array([x2, y2, 1])

        # Now, we can compute the transfer
        # error for each point, which is the
        # magnitude of the distance between
        # the point and the epipolar line
        transferErrorOne = (
            np.dot(x2_H, fMat.matrix @ x1_H) ** 2
        ) * (
            (
                1
                / (
                    np.linalg.norm(fMat.matrix @ x1_H, ord=1)
                    ** 2
                    + np.linalg.norm(fMat.matrix.T @ x1_H, ord=2)
                    ** 2
                )
            )
        )
        transferErrorTwo = (
            np.dot(x1_H, fMat.matrix.T @ x2_H) ** 2
        ) * (
            (
                1
                / (
                    np.linalg.norm(fMat.matrix.T @ x2_H, ord=1)
                    ** 2
                    + np.linalg.norm(fMat.matrix.T @ x2_H, ord=2)
                    ** 2
                )
            )
        )

        return float(transferErrorOne), float(transferErrorTwo)

    @staticmethod
    def symetricTransferError(
        fMat: "FundamentalMatrix",
    ) -> Callable[[Keypoint, Keypoint], float]:
        """
        Returns a callable that computes
        the symetric transfer error between
        two points. This is actually the symetric
        epipolar error, but in literature such as ORB-SLAM
        this name is abused as the symetric transfer error.
        We return a Callable since the model itself
        expects a Callable as an error function, so
        this is really a builder for that function.

        :param fMat: The fundamental matrix to use.
        :param pointOne: The first point.
        :param pointTwo: The second point.

        :return: The symetric epipolar error for
            this pair of points.
        """
        # Make sure our matrix is
        # initialized
        assert not ((fMat.matrix == np.eye(3)).all())

        def returnFunc(
            pointOne: Keypoint, pointTwo: Keypoint
        ) -> float:
            return sum(
                fMat.getSymetricTransferErrorPair(
                    fMat, pointOne, pointTwo
                )
            )

        return returnFunc

    def findInlierIndices(
        self,
        data: RANSACDataset[Tuple[Keypoint, Keypoint]],
    ) -> np.ndarray:
        """
        Finds the indices of the inliers in the dataset.
        This is done by computing an error function
        for each point in the dataset, and then
        returning the indices of the points that
        have an error below the threshold.

        The threshold and error function
        are configured in the constructor.

        :param data: The dataset to find inliers for.

        :return: Root indices that are determined to be inliers.
        """
        rootIndices: List[int] = []

        errorFunction = self.errorFunction(self)

        for i in range(len(data.indices)):
            error: float = errorFunction(
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

    def getScore(
        self, data: RANSACDataset[Tuple[Keypoint, Keypoint]]
    ) -> float:
        """
        Returns the score of the model. How this
        is computed is configured in the
        constructor.

        :return: The score of the model.
        """
        scoreFunction = self.scoreFunction(self)

        # Now, we can compute the score
        # of the model
        return scoreFunction(data.data)

    def getFourMotionHypotheses(
        self, intrinsics: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Every fundamental matrix, upon conversion
        to an essential matrix, has four possible
        motion hypotheses. This function returns
        those four motion hypotheses. The other function,
        chose solution, is used to choose which one
        is correct.

        :param intrinsics: The intrinsics of the camera.
            Needed to convert the fundamental matrix
            to an essential matrix.

        :return: The four possible motion hypotheses,
            as a list of tuples of (rotation, translation).
            These have shape(3, 3) and shape(3, 1) respectively.
        """
        rawEMat = intrinsics.T @ self.matrix @ intrinsics

        # One thing we need to do is to
        # ensure the E matrix is rank 2;
        # low rank approximation with the SVD
        u, s, v = np.linalg.svd(rawEMat)
        eMatCorrected = u @ np.diag([s[0], s[1], 0]) @ v

        u, d, vt = np.linalg.svd(eMatCorrected)

        # Now, we can compute the four
        # possible motion hypotheses
        w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        # There are 2 rotations posisble,
        # so first build those.
        r1 = (u @ w @ vt).T
        r2 = (u @ w.T @ vt).T

        # Also, 2 translations are possible,
        # so build those as well
        C1 = u[:, 2].T
        C2 = -1 * C1.T

        # Now, we can build the four
        # possible motion hypotheses,
        # correcting for the sign of the
        # determinant if needed
        solOne = (
            (C1, r1) if np.linalg.det(r1) > 0 else (-C1, -r1)
        )
        solTwo = (
            (C2, r1) if np.linalg.det(r1) > 0 else (-C2, -r1)
        )
        solThree = (
            (C1, r2) if np.linalg.det(r2) > 0 else (-C1, -r2)
        )
        solFour = (
            (C2, r2) if np.linalg.det(r2) > 0 else (-C2, -r2)
        )

        return [solOne, solTwo, solThree, solFour]

    # TODO: Really clean up the way this API is designed.
    # Ugly as hell right now
    def chooseSolution(
        self,
        intrinsics: np.ndarray,
        motionHypotheses: List[Tuple[np.ndarray, np.ndarray]],
        pointsOne: List[Keypoint],
        pointsTwo: List[Keypoint],
        inlierIndices: np.ndarray,
        triangulateAll : bool = False
    ) -> Tuple[Camera, int, np.ndarray, bool]:
        """
        Chooses the correct motion hypothesis from
        the four possible ones. This is done by
        computing by triangulating the points
        seen in the cameras, and counting
        the number of points that are in front
        the camera. The hypothesis with the most
        points in front of the camera is chosen,
        alongisde a boolean flag indicating
        whether or not the solution is likely.
        This is done by checking if the number
        of points 

        :param intrinsics: The intrinsics of the camera.
        :param motionHypotheses: The four possible
            motion hypotheses.
        :param pointsOne: The points seen in the first
            camera.
        :param pointsTwo: The points seen in the second
            camera.

        :return: A tuple of (camera matrix, num points visible, triangulated points,
            and whether or not the solution is likely to be good)
        """
        # First, we need to convert the
        # motion hypotheses into camera
        # matrices
        cameraTupleToMatrix = (
            lambda C, r: intrinsics
            @ r
            @ np.hstack((np.eye(3), -1 * C.reshape((3, 1))))
        )
        cameraMatrices = [
            cameraTupleToMatrix(*hypothesis)
            for hypothesis in motionHypotheses
        ]
        cameraObjs = [
            Camera(intrinsics) for cameraMatrix in cameraMatrices
        ]
        for idx, cameraMat in enumerate(cameraMatrices):
            cameraObjs[idx].extrinsics = motionHypotheses[idx][
                1
            ] @ np.hstack(
                (
                    np.eye(3),
                    -1
                    * motionHypotheses[idx][0].reshape((3, 1)),
                )
            )
            cameraObjs[idx].cameraMat = (
                cameraObjs[idx].intrinsics
                @ cameraObjs[idx].extrinsics
            )

        # Store all counts of point inliers in here;
        # allows us to look at the distribution, which
        # is more useful than just a point estimate,
        # since then we can do tests to ensure one prominent
        # solution is available
        inlierCounts : List[int] = []

        # Stores the num of triangulate points for
        # the best model
        triangulatedPoints : np.ndarray = np.array([])

        # To improve performance, only triangulate
        # 5 or so points, picking from the inliers
        # of the model
        NUM_POINTS_FOR_CHERIALITY = 7
        inlierIndicesToEsimateOn = np.random.choice(
            inlierIndices, min(NUM_POINTS_FOR_CHERIALITY, len(inlierIndices))
        )
        pointsOneToEstimateOn = [
            pointsOne[idx] for idx in inlierIndicesToEsimateOn
        ]
        pointsTwoToEstimateOn = [
            pointsTwo[idx] for idx in inlierIndicesToEsimateOn
        ]

        # Defines an identity camera that's basically
        # a camera at the origin; the other camera
        # is relative to this one
        identityCam = Camera(intrinsics)

        for idx, cameraObj in enumerate(cameraObjs):
            triPoints = triangulatePoints(
                identityCam, cameraObj, pointsOneToEstimateOn, pointsTwoToEstimateOn
            )

            # Convert to a numpy array, since it'll be easier to work with here
            triPointsNP = np.vstack(triPoints)

            # For the second camera, subtract the camera center, and dot with
            # the last row of the rotation mat(viewing angle of the cam);
            # if greater than 0, it's good
            secondCamDots = (
                triPointsNP
                - motionHypotheses[idx][0].reshape(1, -1)
            ) @ motionHypotheses[idx][1][-1].reshape(3, -1).squeeze()

            # We want to find how many points are IN FRONT
            # of BOTH cameras
            firstCamVisible = np.array(triPointsNP[:, -1] > 0, dtype=int)

            secondCamVisible = np.array(
                secondCamDots > 0, dtype=int
            )

            # Whether they're visible in both is just elementwise
            # product of the 2 above
            visibleInBothCams = (
                firstCamVisible * secondCamVisible
            )

            numVisible = int(
                np.sum(
                    np.array(visibleInBothCams == 1, dtype=int)
                )
            )

            inlierCounts.append(numVisible)
            
            # If this is the best model so far,
            # save its triangulated points
            if len(inlierCounts) < 2 or inlierCounts[-1] > max( inlierCounts[:len(inlierCounts) - 1] ):
                triangulatedPoints = triPointsNP[np.argwhere(visibleInBothCams)]
        
        # Test if the best model is distinguishable
        # enough from other models; if yes, return
        # that this is a good solution
        isGoodModel = True

        # There are a couple conditions that determine if
        # a model is "good"
        # 1. The best model needs to have more than
        # NUM_POINTS_FOR_CHERIALITY * 0.75 inliers
        # 2. The best model needs to beat the all other models
        # by a factor 2 at the minimum
        # If these are both satisfied, it's a good model
        sortedInlierCounts = np.array( sorted(inlierCounts) )
        isGoodModel = isGoodModel and sortedInlierCounts[-1] >= 2 * sortedInlierCounts[-2]
        isGoodModel = isGoodModel and sortedInlierCounts[-1] >= NUM_POINTS_FOR_CHERIALITY * .75

        # Get the best model idx
        bestModelIdx = np.argmax(inlierCounts)
        
        # If we want to triangulate on all
        # inliers, go ahead and do that
        if triangulateAll:
            pointsOneToTriangulate = [
                pointsOne[idx] for idx in inlierIndices
            ]
            pointsTwoToTriangulate = [
                pointsTwo[idx] for idx in inlierIndices
            ]
            triangulatedPoints = np.vstack(triangulatePoints(
                identityCam, cameraObjs[bestModelIdx], pointsOneToTriangulate, pointsTwoToTriangulate
            ))

        # Now, just return the best cam, with the number
        # of visible points
        return cameraObjs[bestModelIdx], max(inlierCounts), triangulatedPoints.reshape(-1, 3), isGoodModel


class HomographyMatrix(RANSACModel[Tuple[Keypoint, Keypoint]]):
    def __init__(
        self,
        threshold: float,
        errorFunction: Callable[
            ["HomographyMatrix"],
            Callable[[Keypoint, Keypoint], float],
        ],
        scoreFunction: Callable[
            ["HomographyMatrix"],
            Callable[[List[Tuple[Keypoint, Keypoint]]], float],
        ],
    ) -> None:
        """
        Implements a model for estimating a homography
        matrix from a set of points.

        :param threshold: The threshold to use for determining
            whether a point is an inlier or not.
        :param errorFunction: The error function to use for
            computing the error of a point.
        :param scoreFunction: The score function to use for
            computing the score of the model.
        """
        super().__init__()
        self.matrix: np.ndarray = np.eye(3)
        self.threshold: float = threshold
        self.errorFunction: Callable[
            ["HomographyMatrix"],
            Callable[[Keypoint, Keypoint], float],
        ] = errorFunction
        self.scoreFunction: Callable[
            ["HomographyMatrix"],
            Callable[[List[Tuple[Keypoint, Keypoint]]], float],
        ] = scoreFunction

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

        # Add one more dimension to constraint
        # matrix to allow it to work in
        # a batch
        constraintMatrix = np.expand_dims(
            constraintMatrix, axis=0
        )

        # Now, we can get the solution
        # for the homography matrix
        solMat = homogenousMatrixLeastSquares(
            jnp.array(constraintMatrix)
        ).reshape(3, 3)

        self.matrix = np.array(solMat)

    @staticmethod
    def getSymetricTransferErrorPair(
        hMat: "HomographyMatrix",
        pointOne: Keypoint,
        pointTwo: Keypoint,
    ) -> Tuple[float, float]:
        """
        :param hMat: The homography matrix to build
            the error function for.
        :param pointOne: The first point.
        :param pointTwo: The second point.

        :return: A tuple of the transfer error for
            each point. This is the the distance
            between point one and the predicted point in
            image two, and the distance between point two
            and the predicted point in image one.
        """
        # Predict point in each image first.
        predictedPoint_imgTwo: np.ndarray = (
            hMat.matrix @ pointOne.asHomogenous()
        )
        predictedPoint_imgOne: np.ndarray = (
            np.linalg.inv(hMat.matrix) @ pointTwo.asHomogenous()
        )

        # Convert to heterogenous coordinates
        predictedPoint_imgTwo = (
            predictedPoint_imgTwo / predictedPoint_imgTwo[2]
        )
        predictedPoint_imgOne = (
            predictedPoint_imgOne / predictedPoint_imgOne[2]
        )

        # Compute the transfer error
        transferError_imgOne = np.linalg.norm(
            pointOne.asHeterogenous() - predictedPoint_imgOne[:2]
        )
        transferError_imgTwo = np.linalg.norm(
            pointTwo.asHeterogenous() - predictedPoint_imgTwo[:2]
        )

        return float(transferError_imgOne), float(
            transferError_imgTwo
        )

    @staticmethod
    def symetricTransferError(
        hMat: "HomographyMatrix",
    ) -> Callable[[Keypoint, Keypoint], float]:
        """
        :param hMat: The homography matrix to build
            the error function for.

        :return: A calable that computes the transfer
            error for a point pair. This is the the distance
            between point one and the predicted point in
            image two, and the distance between point two
            and the predicted point in image one.
        """

        def errorFunction(
            pointOne: Keypoint, pointTwo: Keypoint
        ) -> float:
            return sum(
                hMat.getSymetricTransferErrorPair(
                    hMat, pointOne, pointTwo
                )
            )

        return errorFunction

    def findInlierIndices(
        self,
        data: RANSACDataset[Tuple[Keypoint, Keypoint]],
    ) -> np.ndarray:
        """
        Finds the indices of the inliers in the dataset.
        This is done using the threshold and error function
        specified in the constructor.

        :param data: The dataset to find inliers for.

        :return: Root indices that are determined to be inliers.
        """
        rootIndices: List[int] = []

        errorFunction = self.errorFunction(self)

        for i in range(len(data.indices)):
            error: float = errorFunction(
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

    def getScore(
        self,
        data: RANSACDataset[Tuple[Keypoint, Keypoint]],
    ) -> float:
        """
        Computes the score of the model. This is done
        by computing the error of each point in the dataset
        and then using the score function specified in the
        constructor.

        :param data: The dataset to compute the score for.

        :return: The score of the model.
        """

        return self.scoreFunction(self)(data.data)
