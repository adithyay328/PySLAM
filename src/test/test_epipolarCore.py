"""

Containts unit tests for the epipolarCore module.
This is one of the more important
modules in the package, as it contains the core
fundamental matrix and homography matrix classes,
which underpinds all classical
SLAM algorithms.

"""
import random
random.seed(42)
import asyncio
import math

import transforms3d

from pyslam.epipolar_core import *
from pyslam.camera_matrix import Camera
from pyslam.image_processing.feature_descriptors.Keypoint import Keypoint
from pyslam.optim.ransac import (
    RANSACDataset,
    RANSACModel,
    RANSACEstimator,
)
from pyslam.systems.orb_slam import ORB_Homograhy_Scoring_Function, ORB_Fundamental_Scoring_Function, ORB_Model_Pick_Homography, ORB_IsModelGood

def generatePlanarPoints(numPoints, scale = 1):
    """

    Generates numPoints points on a the
    xy plane, but shifted to z=2. Returns
    homogenous world coordinates.
    """
    plane = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
    ])

    # Generate numPoints random points on the plane; shape (3, numPoints), where
    # each column is a point
    points = (plane @ np.random.rand(3, numPoints) * scale)

    # Add a positive z value to the points
    points[2, :] += 2

    # Make all points homogeneous
    pointsHomog = np.vstack((points, np.ones((1, numPoints))))

    return pointsHomog

def generateRandomCamera():
    """
    Generates a random camera matrix.
    """
    # Generate a random camera matrix
    randomPosition = np.random.rand(3) * 4
    randomRotation = transforms3d.euler.euler2mat(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

    randomCam = Camera(np.eye(3), randomPosition, randomRotation)

    return randomCam


def test_triangulation():
    """
    Tests the triangulation function.
    Functions by generating 2 cameras,
    generating world points, and then
    projecting them into the cameras.

    Then, it triangulates the points
    and checks that the triangulated
    points are close to the original
    world points.
    """
    NUM_OF_TESTS = 30
    for i in range(NUM_OF_TESTS):
        # The first camera is always at the origin
        identityCam = Camera(np.eye(3))
        otherCam = generateRandomCamera()

        NUM_POINTS = 30

        # Generate planar points
        pointsHomog = generatePlanarPoints(numPoints=NUM_POINTS)

        # Make a heterogenous version
        points = pointsHomog[:3, :]

        # Now, project onto each camera using the camera matrix
        imOnePointsHomog = (identityCam.cameraMat @ pointsHomog).T
        imTwoPointsHomog = (otherCam.cameraMat @ pointsHomog).T

        assert imOnePointsHomog.shape == (30, 3)
        assert imTwoPointsHomog.shape == (30, 3)

        # Now, convert points to keypoints
        imOneKeypoints = [Keypoint(point.squeeze()) for point in np.vsplit(imOnePointsHomog, NUM_POINTS)]
        imTwoKeypoints = [Keypoint(point.squeeze()) for point in np.vsplit(imTwoPointsHomog, NUM_POINTS)]

        # Now, triangulate points
        triangulatedPoints = triangulatePoints(identityCam, otherCam, imOneKeypoints, imTwoKeypoints)

        # Compute average l1 distance between predicte and actual points
        averageL1 = np.mean(  np.abs((np.vstack(triangulatedPoints) - points.T).flatten())  )

        # Max threshold for the points to be correct
        MAX_THRESHOLD = 0.00001
        assert averageL1 < MAX_THRESHOLD
    
def test_homography_estimation():
    """

    Tests homography estimation with ransac.
    We do this by generating random planar
    points, and then projecting them onto
    a random camera. We then estimate the
    homography from the camera to the
    plane, and then check that the
    homography returns all indices
    as inliers.

    """
    # The first camera is always at the origin
    identityCam = Camera(np.eye(3))
    otherCam = generateRandomCamera()

    NUM_POINTS = 30

    # Generate planar points
    pointsHomog = generatePlanarPoints(numPoints=NUM_POINTS)

    # Make a heterogenous version
    points = pointsHomog[:3, :]

    # Now, project onto each camera using the camera matrix
    imOnePointsHomog = (identityCam.cameraMat @ pointsHomog).T
    imTwoPointsHomog = (otherCam.cameraMat @ pointsHomog).T

    assert imOnePointsHomog.shape == (30, 3)
    assert imTwoPointsHomog.shape == (30, 3)

    # Now, convert points to keypoints
    imOneKeypoints = [Keypoint(point.squeeze()) for point in np.vsplit(imOnePointsHomog, NUM_POINTS)]
    imTwoKeypoints = [Keypoint(point.squeeze()) for point in np.vsplit(imTwoPointsHomog, NUM_POINTS)]

    # Build a RANSAC estimator, and using just 10 iterations,
    # get the homography
    homogMatConstructor = lambda: HomographyMatrix(
        10,
        HomographyMatrix.symetricTransferError,
        ORB_Homograhy_Scoring_Function,
    )

    homogEstimator = RANSACEstimator(
        8)

    dataset = RANSACDataset(
        [ (imOneKeypoints[i], imTwoKeypoints[i]) for i in range(NUM_POINTS) ],
        indices=None
    )

    homogMat, indices, homogScore = asyncio.run(
        homogEstimator.fit(
            dataset, homogMatConstructor, 30, 8, False
        )
    )

    assert len(indices) == NUM_POINTS

def test_fundamental_matrix_estimation():
    """

    Tests fundamental matrix estimation with ransac.
    We do this by generating random planar
    points, adding noise to make the points
    non-planar, and then projecting them onto
    a random camera. We then estimate the
    fundamental matrix from the camera to the
    plane, and then check that the
    fundamental matrix returns all indices
    as inliers.

    """
    # The first camera is always at the origin
    identityCam = Camera(np.eye(3))
    otherCam = generateRandomCamera()

    NUM_POINTS = 30

    # Generate planar points
    pointsHomog = generatePlanarPoints(numPoints=NUM_POINTS)

    # Make a heterogenous version
    points = pointsHomog[:3, :]

    # Now, project onto each camera using the camera matrix
    imOnePointsHomog = (identityCam.cameraMat @ pointsHomog).T
    imTwoPointsHomog = (otherCam.cameraMat @ pointsHomog).T

    assert imOnePointsHomog.shape == (30, 3)
    assert imTwoPointsHomog.shape == (30, 3)

    # Now, convert points to keypoints
    imOneKeypoints = [Keypoint(point.squeeze()) for point in np.vsplit(imOnePointsHomog, NUM_POINTS)]
    imTwoKeypoints = [Keypoint(point.squeeze()) for point in np.vsplit(imTwoPointsHomog, NUM_POINTS)]

    # Build a RANSAC estimator, and using just 10 iterations,
    # get the homography
    fundMatConstructor = lambda: FundamentalMatrix(
        10,
        FundamentalMatrix.symetricTransferError,
        ORB_Fundamental_Scoring_Function,
    )

    fundEstimator = RANSACEstimator(
        8)

    dataset = RANSACDataset(
        [ (imOneKeypoints[i], imTwoKeypoints[i]) for i in range(NUM_POINTS) ],
        indices=None
    )

    fundMat, indices, fundScore = asyncio.run(
        fundEstimator.fit(
            dataset, fundMatConstructor, 30, 8, False
        )
    )


    assert len(indices) == NUM_POINTS

def test_fundamental_matrix_pose_estimation():
    """

    Tests fundamental matrix pose estimation with ransac.
    We do this by generating random planar
    points, and then projecting them onto
    a random camera. We then estimate the
    fundamental matrix from the camera to the
    plane, and then check that the
    fundamental matrix returns all indices
    as inliers.

    """
    # The first camera is always at the origin
    identityCam = Camera(np.eye(3))
    otherCam = generateRandomCamera()

    NUM_POINTS = 100
    NUM_TRIALS = 5

    for i in range(NUM_TRIALS):
        # Generate planar points
        pointsHomog = generatePlanarPoints(numPoints=NUM_POINTS, scale=20)

        # Add random values to the z-axis, since we want to
        # make sure that the points are not coplanar,
        # since fundamental matrices don't fit very
        # well in that case
        pointsHomog[2, :] += 30 * np.random.rand(NUM_POINTS)

        # Make a heterogenous version
        points = pointsHomog[:3, :]

        # Now, project onto each camera using the camera matrix
        imOnePointsHomog = (identityCam.cameraMat @ pointsHomog).T
        imTwoPointsHomog = (otherCam.cameraMat @ pointsHomog).T

        assert imOnePointsHomog.shape == (NUM_POINTS, 3)
        assert imTwoPointsHomog.shape == (NUM_POINTS, 3)

        # Now, convert points to keypoints
        imOneKeypoints = [Keypoint(point.squeeze()) for point in np.vsplit(imOnePointsHomog, NUM_POINTS)]
        imTwoKeypoints = [Keypoint(point.squeeze()) for point in np.vsplit(imTwoPointsHomog, NUM_POINTS)]

        # Build a RANSAC estimator, and using just 10 iterations,
        # get the fundamental matrix
        fundMatConstructor = lambda: FundamentalMatrix(
            10,
            FundamentalMatrix.symetricTransferError,
            ORB_Fundamental_Scoring_Function,
        )

        fundEstimator = RANSACEstimator(
            8)

        dataset = RANSACDataset(
            [ (imOneKeypoints[i], imTwoKeypoints[i]) for i in range(NUM_POINTS) ],
            indices=None
        )

        fundMat, indices, fundScore = asyncio.run(
            fundEstimator.fit(
                dataset, fundMatConstructor, NUM_POINTS, 8, True
            )
        )

        # Now, estimate the pose
        assert type(fundMat) == FundamentalMatrix
        poseHypotheses = fundMat.getFourMotionHypotheses(np.eye(3))
        finalSol = fundMat.chooseSolution(np.eye(3), poseHypotheses, imOneKeypoints, imTwoKeypoints, indices, False)
        finalSolRotation = finalSol[0].cameraMat[:, :3]
        finalSolTranslation = finalSol[0].cameraMat[:, -1]

        # To compute angle between translation vectors,
        # take dot product, divide by product of
        # norms, and then take the inverse cosine
        # to get the angle in radians        
        translationVectorErrorRads = np.arccos(
            np.dot( finalSolTranslation, otherCam.cameraMat[:, -1] ) /
            (np.linalg.norm(finalSolTranslation) * np.linalg.norm(otherCam.cameraMat[:, -1]))
        )
        # Now, compute the angle between the two rotation
        # matrices. This is done by multiplying
        # one by the transpose of the other, and then
        # computng the magnitude of the rotation given
        # by that matrix

        rotationMatrixErrorRads = np.arccos(
            (np.trace(finalSolRotation @ otherCam.cameraMat[:, :3].T) - 1) / 2
        )

        # Thresholds for each error
        translationThreshold = 0.1
        rotationThreshold = 0.1
        
        assert translationVectorErrorRads < translationThreshold and rotationMatrixErrorRads < rotationThreshold