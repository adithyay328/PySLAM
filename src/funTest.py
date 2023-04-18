import asyncio
from typing import List, Tuple

import PIL.Image as Image
import cv2
import numpy as np
import jax.numpy as jnp
import open3d as o3d

from pyslam.epipolar_core import (
    FundamentalMatrix,
    HomographyMatrix,
)
from pyslam.image_processing.feature_descriptors.ImageFeatures import (
    ImageFeatures,
)
from pyslam.image_processing.feature_descriptors.descriptors.ORB import (
    ORB,
)
from pyslam.image_processing.feature_descriptors.Keypoint import (
    Keypoint,
)
from pyslam.image_processing.feature_descriptors.extractors.end2end.orb_detect_and_compute import (
    ORB_Detect_And_Compute,
)
from pyslam.image_processing.feature_descriptors.ImagePairFeatureMatches import (
    ImagePairMatches,
)
from pyslam.optim.ransac import (
    RANSACDataset,
    RANSACModel,
    RANSACEstimator,
)
from pyslam.visualize.DrawEpipolar import (
    drawEpipolarLines,
    drawHomographyHypotheses,
)
from pyslam.image_processing.cv_pillow import (
    pillowToArray,
    arrayToPillowImage,
    PillowColorFormat,
)
from pyslam.visualize.DrawFeatures import drawFeatures
from pyslam.visualize.DrawMatches import drawStereoMatches

from pyslam.systems.orb_slam import (
    ORB_Homograhy_Scoring_Function,
    ORB_Fundamental_Scoring_Function,
    ORB_Model_Pick_Homography,
    ORB_IsModelGood,
)

CALIB = np.array(
    [
        [1297, 0, 1054],
        [0, 1274, 541.33],
        [0.0, 0.0, 1.0],
    ]
)

IMAGE_ONE_FNAME = "src/Noam1.jpg"
IMAGE_TWO_FNAME = "src/Noam2.jpg"


async def run() -> None:
    # First, get features on both
    imOne = Image.open(IMAGE_ONE_FNAME).reduce(2)#.rotate(270)
    imTwo = Image.open(IMAGE_TWO_FNAME).reduce(2)#.rotate(270)
    # imOne = Image.open(IMAGE_ONE_FNAME).resize((400, 300))
    # imTwo = Image.open(IMAGE_TWO_FNAME).resize((400, 300))
    imOneExtractor = ORB_Detect_And_Compute(600)
    imTwoExtractor = ORB_Detect_And_Compute(600)
    imOneFeatures: ImageFeatures[ORB] = ImageFeatures[ORB](
        imOne, imOneExtractor, imOneExtractor
    )
    imTwoFeatures: ImageFeatures[ORB] = ImageFeatures[ORB](
        imTwo, imTwoExtractor, imTwoExtractor
    )

    # Drawing
    imgOneWithFeatures = drawFeatures(imOne, imOneFeatures)
    imgTwoWithFeatures = drawFeatures(imTwo, imTwoFeatures)

    cv2.imshow(
        "Img One Features",
        cv2.cvtColor(
            pillowToArray(imgOneWithFeatures), cv2.COLOR_RGB2BGR
        ),
    )
    cv2.imshow(
        "Img Two Features",
        cv2.cvtColor(
            pillowToArray(imgTwoWithFeatures), cv2.COLOR_RGB2BGR
        ),
    )

    # Build normalised image coordinates
    imOneFeatures.buildNormalizedKeypoints()
    imTwoFeatures.buildNormalizedKeypoints()

    # Compute feature matches
    matches = ImagePairMatches(imOneFeatures, imTwoFeatures)
    matches.computeMatches()

    # Draw matches
    imgOut = drawStereoMatches(
        imOne, imTwo, imOneFeatures, imTwoFeatures, matches
    )
    cv2.imshow(
        "Matches",
        cv2.cvtColor(pillowToArray(imgOut), cv2.COLOR_RGB2BGR),
    )

    # Now, compute a fundamental and homography matrix; first though,
    # we need to build an appropiate RANSAC dataset
    dataset: RANSACDataset[
        Tuple[Keypoint, Keypoint]
    ] = matches.toRANSACDataset(True)

    ransacEstimator: RANSACEstimator = RANSACEstimator(8)

    # Create a lambda to construct both types of matrices
    funMatConstructor = lambda: FundamentalMatrix(
        10,
        FundamentalMatrix.symetricTransferError,
        ORB_Fundamental_Scoring_Function,
    )
    homogMatConstructor = lambda: HomographyMatrix(
        10,
        HomographyMatrix.symetricTransferError,
        ORB_Homograhy_Scoring_Function,
    )

    from datetime import datetime

    start = datetime.now()

    fundamentalTask = asyncio.create_task(
        ransacEstimator.fit(
            dataset, funMatConstructor, 70, 7, False
        )
    )
    homographyTask = asyncio.create_task(
        ransacEstimator.fit(
            dataset, homogMatConstructor, 30, 7, False
        )
    )

    funMat, fundamentalInliers, funScore = await fundamentalTask
    homogMat, homographyInliers, homogScore = await homographyTask

    end = datetime.now()
    print(
        "RANSAC took", (end - start).total_seconds(), "seconds"
    )

    print("Fundamental Score:", funScore)
    print("Homography Score:", homogScore)

    assert type(funMat) == FundamentalMatrix
    possibleSols = funMat.getFourMotionHypotheses(CALIB)
    bestSol, numVisible, points = funMat.chooseSolution(
        CALIB,
        possibleSols,
        imOneFeatures.normalizedKeypoints,
        imTwoFeatures.normalizedKeypoints,
        fundamentalInliers,
        True
    )
    print(bestSol.cameraMat)
    print(numVisible)
    print(points.shape)

    imgOut = 0

    # Visualize with o3d
    # vector3d = o3d.utility.Vector3dVector(points)
    # pcd = o3d.geometry.PointCloud(vector3d)
    # o3d.visualization.draw_geometries([pcd])

    if ORB_Model_Pick_Homography(
        homogScore, funScore
    ) and ORB_IsModelGood(homogScore):
        assert type(homogMat) == HomographyMatrix
        imgOut = drawHomographyHypotheses(
            homogMat,
            imOne,
            imOneFeatures,
            imTwo,
            imTwoFeatures,
            matches,
        )
    elif ORB_IsModelGood(funScore):
        assert type(funMat) == FundamentalMatrix
        imgOut = drawEpipolarLines(
            funMat,
            imOne,
            imOneFeatures,
            imTwo,
            imTwoFeatures,
            matches,
        )
    else:
        print("No good model found")

    # assert isinstance(model, FundamentalMatrix)
    # assert isinstance(model, HomographyMatrix)

    # print(model.unnormalize(np.array(imOneFeatures.normalizeMat), np.array(imTwoFeatures.normalizeMat)))

    # imgOut = drawEpipolarLines(
    #     model,
    #     imOne,
    #     imOneFeatures,
    #     imTwo,
    #     imTwoFeatures,
    #     matches,
    # )
    # # imgOut = drawHomographyHypotheses(model, imOne, imOneFeatures, imTwo, imTwoFeatures, matches)

    # assert type(imgOut) == Image.Image
    # cv2.imshow(
    #     "Model",
    #     cv2.cvtColor(pillowToArray(imgOut), cv2.COLOR_RGB2BGR),
    # )
    cv2.waitKey(0)


if __name__ == "__main__":
    asyncio.run(run())
