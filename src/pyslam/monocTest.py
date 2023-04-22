"""
Monocular testing of frame visualizer
on comma ai dataset.
"""
import asyncio
from typing import Optional, Tuple

import cv2
from PIL.Image import Image
import numpy as np

from pyslam.common_capture.MonocularVideoExtractor import MonocularVideoExtractor
from pyslam.image_processing.cv_pillow import arrayToPillowImage, PillowColorFormat, pillowToArray
from pyslam.visualize.PyGameFrameWindow import PyGameFrameWindow
from pyslam.image_processing.feature_descriptors.ImageFeatures import ImageFeatures
from pyslam.image_processing.feature_descriptors.extractors.end2end.orb_detect_and_compute import ORB_Detect_And_Compute
from pyslam.image_processing.feature_descriptors.descriptors.ORB import ORB
from pyslam.image_processing.feature_descriptors.ImagePairFeatureMatches import ImagePairMatches
from pyslam.visualize.DrawMatches import drawStereoMatches
from pyslam.pubsub import Publisher
from pyslam.optim.ransac import RANSACDataset, RANSACModel, RANSACEstimator
from pyslam.image_processing.feature_descriptors.Keypoint import Keypoint
from pyslam.epipolar_core import FundamentalMatrix, HomographyMatrix
from pyslam.systems.orb_slam import ORB_Homograhy_Scoring_Function, ORB_Fundamental_Scoring_Function, ORB_Model_Pick_Homography, ORB_IsModelGood

eon_f_focal_length = 910.0
eon_d_focal_length = 650.0
tici_f_focal_length = 2648.0
tici_e_focal_length = tici_d_focal_length = 567.0 # probably wrong? magnification is not consistent across frame

eon_f_frame_size = (1164, 874)
eon_d_frame_size = (816, 612)
tici_f_frame_size = tici_e_frame_size = tici_d_frame_size = (1928, 1208)

# aka 'K' aka camera_frame_from_view_frame
eon_fcam_intrinsics = np.array([
  [eon_f_focal_length,  0.0,  float(eon_f_frame_size[0])/2],
  [0.0,  eon_f_focal_length,  float(eon_f_frame_size[1])/2],
  [0.0,  0.0,                                          1.0]])

CALIB = eon_fcam_intrinsics

async def run():
    # Build a PyGameFrameWindow
    # and a MonocularVideoExtractor
    
    # Building a MonocularVideoExtractor
    videoExtractor = MonocularVideoExtractor("3.mp4", PillowColorFormat.RGB, cv2.COLOR_BGR2RGB)

    # Build a Publisher for Images
    pub = Publisher[Image]()

    # I really want to show epipolar lines, so draw that instead.
    frameWindow = PyGameFrameWindow(pub.subscribe())

    # Get a listener for the video extractor
    listener = videoExtractor.subscribe()

    # Start video loop
    videoExtractor.startCaptureLoop(20)

    frameWindow.startListenLoop()

    # Build an extractor for features
    imOneExtractor = ORB_Detect_And_Compute(600)

    # Keep a sliding window of 2 frames, with feature correpsondences
    # accross them; only swap when a good model is determined i.e.
    # a new keyframe
    firstFrame : Image = None
    secondFrame : Image = None
    firstImageFeatures : Optional[ImageFeatures] = None
    secondImageFeatures : Optional[ImageFeatures] = None

    # Keep track of current position
    position = np.array([0.0, 0.0, 0.0])

    ransacEstimator: RANSACEstimator = RANSACEstimator(8)

    # Create a lambda to construct a FundamentalMatrix
    funMatConstructor = lambda: FundamentalMatrix(
        10,
        FundamentalMatrix.symetricTransferError,
        ORB_Fundamental_Scoring_Function,
    )

    # Loop through frames
    while True:
        # Get a frame
        frame = listener.listen(block=True, timeout=-1)
        if frame is None:
            raise ValueError("Unexpected NoneType from camera")
        
        # First, extract features
        frameFeatures : ImageFeatures[ORB] = ImageFeatures[ORB](
                frame.image, imOneExtractor, imOneExtractor
        )
        frameFeatures.buildNormalizedKeypoints()

        # If we haven't yet populated first and second image, just toss this in
        # there
        if firstFrame is None or secondFrame is None:
            # Move second into first
            firstFrame = secondFrame
            firstImageFeatures = secondImageFeatures

            # Update second
            secondFrame = frame.image
            secondImageFeatures = frameFeatures

            # Skip this iteration
            continue
        
        # If we made it this far, the first 2 frames are populated,
        # and now we can actually do our keyframing

        # If we made it this far, we need to compute new
        # matches between the new frame and the first frame,
        # so do that now
        matches = ImagePairMatches(secondImageFeatures, frameFeatures)
        matches.computeMatches()

        # Compute pose esimates
        dataset: RANSACDataset[
            Tuple[Keypoint, Keypoint]
        ] = matches.toRANSACDataset(True)
        
        fundamentalTask = asyncio.create_task(
        ransacEstimator.fit(
            dataset, funMatConstructor, 30, 7, True
        )
        )

        funMat, fundamentalInliers, funScore = await fundamentalTask

        # Now that we have a fundamental matrix, we can compute the
        # poses
        assert(type(funMat) == FundamentalMatrix)
        possibleSols = funMat.getFourMotionHypotheses(CALIB)

        bestSol, numVisible, points, goodSol = funMat.chooseSolution(
            CALIB,
            possibleSols,
            firstImageFeatures.normalizedKeypoints,
            secondImageFeatures.normalizedKeypoints,
            fundamentalInliers,
        False)

        # If we have both frames, compute the matches
        # and draw them; visualize those matches
        # on the frame window
        stereoMatchFrame : Image = drawStereoMatches(
            secondFrame, frame.image, secondImageFeatures, frameFeatures, matches
        )
            
        # Publish to the frame window
        pub.publish(stereoMatchFrame)

        # If good sol is false, ignore it
        if not goodSol:
            continue
        
        # Otherwise, add to current position, and update frames

        # Add to position, and print
        position += bestSol.cameraMat[0:3, 3]
        print(position)
        
        firstFrame = secondFrame
        firstImageFeatures = secondImageFeatures

        secondFrame = frame.image
        secondImageFeatures = frameFeatures
                   

if __name__ == "__main__":
    asyncio.run(run())
