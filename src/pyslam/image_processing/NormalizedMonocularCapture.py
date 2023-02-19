import numpy as np

from pyslam.sensing.MonocularCameraCapture import MonocularCameraCapture

class NormalizedMonocularCapture:
    """This class takes in a Monocular Camera Capture, 
    and applies translation + isotropic(uniform) scaling to normalize
    the point's coordinates such that it is zero-meaned with the 
    mean distance of the points to the origin being sqrt(2). 
    This is the normalization expected by the normalized eight point 
    and normalized DLT algorithms for fundamental and homography 
    matrix estimation, respectively."""
    def __init__(self, monocularCameraCapture : MonocularCameraCapture):
        # Store the core monocular capture before running our normalization
        # procedures
        self.monocularCameraCapture = monocularCameraCapture

        # Extract the bw matrix as a numpy array; don't assign this
        # to self to save memory
        bwSourceMat = np.array(self.monocularCameraCapture.bwImgmat)

        # Compute the mean point coordinate; this is used for
        # translation to zero-mean the points, which is done before
        # the isotropic scaling
        self.meanCoord = np.mean(bwSourceMat, axis=0)

        # Construct the translation matrix associated with the translation
        # we're about to execute; useful for constructing 
        # un-normalized fundamental matrices from the normalized, for
        # example
        self.translationMat = np.eye(3)
        self.translationMat[:2, -1] = self.meanCoord

        # Apply the translation
        zeroMeanedMat = bwSourceMat - self.meanCoord

        # Now, compute mean magnitude, which we can then
        # apply isotropic scaling with
        normOfAllPoints = np.linalg.norm(zeroMeanedMat, axis=1)
        meanNorm = np.mean(normOfAllPoints)

        # The final scale factor we will use; we want to normalize
        # such that the final scale factor is sqrt(2), so that's computed
        # here
        self.finalScaleFactor = (2 ** 0.5) / meanNorm

        # Now, like before, comptue and set the 3x3 matrix that
        # represents this scale operation
        self.scaleMat = np.eye(3)
        self.scaleMat[0][0] = self.finalScaleFactor
        self.scaleMat[1][1] = self.finalScaleFactor
        
        # Apply isotropic normalization and store internally
        self.normalized = zeroMeanedMat * self.finalScaleFactor

        # Now, for convenience, compute the full transformation
        # we applied
        self.fullTransformMat = self.scaleMat @ self.translationMat