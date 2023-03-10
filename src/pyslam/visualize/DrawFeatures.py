import cv2
from PIL.Image import Image
import numpy as np

from pyslam.image_processing.feature_descriptors.ImageFeatures import ImageFeatures
from pyslam.image_processing.cv_pillow import pillowToArray, arrayToPillowImage, PillowColorFormat

def drawFeatures(image : Image, features : ImageFeatures) -> Image:
    """
    Given an Image and a Features object, draw the features onto
    the image and return it.

    :param image: The Pillow Image to draw onto
    :param features: The Image Features to visualize
    
    :return: Returns a Pillow image with all Keypoints drawn
        onto it.
    """
    cv2Mat : cv2.Mat = pillowToArray(image)

    for keypoint in features.keypoints:
        hetCoords : np.ndarray = keypoint.asHeterogenous()

        cv2.circle(
            cv2Mat, (int(hetCoords[0]), int(hetCoords[1])), 2, (255, 0, 0)
        )
    
    return arrayToPillowImage(cv2Mat, PillowColorFormat.RGB)
