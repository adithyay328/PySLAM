"""
This module defines some helper logic to convert to-and-fro
Pillow images and CV2 Mats.
"""

from enum import Enum
from typing import Union

from PIL import Image
import cv2
import numpy as np

class PillowColorFormat(Enum):
    """An Enum defining the 2 color formats
    we work with in Pillow; RGB and "L"(grayscale)
    """
    RGB = 0
    L = 1

def arrayToPillowImage(inArr : Union[cv2.Mat, np.ndarray], colorFormat : PillowColorFormat) -> Image.Image:
    """
    Given an input cv2 Mat or numpy array(they're the same under the hood),
    return a Pillow image with the same contents.

    :param inArr: The input array to convert
    :param colorFormat: The desired color format of the resultant
        Pillow Image.

    :return: The resultant Pillow image.
    """
    if colorFormat == PillowColorFormat.RGB:
        return Image.fromarray(inArr, "RGB")
    elif colorFormat == PillowColorFormat.L:
        return Image.fromarray(inArr, "L")
    else:
        raise ValueError(f"Invalid value of PillowColorFormat {colorFormat}")
    

def pillowToArray(inImg : Image.Image) -> np.ndarray:
    """
    Given an input Pillow image, returns a numpy array
    with the same contents as the image.

    :param inImg: The input Pillow image to convert.

    :return: Returns a numpy array with the same contents
    """
    return np.asarray(inImg)