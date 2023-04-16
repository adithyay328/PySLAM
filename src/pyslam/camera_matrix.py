"""
Implements a notion of a camera matrix, which
basically is just a combination of translation
and rotation. Mainly just creates a common
interface for these kinds of things.
"""
import numpy as np
import transforms3d


class Camera:
    """
    A class representing a camera,
    while really is just intrinsics and extrinsics
    """

    def __init__(
        self,
        intrinsics: np.ndarray,
        translation: np.ndarray = np.zeros(3),
        rotation: np.ndarray = np.eye(3),
    ):
        self.rotation = rotation
        self.translation = translation
        self.extrinsics: np.ndarray = np.hstack(
            (rotation, translation.reshape(3, 1))
        )
        self.intrinsics: np.ndarray = intrinsics

        # A camera matrix, which is really the product of extrinsics and intrinsics
        self.cameraMat: np.ndarray = (
            self.intrinsics @ self.extrinsics
        )
