from copy import copy

import numpy as np


class Keypoint:
    """This class is quite simple; it corresponds to
    a 2D numpy array, which represents a 2D coordinate
    in an opencv matrix(an image).

    If the numpy array has 2 numbers in it, we assume
    the given point to be heterogenous; if 3 numbers
    are in it, we assume it to be homogenous. This
    class has functions to do simple conversion
    to-and-fro heterogenous and homogenous coordinates.

    :param coords: The 2D coordinates of this keypoint,
      either as a homogenous or heterogenous coordinate;
      as such, shape should be either (2,) or (3,)
    """

    def __init__(self, coords: np.ndarray) -> None:
        # Make sure that the given coords has a valid shape
        if (
            type(coords) != np.ndarray
            or len(coords.shape) != 1
            or coords.shape[0] not in [2, 3]
        ):
            raise ValueError(
                f"Expected input coordinate to have shape (2) or (3); instead got {coords.shape}"
            )

        self.__coords: np.ndarray = coords

    @property
    def coords(self) -> np.ndarray:
        """
        Getter for the internal coordinates of this Keypoint,
        without converting to heterogenous or homogenous.

        :return: A copy of the internal numpy array
        """
        return self.__coords.copy()

    @coords.setter
    def coords(self, newCoords: np.ndarray) -> None:
        """
        Setter function for the internal coordinates. Does
        internal checks to ensure that the coordinates are valid.

        :param newCoords: New coordinates that we want to update the
            internall coordinates to. Must be 2 or 3 wide, same as
            constructor.
        """
        # Make sure that the given coords has a valid shape
        if (
            type(newCoords) != np.ndarray
            or len(newCoords.shape) != 1
            or newCoords.shape[0] not in [2, 3]
        ):
            raise ValueError(
                f"Expected input coordinate to have shape (2) or (3); instead got {newCoords.shape}"
            )

        self.__coords = newCoords.copy()

    def makeHeterogenous(self) -> None:
        """Converts the internal representation of the keypoint into heterogenous for storage"""
        if self.__coords.shape[0] == 3:
            hetCoords: np.ndarray = (
                self.__coords[0:3] / self.__coords[-1]
            )
            self.__coords = hetCoords

    def makeHomogenous(self) -> None:
        """Converts the internal representation of the keypoint into homogenous for storage"""
        if self.__coords.shape[0] == 2:
            self.__coords = np.hstack(
                (self.__coords, np.array([1]))
            )

    def asHeterogenous(self) -> np.ndarray:
        """Returns a heterogenous representation of the internal coordinates

        :return: Returns the internal point array, converted to heterogenous,
            which means a 2D vector with no scale dimension.
        """
        if self.__coords.shape[0] == 2:
            return self.__coords.copy()
        else:
            hetCoords: np.ndarray = (
                self.__coords[0:3] / self.__coords[3]
            )
            return hetCoords.copy()

    def asHomogenous(self) -> np.ndarray:
        """Returns a homogenous representation of the internal coordinates

        :return: Returns the internal point array, converted to homogenous,
            which means a 3D vector with a scale dimension.
        """
        if self.__coords.shape[0] == 3:
            return self.__coords.copy()
        else:
            homCoords: np.ndarray = np.hstack(
                (self.__coords, np.array([1]))
            )
            return homCoords.copy()
