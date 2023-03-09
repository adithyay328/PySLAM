from abc import ABC, abstractmethod

import numpy as np


class Descriptor(ABC):
    """
    An abstract base class that can be subclassed
    by all the different types of Descriptors that
    we can work with.

    Descriptors only vary in the way their distances
    are computed, and so beyond using it for typing/
    identification of descriptor types, distance
    calculation is the only thing child classes
    need to implement

    :param descriptorArr: A numpy array containing the
        contents/data for this descriptor.
    """

    def __init__(self, descriptorArr: np.ndarray) -> None:
        self.data: np.ndarray = descriptorArr

    @abstractmethod
    def computeDistance(self, other: "Descriptor") -> float:
        """
        An abstract method that must be implemented that
        computes the distance between this descriptor's data
        and another descriptor's data.

        :param other: The descriptor to compute distance to.
        """
        pass
