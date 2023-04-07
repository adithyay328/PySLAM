from abc import ABC, abstractmethod, abstractstaticmethod
from typing import TypeVar, Generic

import numpy as np

from pyslam.optim.ransac import RANSACDataset

T = TypeVar("T")


class RANSACModel(Generic[T], ABC):
    """An abstract class representing some kind of model
    that we wish to estimate using RANSAC against some
    RANSACDataset. The generic type it takes in defines
    what kind of data it fits against.
    """

    @abstractmethod
    def fit(self, data: RANSACDataset[T]) -> None:
        """Fit the model to the specified RANSACDataset."""
        pass

    @abstractmethod
    def findInlierIndices(
        self, data: RANSACDataset[T]
    ) -> np.ndarray:
        """
        Given a RANSACDataset, determines which indices of its root dataset
        are inliers for this model. This is a needed step in determining
        the quality of a model, as models with the highest number of inliers
        have the highest quality. As well, by keeping track of which indices
        are inliers, we can optionally re-fit the model on inliers after
        selecting the best model.

        :param data: The dataset we are going to determine inliers on.

        :return: Returns a numpy array containing all root indices
          determined to be inliers.
        """
        pass
