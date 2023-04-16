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

    @abstractmethod
    def getScore(self, data: RANSACDataset[T]) -> float:
        """
        Given a RANSACDataset, determines the score of this model
        against that dataset. The score is used to determine which
        model is the best model, irrespective of the number of inliers.

        :param data: The dataset we are going to determine the score on.

        :return: Returns the score of this model against the given dataset.
        """
        pass
