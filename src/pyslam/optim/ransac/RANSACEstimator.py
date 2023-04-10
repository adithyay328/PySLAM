from typing import (
    TypeVar,
    Type,
    Awaitable,
    Callable,
    List,
    Optional,
    Tuple,
)
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

import asyncio
from asyncio import Future

import numpy as np

from pyslam.optim.ransac import RANSACModel, RANSACDataset

D = TypeVar("D")

def getModelAndInliers(
        fullDataset: RANSACDataset[D],
        subset: RANSACDataset[D],
        model: RANSACModel[D],
    ) -> Tuple[RANSACModel[D], np.ndarray]:
        """
        A helper function that returns a fitted RANSACModel and its predicted inliers in one function.

        :param fullDataset: The full dataset we were given.
        :param subset: A sub-set of the dataset to fit the model to.
        :param modelConstructor: A function tha takes no paramaters
          and returns an instance of the RANSACModel type we are trying to fit.
          Allows client code to use partial functions to pass in paramaters
          required by the constructor of the RANASCModel; these could be
          configuration paramaters, etc.

        :return: Returns a tuple of (estimated model, numpy array of inlier indices)
        """
        model.fit(subset)
        inliers: np.ndarray = model.findInlierIndices(
            fullDataset
        )

        return (model, inliers)


class RANSACEstimator:
    """A class responsible for estimating
    a RANSACModel from a RANSACDataset using
    a specified number of random iterations.
    Internally, all the constructor does is create
    a thread-pool for running estimation in parralel.

    :param numWorkers: The number of workers to instantiate
      in the internal ProcessPoolExecutor
    """

    def __init__(self, numWorkers: int):
        self.__processPool = ProcessPoolExecutor(numWorkers)

    async def fit(
        self,
        data: RANSACDataset[D],
        modelConstructor: Callable[[], RANSACModel[D]],
        iterations: int,
        dataPointsPerIteration: int,
        refit: bool,
    ) -> RANSACModel[D]:
        """
        Fits a RANSACModel to a given dataset. Runs
        each iteration in parralel, and optionally re-fits the model on
        all of the inliers determined by the best initial model.

        :param data: The RANSACDataset to fit to.
        :param modelConstructor: A function that takes no paramaters
          and returns an instance of the RANSACModel type we are trying to fit.
          Allows client code to use partial functions to pass in paramaters
          required by the constructor of the RANASCModel; these could be
          configuration paramaters, etc.
        :param iterations: The number of RANSAC iterations to run.
        :param dataPointsPerIteration: The number of datapoints to sample for each
          model fitting iteration.
        :param refit: Whether or not to refit the best determined model on its
          inliers.

        :return: Retuns the best determined model.
        """

        # 2 variables that keep track of the best model
        # and set of inliers we've seen so far
        bestModel: Optional[RANSACModel[D]] = None
        bestInliers: Optional[np.ndarray] = None

        # All the futures we get from queueing up calls
        futureObjects: List[Future] = []

        # Queue up n iterations of RANSAC
        loop = asyncio.get_running_loop()
        for i in range(iterations):
            # For each iteration of RANSAC, get a random sub-set
            randomSubset = data.getRandomSubsample(
                dataPointsPerIteration
            )

            # Now, get a future object representing this iteration,
            # and add it to the list
            futureObj = loop.run_in_executor(
                self.__processPool,
                getModelAndInliers,
                data, randomSubset, modelConstructor()
            )

            futureObjects.append(futureObj)

        # Now, wait for all the models to be estimated, and keep the best one
        for future in futureObjects:
            nextModel, nextInliers = await future

            # Keep this model only if there are more inliers
            if bestInliers is None or len(bestInliers) < len(
                nextInliers
            ):
                bestModel = nextModel
                bestInliers = nextInliers

        # Before doing anything else, make sure our optionals have their values. If not,
        # error out since something went wrong

        if bestModel is None or bestInliers is None:
            raise ValueError(
                "After RANSAC Estimation, our model or inlier array are still undetermined. Illegal state."
            )

        # At this point, we have our best model. If refit is set to true, refit on the best indices.
        # Otherwise, return as is

        if refit:
            inlierDataset: RANSACDataset[D] = data.getSubsample(
                bestInliers
            )
            modelInstance: RANSACModel[D] = modelConstructor()
            modelInstance.fit(inlierDataset)
            return modelInstance
        else:
            return bestModel
