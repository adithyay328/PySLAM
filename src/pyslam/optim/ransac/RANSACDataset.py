import random
from typing import List, TypeVar, Generic, Optional, Dict

import numpy as np

T = TypeVar("T")


class RANSACDataset(Generic[T]):
    """A Generic class representing a RANSAC dataset.
    Expects data to be all in-memory as an array, and for
    that to be passed in; this is just an assumption we make
    for simplicity.

    :param data: The data we are using in our dataset. We assume this
      to be an array-like structure.
    :param indices: The indices corresponding to our data. Allows us to
      keep track of which datapoints correspond to which dataset indices,
      which is especially useful once we compute random sub-samples.

      By convention, all paramater indices correspond to the indices in the root dataset;
      so, for example, if the first item in this sub-slice of the root dataset
      corresponds to item 5 of the root dataset, the first item in indices
      should be 4(5th element with zero indexing)
    """

    def __init__(
        self, data: List[T], indices: Optional[np.ndarray]
    ) -> None:
        self.data: List[T] = data
        self.indices = (
            np.array([i for i in range(len(self.data))])
            if indices is None
            else indices
        )

        # One thing we'll build right now, for performance, are 2
        # dicts, one allowing us to do a lookup of root idx -> idx in this array,
        # and another for lookups of idx in this array -> root idx
        self.rootIdxToDataIdx: Dict[int, int] = {}
        self.dataIdxToRootIdx: Dict[int, int] = {}
        for dataIdx, rootIdx in enumerate(self.indices):
            self.rootIdxToDataIdx[rootIdx] = dataIdx
            self.dataIdxToRootIdx[dataIdx] = rootIdx

    def __len__(self) -> int:
        return len(self.data)

    def getSubsample(
        self, indices: np.ndarray
    ) -> "RANSACDataset[T]":
        """Given a set of root idx indices, return a new RANSACDataset
        with all requested data in it

        :param indices: A numpy array containing all root indices
          whose data we want in the resultant sub-sample. These indices
          correspond to the root indices, not indices of this object's
          data array.

        :return: Returns a new RANSACDataset with the specified data.
        """
        data: List[T] = []

        for rootIdx in indices:
            data.append(
                self.data[self.rootIdxToDataIdx[rootIdx]]
            )

        return RANSACDataset[T](data, indices)

    def getRandomSubsample(
        self, numOfPoints: int
    ) -> "RANSACDataset[T]":
        """Vanilla RANSAC, which is the scheme this
        dataset expects to be involved with, uses
        random, uniform sub-sampling. This function
        returns another object of type self, but with
        less DataPoints in it.

        :param numOfPoints: The number of datapoints to
          sample for this new sub-sample.
        :return: Returns a new RANSACDataset with a sub-sample
          of the data in this datasets
        """

        rootIndicesToSample = np.array(
            [
                random.choices(
                    list(self.rootIdxToDataIdx.keys()),
                    k=numOfPoints,
                )
            ]
        )

        return self.getSubsample(rootIndicesToSample)
