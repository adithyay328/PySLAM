from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional
import asyncio
import multiprocessing
from multiprocessing import synchronize
from datetime import datetime, timezone, timedelta
import time
import sys

from pyslam.capture.Measurement import Measurement
from pyslam.pubsub.Publisher import Publisher
from pyslam.uid import UID
import pyslam.capture as capture

T = TypeVar("T", bound=Measurement)


class MeasurementSource(Publisher[T], Generic[T], ABC):
    """
    An abstract base class representing a source for
    Measurements; this could be a sensor, a ROS topic, a
    pre-recorded set of datapoints from a SLAM dataset, whatever.
    Has a UID that can be referenced by
    measurements, but apart from that is a pretty
    lightweight interface to subclass.

    :param uid: A UID that uniquely identifies this MeasurementSource
    """

    def __init__(self, uid: Optional[UID]):
        super().__init__()

        # Set initial values
        self._uid: UID = UID()
        if uid is not None:
            self._uid = uid

        # Register self with the lookup
        capture.GLOBAL_LOOKUP[self._uid] = self
