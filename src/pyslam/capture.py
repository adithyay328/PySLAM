"""
Contains core logic and types related to 
getting measurements
from sensors and other sources.
"""

from abc import ABC, abstractmethod
from copy import copy
import weakref
from typing import TypeVar, Generic, Optional, Union
import multiprocessing
from multiprocessing import synchronize
import time
from datetime import datetime, timezone, timedelta

from pyslam.uid import UID
from pyslam.pubsub.Publisher import Publisher

# This WeakDict serves as a lookup for
# Measurements, MeasurementSources and Sensors.
# Allows global lookups, while not impeding
# garbage collection
GLOBAL_LOOKUP: weakref.WeakValueDictionary[
    UID, Union["Measurement", "MeasurementSource", "Sensor"]
] = weakref.WeakValueDictionary()


class Measurement(ABC):
    """
    A generic base class that represents some kind of
    capture/measurement from a sensor.

    :param uid: A UID object that uniquely identifies this measurement.
    :param sourceUID: The UID of the MeasurementSource
    :param timestamp: A datetime object indicating the time when this
      measurement was taken. Timezones are always in UTC.
    """

    def __init__(
        self,
        uid: Optional[UID],
        sourceUID: UID,
        timestamp: Optional[datetime] = None,
    ) -> None:
        # Set initial values
        self.__uid: UID = UID()
        self.__timestamp: datetime = datetime.now(
            tz=timezone.utc
        )
        self.__sourceUID: UID = sourceUID

        # Update any attributes that are passed in via optionals
        if uid is not None:
            self.__uid = uid
        if timestamp is not None:
            self.__timestamp = timestamp

        # Register self with the lookup
        GLOBAL_LOOKUP[self.__uid] = self

    @property
    def uid(self) -> UID:
        """Returns a copy of the UID for this measurement"""
        return copy(self.__uid)

    @property
    def timestamp(self) -> datetime:
        """Returns a copy of the timestamp of this measurement"""
        return copy(self.__timestamp)

    @property
    def sourceUID(self) -> UID:
        """
        Returns a reference to the MeasurementSource that this measurement
        came from
        """
        return copy(self.__sourceUID)


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
        GLOBAL_LOOKUP[self._uid] = self


class Sensor(MeasurementSource[T], Generic[T], ABC):
    """
    A special kind of MeasurementSource that is intended
    to wrap around a physical sensor; allows client code
    to implement just 3 functions, with the rest of the
    capture loop managed by generic code.

    :param uid: A UID that uniquely identifies this Sensor
    """

    def __init__(self, uid: Optional[UID]) -> None:
        # Init Measurement Source
        super().__init__(uid)

        # A state variable indicating whether the
        # capture loop is still active or not
        self.__active: bool = False

        # A process containing the capture loop
        self.__process: Optional[multiprocessing.Process] = None

        # A lock to access the above variable
        self.__lock: synchronize.RLock = multiprocessing.RLock()

    @abstractmethod
    def makeActive(self, captureFrequency: float) -> None:
        """
        When called, the framework is asking this sensor to be
        prepared to start returning captures. Internally, this
        function can be used for any setup needed to get the sensor
        into an active state.

        For example, for a camera, this might
        start an interal capture loop that gets opencv frames from the
        camera at some frequency, and then saves them to some internal
        buffer. This would allow the getCapture() function to complete
        almost instantly, by simply popping the most recent capture
        off the buffer.

        :param captureFrequency: The number of times per second this
          sensor should yield a measurement.
        """
        pass

    @abstractmethod
    def getMeasurement(self) -> T:
        """A function that returns the latest
        measurement from this sensor. This is expected
        to return almost immediately"""
        pass

    @abstractmethod
    def leaveActive(self) -> None:
        """
        Performs any cleanup to take this sensor out of an active state.
        For example, turn a camera sensor to an idle state and stop
        capturing frames.
        """
        pass

    def __internalCaptureLoop(self, captureRate: float) -> None:
        """
        An internal function that does all the work related to
        getting sensor measurements, propogating to all
        listeners and waiting for the appropriate amount of time
        to run next

        :param captureRate: A float indicating how many measurements
          we should pull from the sensor per second.
        """
        while True:
            # Check if we should actually run in here; access and check lock
            with self.__lock:
                if not self.__active:
                    break

            # Monitor start time using datetime; allows for more
            # accurate sleep timing
            loopStartTime: datetime = datetime.now(
                tz=timezone.utc
            )

            nextMeasurement: T = self.getMeasurement()

            # Propogate to listeners
            self.publish(nextMeasurement)

            # Now, just wait the appropriate amount of time
            desiredNextRunTime: datetime = (
                loopStartTime
                + timedelta(seconds=1 / captureRate)
            )
            sleepComputeTime: datetime = datetime.now(
                tz=timezone.utc
            )
            secondsToSleep: float = (
                desiredNextRunTime - sleepComputeTime
            ).total_seconds()
            time.sleep(max(0, secondsToSleep))

    def startCaptureLoop(self, captureRate: float) -> None:
        """
        Starts the capture loop on a new process. This will
        return immediately, and the capture loop will run
        in the background.

        :param captureRate: The rate at which the underlying
          sensor and this wrapper should get new measurements.
          Units are in measurements per second.
        """
        with self.__lock:
            self.__active = True
            self.makeActive(captureRate)
            self.__process = multiprocessing.Process(
                target=self.__internalCaptureLoop,
                args=(captureRate,),
            )
            self.__process.start()

    def stopCaptureLoop(self) -> None:
        """
        Stops the capture loop. This is a blocking function,
        and will not return until the capture loop is stopped.
        """
        with self.__lock:
            self.__active = False

            if self.__process is None:
                raise ValueError("Capture loop not running")

            self.__process.join()
            self.__process = None

            self.leaveActive()
