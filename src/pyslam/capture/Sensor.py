from typing import TypeVar, Generic, Optional
from abc import ABC, abstractmethod
import multiprocessing
from multiprocessing import synchronize
from datetime import datetime, timezone, timedelta
import asyncio
import time

from pyslam.capture.Measurement import Measurement
from pyslam.capture.MeasurementSource import MeasurementSource
from pyslam.uid import UID

T = TypeVar("T", bound=Measurement)


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

    async def captureLoop(self, captureRate: float) -> None:
        """
        Function to run the capture loop. Execute this
        using an asyncio task to keep it alive while
        you want this sensor to capture. Cancellation
        is also easily handled by simply sending a cancel
        request to the enclosing task.

        :param captureRate: The rate at which the underlying
          sensor and this wrapper should get new measurements.
          Units are in measurements per second.
        """

        # Set active to true
        with self.__lock:
            self.__active = True
            self.makeActive(captureRate)

        # This function only runs the capture loop
        # on its own multiprocessing process, and handles
        # cleaup afterwards
        p = multiprocessing.Process(
            target=self.__internalCaptureLoop,
            args=(captureRate,),
        )

        p.start()

        try:
            # The try block doens't actually do anything, it just waits
            while True:
                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            # Run all cleanup here; this allows
            # for a graceful shutdown
            with self.__lock:
                self.__active = False
                self.leaveActive()

            p.join()

            # Re-raise error to propogate it
            raise
