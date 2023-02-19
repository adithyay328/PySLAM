import logging
import multiprocessing
import abc
import queue

from pyslam.uid import (
    generateUID,
)
from RawSensor import (
    RawSensor,
)


class SensorWrapper(abc.ABC):
    """This class wraps around a raw object for a physical camera
    sensor attached to our system.
    This class deals with capturing images from that sensor, and all the piping
    and inter-process communication related to that.

    The ImageSensor has a couple high-level functions it needs to support:
      1. Support for a capture loop. When commanded, image sensors should
       be able to put themselves into a capture loop state, where the
       timings of their captures are entirely decided by the capture
       loop in the CaptureManager. This should be in a different thread
       to allow for more accurate timing.
      2. Take a single capture. This seems like a pretty fundamental
        thing to be able to do,
         and so all SensorWrapper allow this. The way this is
         implemented is with 2 cases, one where the Sensor is in
         a capture loop and one where it isn't.
          a. When in a capture loop, we shouldn't inerfere with the
             capture and should instead grab a copy of the image from the
             camera loop. To do this, we have an internal queue
             where we post the most recent capture from the loop. When
             you want to get a capture, simply wait for this queue to have
             one on it.
          b. In the case where we aren't currently in a capture loop, simply
              start capturing, pull a frame, and stop capturing.

    """

    def __init__(
        self,
        name: str,
        rawSensor: RawSensor,
    ):
        """
        :param name: A string name for this sensor
        :param rawSensor: The raw sensor this wrapper wraps around

        :return: Instantiated ImageSensor class.
        """
        # Setting all params passed in
        self.name = name
        self.rawSensor = rawSensor

        # These need to be set by the capture manager when registered
        self.controlQueue: multiprocessing.Queue = (
            multiprocessing.Queue()
        )
        self.externalCaptureQueue: multiprocessing.Queue = (
            multiprocessing.Queue()
        )

        self.uid = generateUID()

        # The following lines deal with logic related
        # to the capture state of the camera.
        # This variable indicates if the camera is currently
        # "recording", in which case we can easily pull a frame
        self.active = False
        self.inCaptureLoop = False
        # This is a queue we use internally to allow
        # other threads to get the most recent
        # capture from the camera without interfering with
        # a capture loop, if there is one.
        self.internalCaptureQueue = multiprocessing.Queue()
        # This is a recurrent lock object that we use to allow
        # one thread to take control  the current capture state
        # of the camera. This is mainly used in a) the capture
        # loop and b) the function to get a single image from the camera.
        self.captureLock = multiprocessing.RLock()

    def __startCapture(
        self,
    ):
        """Sets active flag to true, acquires the capture lock,
        and configures our sensor to be ready for capture"""

        # Take ownership of the capture state of this Sensor;
        # this will be released if we
        # error out, or when endCapture is called
        self.captureLock.acquire()

        if self.__active:
            # Log a warning and release lock
            logging.warning(
                "Attempted to startCapture while sensor is already capturing"
            )
            self.captureLock.release()
        else:
            # Ready the sensor, and don't release the lock.
            self.rawSensor.activateSensor()
            self.__active = True

    def __endCapture(
        self,
    ):
        """Sets active flag to false, clears the capture lock,
        and tears down the sensor."""
        with self.captureLock:
            if not self.__active:
                logging.warning(
                    "Attempted to stopCapture while sensor is already inactive"
                )
            else:
                self.rawSensor.teardownSensor()
                self.__active = False

        # One more release for the lock we got in startCapture
        self.captureLock.release()

    def __captureLoop(
        self,
    ):
        """The internal capture loop function that startCaptureLoop executes in a
        different thread"""
        with self.captureLock:
            # Start capture
            self.__startCapture()

            # Set capture loop to true
            self.__inCaptureLoop = True

            # Continuously get messages from the control queue
            while True:
                # Get a string from the control queue,
                # blocking until we have one
                controlString = self.controlQueue.get(
                    block=True,
                    timeout=None,
                )

                # If we got "STOP", break out of this loop
                if (
                    type(controlString) is str
                    and controlString == "NONE"
                ):
                    break

                # Otherwise, continue; we got the signal to take a
                # capture, so get that capture and
                # log it to BOTH the external and internal queue
                newCapture = self.rawSensor.capture(self.uid)

                # For the internal queue, clear it first and then log
                # this to it
                while not self.internalCaptureQueue.empty():
                    try:
                        self.internalCaptureQueue.get(block=False)
                    except queue.Full:
                        # If we errored the queue is empty, so skip
                        break

                # Log to internal queue
                self.internalCaptureQueue.put(
                    newCapture,
                    block=True,
                )

                # Also log to external queue
                self.externalCaptureQueue.put(
                    newCapture,
                    block=True,
                )

                # Done, keep looping

            # If we got here we were ordered to stop capture,
            # so do cleanup here
            self.__inCaptureLoop = False
            self.__endCapture()

    def startCaptureLoop(
        self,
    ):
        """External function to start the capture loop
        in its own multi-process"""

        # Start the capture loop process; it will terminate on its own
        p = multiprocessing.Process(target=self.__captureLoop)
        p.start()

    def getCapture(
        self,
    ):
        """
        This function implements a thread-safe and non-intrusive
        way to get the latest capture from the sensor. It deals
        with the problems of co-operating with the capture loop
        if it's currently running, and generally makes it easy
        to get a capture from a SensorWrapper
        """

        # 2 major cases; one where the wrapper is currently in
        # a capture loop in a different thread, and one
        # where it's not.
        if self.__inCaptureLoop:
            # In this case, just return the next mesasge on the
            # internal capture queue
            return self.internalCaptureQueue.get(block=True)
        else:
            # If not, acquire RLock, configure sensor and
            # get a capture
            with self.captureLock:
                self.__startCapture()
                cap = self.rawSensor.capture(self.uid)
                self.__endCapture()

                return cap
