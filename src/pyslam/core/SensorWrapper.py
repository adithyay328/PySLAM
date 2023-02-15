import logging
import multiprocessing
import abc

import numpy as np

from uid import generateUID
from RawSensor import RawSensor

class SensorWrapper(abc.ABC):
  """This class wraps around a raw object for a physical camera sensor attached to our system.
  This class deals with capturing images from that sensor, and all the piping 
  and inter-process communication related to that. This is called 
  "Image Sensor" to distinguish it from the Camera class, which represents a Camera
  matrix in our odometry problem. This is a PHYSICAL camera.
  
  The ImageSensor has a couple high-level functions it needs to support:
    1. Support for a capture loop. When commanded, image sensors should be able to
       put themselves into a capture loop state, where the timings of their captures
       are entirely decided by the capture loop in the Capture Manager. This should be in a
       different thread to allow for more accurate timing.
    2. Take a single capture. This seems like a pretty fundamental thing to be able to do,
       and so all image sensors allow this. The way this is implemented is with 2 cases,
       one where the camera is in a capture loop and one where it isn't.
        a. When in a capture loop, we shouldn't inerfere with the capture and should instead
           grab a copy of the image from the camera loop. To do this, we have an internal queue
           where we post the most recent capture from the loop. When you want to get a capture,
           simply wait for this queue to have one on it.
        b. In the case where we aren't currently in a capture loop, simply start capturing, pull a frame,
           and stop capturing.
       
  """
  def __init__(self, name : str, rawSensor : RawSensor,
    captureEvent : multiprocessing.Event, stopEvent : multiprocessing.Event, captureQueue : multiprocessing.Queue):
    """
    :param name: A string name for this sensor
    :param rawSensor: The raw sensor this wrapper wraps around.
    :param captureEvent: A threading event that this sensor will wait on
      before triggering a capture in the capture loop.
    :param stopEvent: A threading event that allows capture manager to
      signal this sensor to stop capturing.
    :param captureQueue: A queue we use to log our Capture objects to the CaptureManager

    :return: Instantiated ImageSensor class.
    """
    # Setting all params passed in
    self.__name = name
    self.rawSensor = rawSensor
    self.__captureEvent = captureEvent
    self.__stopEvent = stopEvent
    self.__externalCaptureQueue = captureQueue
    
    self.__uid = generateUID()

    ## The following lines deal with logic related to the capture state of the camera.
    # This variable indicates if the camera is currently "recording",
    # in which case we can easily pull a frame
    self.__active = False
    self.__inCaptureLoop = False
    # This is a queue we use internally to allow other threads to get the most recent
    # capture from the camera without interfering with a capture loop, if there is one.
    self.__internalCaptureQueue = multiprocessing.Queue()
    # This is a recurrent lock object that we use to allow one thread to take control
    # of the current capture state of the camera. This is mainly used in a) the capture
    # loop and b) the function to get a single image from the camera.
    self.__captureLock = multiprocessing.RLock()
  
  @property
  def name(self):
    return self.__name
  
  @property
  def uid(self):
    return self.__uid
  
  def __startCapture(self):
    """Sets active flag to true, acquires the capture lock, 
    and configures our sensor to be ready for capture"""

    # Take ownership of the capture state of the camera; this will be released if we
    # error out, or when endCapture is called
    self.__captureLock.acquire()

    if self.__active:
      # Log a warning and release lock
      logging.warning("Attempted to startCapture while camera is already capturing")
      self.__captureLock.release()
    else:
      # Ready the sensor, and don't release the lock.
      self.rawSensor.activateSensor()
      self.__active = True
  
  def __endCapture(self):
    """Sets active flag to false, clears the capture lock, 
    and stops openCV camera capture"""
    with self.__captureLock:
      if not self.__active:
        logging.warning("Attempted to stopCapture while camera is already inactive")
      else:
        self.rawSensor.teardownSensor()
        self.__active = False
    
    # One more release for the lock we got in startCapture
    self.__captureLock.release()
    
  def __captureLoop(self):
    """The internal capture loop function that startCaptureLoop executes in a
    different thread"""
    with self.__captureLock:
      # Start capture
      self.__startCapture()

      # Set capture loop to true
      self.__inCaptureLoop = True

      # Keep running while the stop flag isn't set
      while not self.__stopEvent.is_set():
        # Wait on the capture event for 100ms; if we get it,
        # capture and log our capture object. Otherwise, we timed out,
        # in which case we should re-start the loop to ensure we comply
        # with the stop event quickly
        if self.__captureEvent.wait(timeout=0.1):
          # We got the signal to take a capture, so get that capture and
          # log it to BOTH the external and internal queue
          newCapture = self.rawSensor.capture(self.__uid)

          # For the internal queue, clear it first and then log this to it
          while not self.__internalCaptureQueue.empty():
            try:
              self.__internalCaptureQueue.get(block=False)
            except:
              # If we errored the queue is empty, so skip
              break
          
          # Log to internal queue
          self.__internalCaptureQueue.put(newCapture, block=False)

          # Also log to external queue
          self.__externalCaptureQueue.put(newCapture, block=False)

          # Done, so just go to the next loop
      
      # If we got here we were ordered to stop capture, so do cleanup here
      self.__inCaptureLoop = False
      self.__endCapture()
  
  def startCaptureLoop(self):
    """External function to start the capture loop in its own multi-process"""

    # Start the capture loop process; it will terminate on its own
    p = multiprocessing.Process(target=self.__captureLoop)
    p.start()
  
  def getCapture(self):
    """
    This function implements a thread-safe and non-intrusive way to get the latest
    capture from the sensor. It deals with the problems of co-operating with the capture loop
    if it's currently running, and generally makes it easy to get a capture from a SensorWrapper"""

    # 2 major cases; one where the wrapper is currently in a capture loop in a different thread, and one
    # where it's not.
    if self.__inCaptureLoop:
      # In this case, just return the next mesasge on the internal capture queue
      return self.__internalCaptureQueue.get(block=True)
    else:
      # If not, acquire RLock, configure sensor and get a capture
      with self.__captureLock:
        self.__startCapture()
        cap = self.rawSensor.capture(self.__uid)
        self.__endCapture()

        return cap