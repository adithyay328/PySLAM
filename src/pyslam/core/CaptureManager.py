import multiprocessing
import datetime
import time

from .SensorWrapper import SensorWrapper

class CaptureManager:
  """This class is responsible for managing all physical cameras attached to this system;
  i.e. all ImageSensors are registered with this class. Also contains a central lookup for all
  images we extract, camera matrices, etc."""
  def __init__(self, stopEvent : multiprocessing.Event, captureRate=5):
    """
    :param stopEvent: A multi-processing event that tells the CaptureManager whether to
      stop the capture loop or not. Allows the SLAMSession to tell the CaptureManager to
      stop the capture loop once it starts. Must be clear when the loop starts, and set
      when we want to stop the capture loop
    :param captureRate: An integer indicating how many captures we should take per second.
    """

    # Storing all passed in params
    self.stopEvent = stopEvent
    self.captureRate = captureRate

    # Dict mapping of sensor UID : sensor object
    self.sensors = {}

    # Mapping of sensor UID : control queue
    self.sensorControlQueues = {}

    # Dict mapping of capture timestep : list of all Capture UIDs at that timestep
    self.stepToCaptureUIDs = {}

    # Current timestep
    self.currTimeStep = 0

    # Dict mapping of capture UID to capture object
    self.captures = {}
    
    # A multithreading queue where registered cameras can dump their captures
    self.__captureQueue = multiprocessing.Queue()

    # A multithreading queue where we output groups of captures, all from the same capture timestep
    self.timeStepGroupQueue = multiprocessing.Queue()

  def startCaptureLoop(self):
    """Starts the capture loop, exits when the stop event is given"""
    while not self.stopEvent.is_set():
      # Keep track of when the loop started
      startDTime = datetime.datetime.utcnow()

      # Set of all sensors that haven't given a capture for this timestep yet
      sensorsThatHaventReported = set(list(self.sensors.keys()))

      # Send capture trigger events to all message queues
      for k, v in self.sensorControlQueues:
        v.put("Capture!")

      # Wait until all sensors have given a response
      while len(sensorsThatHaventReported) > 0:
        # Read from the capture queue with a block
        capture = self.__captureQueue.get(block=True)

        # Pop the sensor UID off of the list of sensors we haven't heard from
        sensorsThatHaventReported.remove(capture.sensorWrapperUID)

        # Put in our captures map
        self.captures[capture.uid] = capture

        # Update step to capture UIDs to include this
        if self.currTimeStep not in self.stepToCaptureUIDs:
          self.stepToCaptureUIDs = []
        self.stepToCaptureUIDs[self.currTimeStep].append(capture.uid)

        # Continue
      
      # At this point, this capture timestep is done. Log to the time step
      # group queue
      self.timeStepGroupQueue.put(self.stepToCaptureUIDs[self.currTimeStep])

      # Increment curr timestep
      self.currTimeStep += 1

      # Sleep
      secondsToSleep = (1 / self.captureRate) - (datetime.datetime.utcnow() - startDTime).total_seconds()
      time.sleep(secondsToSleep)
  
  def register(self, sensorWrapper):
    """Registers a new sensor wrapper with self"""
    # Create a new sensor control queue
    sensorControlQueue = multiprocessing.Queue()

    # Set queues in the sensor wrapper
    sensorWrapper.controlQueue = sensorControlQueue
    sensorWrapper.externalCaptureQueue = self.__captureQueue

    # Update internal mapping of sensors
    self.sensors[sensorWrapper.uid] = sensorWrapper
    self.sensorControlQueues[sensorWrapper.uid] = sensorControlQueue