import multiprocessing

class CaptureManager:
  """This class is responsible for managing all physical cameras attached to this system;
  i.e. all ImageSensors are registered with this class. Also contains a central lookup for all
  images we extract, camera matrices, etc."""
  def __init__(self, stopEvent : multiprocessing.Event, captureRate=5):
    """
    :param stopEvent: A multi-processing event that tells the CaptureManager whether to
      stop the capture loop or not. Allows the SLAMSession to tell the CaptureManager to
      stop the capture loop once it starts.
    :param captureRate: An integer indicating how many captures we should take per second.
    """

    # Storing all passed in params
    self.__stopEvent = stopEvent
    self.__captureRate = captureRate

    # Dict mapping of sensor UID : sensor object
    self.sensors = {}

    # Dict mapping of capture timestep : list of all Capture UIDs at that timestep
    self.__capTimeStepUIDs = {}

    # Current capture timestep; increments every time our capture flag increments
    self.__capTimeStep = 0

    # The capture event that we use to trigger a new wave of captures
    self.__captureTriggerEvent = multiprocessing.Event()
    
    # A multithreading queue where registered cameras can dump their captures
    self.__captureQueue = multiprocessing.Queue()