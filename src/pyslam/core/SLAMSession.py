class SLAMSession:
  """Base class for all SLAM Session objects, which represents
  one consistent session of visual-inertial odometry."""
  def __init__(self):
    # Dict mapping of UIDs to objects; note that THESE DO NOT NEED TO BE CAPTURE OBJECTS!
    # This can store capture objects, camera matrices, or really anything else. I'm not crazy
    # about this architecture of having a single lookup for anything, but it is quite simple;
    # I might fix this in the future, but for now we're using this
    self.objectMap = {}

    pass