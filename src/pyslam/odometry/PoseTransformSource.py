from abc import ABC, abstractmethod

from pose3d import Transform


class PoseTransformSource(ABC):
    """This is an abstract base class that is sub-classed
    by any class that can return an estimation of pose
    transformation; with visual odometry these could be
    Fundamental and Homography matrices, and in the case
    of VIO this could be an instance of IMU preintegration
    """

    @abstractmethod
    def getTransform(self) -> Transform:
        """
        Returns the pose transformation
        corresponding to this object.
        """
