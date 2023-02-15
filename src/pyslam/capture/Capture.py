import abc

from pyslam.uid import (
    generateUID,
)


class Capture(abc.ABC):
    """This class represents a "capture" from a Sensor(akin to a single
    sensor measurement for other kinds of sensors). This is an
    abstract base class, with specific types of captures being
    subclassed from here. Subclasses can have extra properties
    specific to that type of capture, such as a depth map
    """

    def __init__(
        self,
        sensorWrapperUID,
    ):
        self.__sensorWrapperUID = sensorWrapperUID
        self.__uid = generateUID()

    @property
    def sensorWrapperUID(
        self,
    ):
        return self.__sensorWrapperUID

    @property
    def uid(
        self,
    ):
        return self.__uid
