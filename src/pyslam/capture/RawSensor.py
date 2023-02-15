import abc

from .Capture import (
    Capture,
)


class RawSensor(abc.ABC):
    """An abstract class that implements
    the core functions of a raw sensor. Specific
    types of sensors can derive themselves from this.
    """

    @abc.abstractmethod
    def activateSensor(
        self,
    ):
        """An function that does any sensor
        initialization and tells it to be prepared
        for a capture call. As an example, this might
        tell opencv to start capturing video
        from a camera so that when the SensorWrapper
        calls for a capture, it happens as
        fast as the sensor can manage."""
        pass

    @abc.abstractmethod
    def teardownSensor(
        self,
    ):
        """The opposite of startCapture; any cleanup that needs
        to be done is done here,
        and this function tells the sensor to stop recording and
        go into a standby state.
        In the context of a camera, this could mean telling opencv
        to stop capturing frames
        and put the camera into an idle state."""
        pass

    @abc.abstractmethod
    def capture(
        self,
        sensorWrapperUID,
    ) -> Capture:
        """Returns a Capture object populated with the data
        the sensor collected."""
        pass
