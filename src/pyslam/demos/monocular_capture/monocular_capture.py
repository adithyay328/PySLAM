from typing import Optional
from PIL.Image import Image

from pyslam.common_capture.UncalibratedMonocularCamera import (
    MonocularUncalibratedFileCamera,
    MonocularUncalibratedCameraMeasurement,
)
from pyslam.image_processing.cv_pillow import PillowColorFormat
from pyslam.pubsub.MessageQueue import MessageQueue
from pyslam.pubsub.Publisher import Publisher
from pyslam.visualize.PyGameFrameWindow import PyGameFrameWindow


def run() -> None:
    """
    Runs the monocular capture demo.
    """

    # Prompt the user for the camera file name
    cameraFName = input(
        "Please type in the file name of the camera: "
    )

    # Build a sensor we can listen to
    cameraSensor: MonocularUncalibratedFileCamera = (
        MonocularUncalibratedFileCamera(
            cameraFName, PillowColorFormat.RGB, -1
        )
    )

    # Get a message queue to listen on
    msgQueue: MessageQueue[
        MonocularUncalibratedCameraMeasurement
    ] = cameraSensor.subscribe()

    # Create an image publisher
    imagePublisher: Publisher[Image] = Publisher[Image]()

    # Spawn the pygame window
    pyGameFrameWindow: PyGameFrameWindow = PyGameFrameWindow(
        imagePublisher.subscribe()
    )

    # Start the sensor capture loop
    cameraSensor.startCaptureLoop(60)

    # Start the pygame window
    pyGameFrameWindow.startListenLoop()

    # Convert the MonocularUncalibratedCameraMeasurement to a PIL Image
    while True:
        nextCamMeasure: Optional[
            MonocularUncalibratedCameraMeasurement
        ] = msgQueue.listen(block=True, timeout=-1)

        assert nextCamMeasure is not None

        # Publish the image
        imagePublisher.publish(nextCamMeasure.image)


if __name__ == "__main__":
    run()
