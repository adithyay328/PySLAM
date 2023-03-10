from typing import Optional
from PIL.Image import Image

from pyslam.capture.common.monocular_uncalibrated_camera import (
    MonocularUncalibratedFileCamera,
    MonocularUncalibratedCameraMeasurement,
)
from pyslam.image_processing.cv_pillow import PillowColorFormat
from pyslam.pubsub.MessageQueue import MessageQueue
from pyslam.pubsub.Publisher import Publisher
from pyslam.visualize.PyGameFrameWindow import PyGameFrameWindow
from pyslam.image_processing.feature_descriptors.ImageFeatures import (
    ImageFeatures,
)
from pyslam.image_processing.feature_descriptors.extractors.end2end.orb_detect_and_compute import (
    ORB_Detect_And_Compute,
)
from pyslam.visualize.DrawFeatures import drawFeatures


def run() -> None:
    """
    Runs the monocular features demo.
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

    # Get a message queue from the camera
    msgQueue: MessageQueue[
        MonocularUncalibratedCameraMeasurement
    ] = cameraSensor.subscribe()

    # Create an image publisher that pushes frames to the PyGame window;
    # we will push to this after computing features
    imagePublisher: Publisher[Image] = Publisher[Image]()

    # Spawn the pygame window
    pyGameFrameWindow: PyGameFrameWindow = PyGameFrameWindow(
        imagePublisher.subscribe()
    )

    # Start the sensor capture loop
    cameraSensor.startCaptureLoop(40)

    # Start the pygame window
    pyGameFrameWindow.startListenLoop()

    # Construct our extractor
    extractor: ORB_Detect_And_Compute = ORB_Detect_And_Compute(
        500
    )

    # Takes in mononcular measurements, computes features, draws them onto an image,
    # and then pushes to the PyGame Window
    while True:
        nextCamMeasure: Optional[
            MonocularUncalibratedCameraMeasurement
        ] = msgQueue.listen(block=True, timeout=-1)

        assert nextCamMeasure is not None

        # Compute features and descriptors
        imageFeatures: ImageFeatures = ImageFeatures(
            nextCamMeasure.image, extractor, extractor
        )

        # Draw features onto the image
        imageWithFeatures: Image = drawFeatures(
            nextCamMeasure.image, imageFeatures
        )

        imagePublisher.publish(imageWithFeatures)


if __name__ == "__main__":
    run()
