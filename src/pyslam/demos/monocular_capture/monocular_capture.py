import asyncio
from typing import Optional

import cv2
import pygame
import multiprocessing

from pyslam.capture.common.monocular_uncalibrated_camera import (
    MonocularUncalibratedFileCamera,
    MonocularUncalibratedCameraMeasurement,
)

from pyslam.pubsub.MessageQueue import MessageQueue


def pygameUpdater(
    msgQueue: MessageQueue[
        MonocularUncalibratedCameraMeasurement
    ],
) -> None:
    """
    A function that spawns a pygame window, and blits
    new opencv frames onto it as they are made available
    by the underlying camera.

    :msgQueue: The message queue that this thread
      will wait on, updating as new messages are
      made available.
    """
    pygame.init()

    # Initially, our surface is a None type; once we get
    # our first measurement, then we know the dimenions of the
    # camera, and can initialize our pygame surface appropriately
    pygameSurface: Optional[pygame.Surface] = None

    while True:
        queueResult: Optional[
            MonocularUncalibratedCameraMeasurement
        ] = msgQueue.listen(block=True, timeout=-1)
        if queueResult is None:
            raise ValueError("Unexpected NoneType from camera")

        nextCamMeasure: MonocularUncalibratedCameraMeasurement = (
            queueResult
        )

        # Extract the source image
        sourceImage: cv2.Mat = nextCamMeasure.image.sourceImgMat

        # If pygame surface is a NoneType, use the dimensions of the received
        # numpy array to initialize it
        if pygameSurface is None:
            # Opencv numpy matrices have shape (y, x, color channels), so build
            # from there
            xDim: int = sourceImage.shape[1]
            yDim: int = sourceImage.shape[0]

            pygameSurface = pygame.display.set_mode((xDim, yDim))

        # At this point, pygameSurface should be non-None; blit
        # the matrix onto the screen and flip it
        if pygameSurface is None:
            raise ValueError(
                "Expected pygameSurface to be initialized at this point; illegal state."
            )

        # Convert the image to bytes, then we can blit onto the pygame surface
        pygameImage: pygame.Surface = pygame.image.frombuffer(
            sourceImage.tobytes(),
            sourceImage.shape[1::-1],
            "BGR",
        )

        # Paint black, then blit and flip
        pygameSurface.fill((0, 0, 0))
        pygameSurface.blit(pygameImage, pygameSurface.get_rect())
        pygame.display.flip()


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
            cameraFName, cv2.COLOR_BGR2GRAY
        )
    )

    # Get a message queue to listen on
    msgQueue: MessageQueue[
        MonocularUncalibratedCameraMeasurement
    ] = cameraSensor.subscribe()

    # Spawn the pygame thread
    pygameThread: multiprocessing.Process = (
        multiprocessing.Process(
            target=pygameUpdater, args=(msgQueue,)
        )
    )
    pygameThread.start()

    asyncio.run(cameraSensor.captureLoop(60))


if __name__ == "__main__":
    run()
