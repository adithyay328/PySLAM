from typing import Optional
import multiprocessing
from multiprocessing import synchronize

import pygame
from PIL.Image import Image
import cv2

from pyslam.pubsub.MessageQueue import MessageQueue
from pyslam.common_capture.UncalibratedMonocularCamera import (
    MonocularUncalibratedCameraMeasurement,
)
from pyslam.image_processing.cv_pillow import pillowToArray


class PyGameFrameWindow:
    """
    A class that wraps around a PyGame window and
    allows client code to visualize individual video
    frames easily. Reads frames from an Image publisher.

    :param msgQueue: The MessageQueue that we are going to
        listen to. We expect it to publish Images.
    """

    def __init__(self, msgQueue: MessageQueue[Image]):
        pygame.init()

        self.msgQueue: MessageQueue[Image] = msgQueue

        # Initially, our surface is a None type; once we get
        # our first measurement, then we know the dimenions of the
        # camera, and can initialize our pygame surface appropriately
        self.pygameSurface: Optional[pygame.Surface] = None

        # Lock object to allow access to loop variable
        self.lock: synchronize.Lock = multiprocessing.Lock()
        # Loop variable that lets us gracefully shut down loop
        self.looping: bool = True

        # Process that the loop runs in
        self.loopProcess: Optional[
            multiprocessing.Process
        ] = None

    def __captureLoop(self) -> None:
        """
        Internal capture loop that will be run in a different thread.
        """
        while True:
            with self.lock:
                if not self.looping:
                    # Close window, and break
                    pygame.display.quit()
                    pygame.quit()
                    break

            queueResult: Optional[Image] = self.msgQueue.listen(
                block=True, timeout=-1
            )
            if queueResult is None:
                raise ValueError(
                    "Unexpected NoneType from camera"
                )

            nextImage: Image = queueResult

            # Extract the source image
            sourceImage: cv2.Mat = pillowToArray(nextImage)

            # If pygame surface is a NoneType, use the dimensions of the received
            # numpy array to initialize it
            if self.pygameSurface is None:
                # Opencv numpy matrices have shape (y, x, color channels), so build
                # from there
                xDim: int = sourceImage.shape[1]
                yDim: int = sourceImage.shape[0]

                self.pygameSurface = pygame.display.set_mode(
                    (xDim, yDim)
                )

            # At this point, pygameSurface should be non-None; blit
            # the matrix onto the screen and flip it
            if self.pygameSurface is None:
                raise ValueError(
                    "Expected pygameSurface to be initialized at this point; illegal state."
                )

            # Convert the image to bytes, then we can blit onto the pygame surface
            pygameImage: pygame.Surface = (
                pygame.image.frombuffer(
                    sourceImage.tobytes(),
                    sourceImage.shape[1::-1],
                    "BGR",
                )
            )

            # Paint black, then blit and flip
            self.pygameSurface.fill((0, 0, 0))
            self.pygameSurface.blit(
                pygameImage, self.pygameSurface.get_rect()
            )
            pygame.display.flip()

    def startListenLoop(self):
        """
        Starts the interal listening loop.
        """
        with self.lock:
            if self.loopProcess is not None:
                raise ValueError("Loop is already running")
            else:
                self.loopProcess = multiprocessing.Process(
                    target=self.__captureLoop
                )
                self.loopProcess.start()

    def stopListenLoop(self):
        """
        Stops the internal listening loop.
        """
        with self.lock:
            if self.loopProcess is None:
                raise ValueError("Loop is not running")
            else:
                self.looping = False
                self.loopProcess.join()
                self.loopProcess = None
