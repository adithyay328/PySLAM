from typing import TypeVar, Generic, Optional, Any
import queue

T = TypeVar("T")


class MessageQueue(Generic[T]):
    """A MessageQueue with typing on received messages.
    Queues must be passed in, as they are expectd to be
    proxy objects managed by a Multiprocessing Manager. Without that,
    they can't be pickled, which restricts their use."""

    def __init__(self, queue: queue.Queue[T]) -> None:
        self.__queue = queue

    def publish(self, message: T) -> None:
        """
        Publish a message on the internal thread queue

        :param message: The message to publish on this queue.
        """

        self.__queue.put(message)

    def listen(
        self, block: bool, timeout: float = 0.0
    ) -> Optional[T]:
        """
        Listens on the internal message queue for a new Message.
        Can be blocking or non-blocking, based on whether

        :param block: Whether or not to block indefiniely
          while listening. If false and no message is on the queue, returns a non-type
        :param timeout: If block=True, limits the max number of seconds we will
          wait before returning None. If block=False, does nothing. Set to a
          negative number to wait indefinitely

        :return: Returns the first message we receive from the queue if
          block = True; if block = False, will return a message if one
          is on the queue, and a None type otherwise.
        """

        try:
            msg: T = self.__queue.get(
                block=block,
                timeout=(timeout if timeout > 0 else None),
            )

            return msg
        except queue.Empty:
            return None
