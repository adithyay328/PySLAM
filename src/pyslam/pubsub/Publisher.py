from abc import ABC
from typing import (
    Generic,
    TypeVar,
)
import queue
import weakref
import multiprocessing

from pyslam.pubsub.MessageQueue import MessageQueue

T = TypeVar("T")


class Publisher(Generic[T]):
    """
    An abstract base class that can be subclassed
    by any class that wants to act as a publisher.
    Internally uses a multiprocessing Manager,
    which allows resultant MessageQueues to be used
    accross threads easily.
    """

    def __init__(self) -> None:
        # One tricky thing is that we can't use Queues as is;
        # since we need to pickle them, we need to return the proxy
        # objects that we get from a Manager, which are
        # pickleable
        self.__queueManager = multiprocessing.Manager()

        # Contains a weak reference to all proxy Message
        # Queues. When we want to publish, just iterate
        # over these and publish. This allows them to
        # be automatically garbage collected when no longer
        # needed
        self.__messageQueues: weakref.WeakSet[
            MessageQueue[T]
        ] = weakref.WeakSet()

    def subscribe(self) -> MessageQueue[T]:
        """
        Constructs and returns a message queue
        that downstream code can listen on for
        new messages. Maintains a weak-refernce
        internally, so as soon as downstream code loses
        the last strong reference it's garbage collected.

        :return: Returns a message queue that downstream code
          can listen on.
        """
        # Construct proxy object and our MessageQueue
        queueProxy: queue.Queue[T] = self.__queueManager.Queue()
        newQueue: MessageQueue[T] = MessageQueue[T](queueProxy)

        # Save a weak reference locally.
        self.__messageQueues.add(newQueue)

        return newQueue

    def publish(self, message: T) -> None:
        """
        Iterates over all message queues
        that haven't yet been garbage collected
        and publishes the message
        """
        for msgQueue in self.__messageQueues:
            msgQueue.publish(message)

        # At this point we're done; exit.
