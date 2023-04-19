"""
This module defines some helper utilities for
building code that operates in a pub/sub model;
this seems to be a common pattern in SLAM systems,
since processing pipelines usually start in-response
to a sensor getting a measurement, and then each
subsequent step of the pipeline runs only in response
to the previous step completing. This can be cleanly
implemented as a series of publishers and subscribers.

The way this is implemented in this module is really simple.
Publishers indicate what kind of message they publish,
and subscribers can chose to subscribe. 

When client code subscribes, they are simply given
an instance of a MessageQueue, and can listen for
new messages to get published messages.

One more piece of this module, which is not a standard
part of the pub/sub design pattern, is the "Reactor"
class; this is an interface that allows us to easily
respond to incoming Messages as a listener, while also
publishing its results as a Publisher(a Reactor is both
a publisher and a listener).

This is something that would be really useful for
intermediate parts of the pipeline, where we want
to wait for a message to come in, do some processing,
and then publish new results in response.

For example, this would be useful for the feature detection
portion of a V-SLAM pipeline; the reactor would listen for new
camera capture Measurements, then run its feature detection
on the given images, and then publish its detected Features
as a Publisher. Then downstream tasks can also act as reactors,
building a full pipeline like that.
"""

from typing import TypeVar, Generic, Optional
import queue
import weakref
import multiprocessing
import copy

# Defining 2 types; one used for
# types decided by the type of
# incoming message, and one for outgoing
INCOMING_T = TypeVar("INCOMING_T")
OUTGOING_T = TypeVar("OUTGOING_T")


class MessageQueue(Generic[INCOMING_T]):
    """A MessageQueue with typing on received messages.
    Queues must be passed in, as they are expectd to be
    proxy objects managed by a Multiprocessing Manager. Without that,
    they can't be pickled, which restricts their use."""

    def __init__(self, queue: queue.Queue[INCOMING_T]) -> None:
        self.__queue = queue

    def publish(self, message: INCOMING_T) -> None:
        """
        Publish a message on the internal thread queue

        :param message: The message to publish on this queue.
        """

        self.__queue.put(message)

    def listen(
        self, block: bool, timeout: float = 0.0
    ) -> Optional[INCOMING_T]:
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
            msg: INCOMING_T = self.__queue.get(
                block=block,
                timeout=(timeout if timeout > 0 else None),
            )

            return msg
        except queue.Empty:
            return None


class Publisher(Generic[OUTGOING_T]):
    """
    A base class that can be subclassed
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
            MessageQueue[OUTGOING_T]
        ] = weakref.WeakSet()

    def subscribe(self) -> MessageQueue[OUTGOING_T]:
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
        queueProxy: queue.Queue[
            OUTGOING_T
        ] = self.__queueManager.Queue()
        newQueue: MessageQueue[OUTGOING_T] = MessageQueue[
            OUTGOING_T
        ](queueProxy)

        # Save a weak reference locally.
        self.__messageQueues.add(newQueue)

        return newQueue

    def publish(self, message: OUTGOING_T) -> None:
        """
        Iterates over all message queues
        that haven't yet been garbage collected
        and publishes the message
        """
        for msgQueue in self.__messageQueues:
            msgQueue.publish(copy.deepcopy(message))

        # At this point we're done; exit.
