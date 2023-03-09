"""This module defines some helper utilities for
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

When they subscribe, they are simply given an instance of a 
MessageQueue, and can listen for new messages to get published 
messages.
"""
