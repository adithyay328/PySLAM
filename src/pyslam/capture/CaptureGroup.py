class CaptureGroup:
    """A group of capture objects together. Just a type that wraps around a
    list effectively.
    """

    def __init__(
        self,
        listOfCaptures,
    ):
        self.listOfCaptures = listOfCaptures
