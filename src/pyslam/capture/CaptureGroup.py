class CaptureGroup:
    """A group of capture object UIDs together. Just a type that wraps around a
    list effectively.
    """

    def __init__(
        self,
        listOfCaptureUIDs,
    ):
        self.listOfCaptureUIDs = listOfCaptureUIDs
