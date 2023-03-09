from abc import ABC
from typing import Optional
from copy import copy
from datetime import datetime, timezone

from pyslam.uid import UID
import pyslam.capture as capture


class Measurement(ABC):
    """
    A generic base class that represents some kind of
    capture/measurement from a sensor.

    :param uid: A UID object that uniquely identifies this measurement.
    :param sourceUID: The UID of the MeasurementSource
    :param timestamp: A datetime object indicating the time when this
      measurement was taken. Timezones are always in UTC.
    """

    def __init__(
        self,
        uid: Optional[UID],
        sourceUID: UID,
        timestamp: Optional[datetime] = None,
    ) -> None:
        # Set initial values
        self.__uid: UID = UID()
        self.__timestamp: datetime = datetime.now(
            tz=timezone.utc
        )
        self.__sourceUID: UID = sourceUID

        # Update any attributes that are passed in via optionals
        if uid is not None:
            self.__uid = uid
        if timestamp is not None:
            self.__timestamp = timestamp

        # Register self with the lookup
        capture.GLOBAL_LOOKUP[self.__uid] = self

    @property
    def uid(self) -> UID:
        """Returns a copy of the UID for this measurement"""
        return copy(self.__uid)

    @property
    def timestamp(self) -> datetime:
        """Returns a copy of the timestamp of this measurement"""
        return copy(self.__timestamp)

    @property
    def sourceUID(self) -> UID:
        """
        Returns a reference to the MeasurementSource that this measurement
        came from
        """
        return copy(self.__sourceUID)
