from datetime import (
    datetime,
)
import secrets
from typing import Optional
import copy


class UID:
    """A class representing a unique identifier"""

    def __init__(self, uid: Optional[str] = None):
        self.__uid = ""

        # Set UID based on whether we got a UID or not
        if uid is not None:
            self.__uid += uid
        else:
            self.__uid = self.generateUID()

    @property
    def uid(self) -> str:
        return copy.copy(self.__uid)

    @staticmethod
    def generateUID():
        # A simple UID generation function to generate UIDs that can be used
        # to uniquely identify camera matrices, etc.
        currentDateTimeString = datetime.utcnow().isoformat()
        randomString = secrets.token_hex(12)

        return randomString + currentDateTimeString
