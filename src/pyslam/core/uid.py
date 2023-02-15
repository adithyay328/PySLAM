from datetime import datetime
import secrets

def generateUID():
  # A simple UID generation function to generate UIDs that can be used
  # to uniquely identify camera matrices, etc.
  currentDateTimeString = datetime.utcnow().isoformat()
  randomString = secrets.token_hex(12)

  return randomString + currentDateTimeString