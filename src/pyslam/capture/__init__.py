import weakref
from typing import Union

from pyslam.uid import UID

# This WeakDict serves as a lookup for
# Measurements, MeasurementSources and Sensors.
# Allows global lookups, while not impeding
# garbage collection
GLOBAL_LOOKUP: weakref.WeakValueDictionary[
    UID, Union["Measurement", "MeasurementSource", "Sensor"]
] = weakref.WeakValueDictionary()


from pyslam.capture.Measurement import Measurement
from pyslam.capture.MeasurementSource import MeasurementSource
from pyslam.capture.Sensor import Sensor
