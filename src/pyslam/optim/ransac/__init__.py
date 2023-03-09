"""
This sub-module implements logic to get rid
of a lot of the boiler plate logic for RANSAC,
and allows it to be automatically parralelized.

This comes after observing that by its nature,
separate RANSAC iterations have no inter-dependency,
and so should run in parralel as much as possible

Below is the pseudocode for a general RANSAC estimation
problem; hopefully it provides some insight into why the
types and class defined in this module are defined as they
are:

RANSAC:

#. Denote as "models" an array containing all models we determine
   in running RANSAC; set this to an empty array initially
#. Denote as "inliers" an array such that "inliers"[i] tells us how
   many inliers "models"[i] had; set this to an empty array
   initially
#. for i = 1 to N iterations:

   #. Pick a random sub-sample of the whole dataset of size N
   #. Fit the model in question to the sub-sample of the dataset
   #. Denote as "inlier_count" the number of inliers; set this to
      0 initially
   #. Denote as "inliers" an array or all inliers we found; set
      this to an empty array initially
   #. For j = 1 to D, where D is the number of datapoints in the
       entire dataset

      #. Compute some "error" between the model and the jth
         datapoint; denote this error as "err"
      #. Denote as "thresh" a value that discriminates between
         inliers and outliers; inliers have "err" < "thresh",
         while outliters have "err" >= "thresh"
      #. If "err" < "thresh", increment "inlier_count" by 1,
          and add the jth datapoint to "inliers"

   #. Add the model we just found to "models", and add our inlier
      count to the end of "inliers"

#. Denote "best" as the model with the largest number of inliers
#. Return "best", optionally re-fitting it to all datapoitns
 
"""

from .RANSACDataset import RANSACDataset
from .RANSACModel import RANSACModel
from .RANSACEstimator import RANSACEstimator
