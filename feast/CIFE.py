#!/usr/bin/env python
import numpy as np
import ctypes as c

from .BetaGamma import BetaGamma
from .util import check_data

libFSToolbox = c.CDLL("libFSToolbox.so"); 

__author__ = ["Calvin Morrison", "Gregory Ditzler"]
__copyright__ = "Copyright 2014, EESI Laboratory"
__credits__ = ["Calvin Morrison", "Gregory Ditzler"]
__license__ = "GPL"
__version__ = "2.0.0"
__email__ = "mutantturkey@gmail.com"
__status__ = "Development"


class CIFE:
  """
    This function implements the Condred feature selection algorithm.
    beta = 1; gamma = 1;
  """
  def __init__(self, n_select):
    self.n_select = n_select

  def fit(self, data, labels):
    """
      fit(self, data, labels)

      @param data: A Numpy array such that len(data) = 
          n_observations, and len(data.transpose()) = n_features
      @type data: ndarray
      @param labels: labels represented in a numpy list with 
          n_observations as the number of elements. That is 
          len(labels) = len(data) = n_observations.
      @type labels: ndarray
      @return selected_features: features in the order they were selected. 
      @rtype: list
    """
    return BetaGamma(data, labels, self.n_select, beta=1.0, gamma=1.0)


