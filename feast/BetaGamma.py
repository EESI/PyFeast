#!/usr/bin/env python
import numpy as np
import ctypes as c

from .util import check_data 

libFSToolbox = c.CDLL("libFSToolbox.so"); 

__author__ = ["Calvin Morrison", "Gregory Ditzler"]
__copyright__ = "Copyright 2014, EESI Laboratory"
__credits__ = ["Calvin Morrison", "Gregory Ditzler"]
__license__ = "GPL"
__version__ = "2.0.0"
__email__ = "mutantturkey@gmail.com"
__status__ = "Development"


class BetaGamma:
  """
      This algorithm implements conditional mutual information 
      feature select, such that beta and gamma control the 
      weight attached to the redundant mutual and conditional
      mutual information, respectively. 
  """
  
  def __init__(self, n_select, beta=1.0, gamma=1.0):
    """
    """
    self.n_select = n_select
    self.beta = beta
    self.gamma = gamma

  def fit(self, data, labels):
    """
      fit(self, data, labels)

        @param data: data in a Numpy array such that len(data) = 
          n_observations, and len(data.transpose()) = n_features
          (REQUIRED)
        @type data: ndarray
        @param labels: labels represented in a numpy list with 
          n_observations as the number of elements. That is 
          len(labels) = len(data) = n_observations.
          (REQUIRED)
        @type labels: ndarray
        @return: features in the order they were selected. 
        @rtype: list
    """
    data, labels = check_data(data, labels)

    # python values
    n_observations, n_features = data.shape
    output = np.zeros(self.n_select)

    # cast as C types
    c_n_observations = c.c_int(n_observations)
    c_n_select = c.c_int(self.n_select)
    c_n_features = c.c_int(n_features)
    c_beta = c.c_double(self.beta)
    c_gamma = c.c_double(self.gamma)

    libFSToolbox.BetaGamma.restype = c.POINTER(c.c_double * self.n_select)
    features = libFSToolbox.BetaGamma(c_n_select,
                                      c_n_observations,
                                      c_n_features, 
                                      data.ctypes.data_as(c.POINTER(c.c_double)),
                                      labels.ctypes.data_as(c.POINTER(c.c_double)),
                                      output.ctypes.data_as(c.POINTER(c.c_double)),
                                      c_beta,
                                      c_gamma
                                      )

    selected_features = []
    for i in features.contents:
      selected_features.append(i)
    return selected_features


