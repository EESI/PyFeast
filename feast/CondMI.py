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


class CondMI:
  """
    This function implements the conditional mutual information
    maximization feature selection algorithm. 
  """
  def __init__(self, n_select):
    self.n_select = n_select

  def fit(self, data, labels):
    """

      @param data: data in a Numpy array such that len(data) = n_observations,
        and len(data.transpose()) = n_features
      @type data: ndarray
      @param labels: represented in a numpy list with 
        n_observations as the number of elements. That is 
        len(labels) = len(data) = n_observations.
      @type labels: ndarray
      @param n_select: number of features to select.
      @type n_select: integer
      @return: features in the order they were selected. 
      @rtype list
    """
    data, labels = check_data(data, labels)

    # python values
    n_observations, n_features = data.shape
    output = np.zeros(self.n_select)

    # cast as C types
    c_n_observations = c.c_int(n_observations)
    c_n_select = c.c_int(self.n_select)
    c_n_features = c.c_int(n_features)

    libFSToolbox.CondMI.restype = c.POINTER(c.c_double * self.n_select)
    features = libFSToolbox.CondMI(c_n_select,
                                   c_n_observations,
                                   c_n_features, 
                                   data.ctypes.data_as(c.POINTER(c.c_double)),
                                   labels.ctypes.data_as(c.POINTER(c.c_double)),
                                   output.ctypes.data_as(c.POINTER(c.c_double))
                                   )
  
    selected_features = []
    for i in features.contents:
      selected_features.append(i)

    return selected_features

