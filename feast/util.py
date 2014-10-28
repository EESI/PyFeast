#!/usr/bin/env python
import numpy as np

def check_data(data, labels):
  """
    Check dimensions of the data and the labels.  Raise and exception
    if there is a problem.

    Data and Labels are automatically cast as doubles before calling the 
    feature selection functions

    @param data: the data 
    @param labels: the labels
    @return (data, labels): ndarray of floats
    @rtype: tuple
  """

  if isinstance(data, np.ndarray) is False:
    raise Exception("data must be an numpy ndarray.")
  if isinstance(labels, np.ndarray) is False:
    raise Exception("labels must be an numpy ndarray.")

  if len(data) != len(labels):
    raise Exception("data and labels must be the same length")

  return 1.0*np.array(data, order="F"), 1.0*np.array(labels, order="F")
