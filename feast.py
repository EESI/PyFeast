'''
  The FEAST module provides an interface between the C-library
  for feature selection to Python. 

  References: 
  1) G. Brown, A. Pocock, M.-J. Zhao, and M. Lujan, "Conditional
      likelihood maximization: A unifying framework for information
      theoretic feature selection," Journal of Machine Learning 
      Research, vol. 13, pp. 27-66, 2012.

'''
__author__ = "Calvin Morrison"
__copyright__ = "Copyright 2013, EESI Laboratory"
__credits__ = ["Calvin Morrison", "Gregory Ditzler"]
__license__ = "GPL"
__version__ = "0.2.0"
__maintainer__ = "Calvin Morrison"
__email__ = "mutantturkey@gmail.com"
__status__ = "Release"

import numpy as np
import ctypes as c

try:
  libFSToolbox = c.CDLL("libFSToolbox.so"); 
except:
  raise Exception("Error: could not load libFSToolbox.so")


def BetaGamma(data, labels, n_select, beta=1.0, gamma=1.0):
  '''
    BetaGamma(data, labels, n_select, beta=1.0, gamma=1.0)

    This algorithm implements conditional mutual information 
    feature select, such that beta and gamma control the 
    weight attached to the redundant mutual and conditional
    mutual information, respectively. 

    Input 
      :data - data in a Numpy array such that len(data) = 
        n_observations, and len(data.transpose()) = n_features
        (REQUIRED)
      :labels - labels represented in a numpy list with 
        n_observations as the number of elements. That is 
        len(labels) = len(data) = n_observations.
        (REQUIRED)
      :n_select - number of features to select. (REQUIRED)
      :beta - penalty attacted to I(X_j;X_k) 
      :gamma - positive weight attached to the conditional
        redundancy term I(X_k;X_j|Y)
    Output 
      :selected_features - returns a list containing the features
        in the order they were selected. 
  '''
  data, labels = check_data(data, labels)

  # python values
  n_observations, n_features = data.shape
  output = np.zeros(n_select)

  # cast as C types
  c_n_observations = c.c_int(n_observations)
  c_n_select = c.c_int(n_select)
  c_n_features = c.c_int(n_features)
  c_beta = c.c_double(beta)
  c_gamma = c.c_double(gamma)

  libFSToolbox.BetaGamma.restype = c.POINTER(c.c_double * n_select)
  features = libFSToolbox.BetaGamma(c_n_select,
                   c_n_observations,
                   c_n_features, 
                   data.ctypes.data_as(c.POINTER(c.c_double)),
                   labels.ctypes.data_as(c.POINTER(c.c_double)),
                   output.ctypes.data_as(c.POINTER(c.c_double)),
                   c_beta,
                   c_gamma
                   )

  # turn our output into a list
  selected_features = []
  for i in features.contents:
    # recall that feast was implemented with Matlab in mind, so the 
    # authors assumed the indexing started a one; however, in Python 
    # the indexing starts at zero. 
    selected_features.append(i - 1)

  return selected_features



def CIFE(data, labels, n_select):
  '''
    CIFE(data, labels, n_select)

    This function implements the Condred feature selection algorithm.
    beta = 1; gamma = 1;

    Input 
      :data - data in a Numpy array such that len(data) = 
        n_observations, and len(data.transpose()) = n_features
        (REQUIRED)
      :labels - labels represented in a numpy list with 
        n_observations as the number of elements. That is 
        len(labels) = len(data) = n_observations.
        (REQUIRED)
      :n_select - number of features to select. (REQUIRED)
    Output 
      :selected_features - returns a list containing the features
        in the order they were selected. 
  '''

  return BetaGamma(data, labels, n_select, beta=1.0, gamma=1.0)




def CMIM(data, labels, n_select):
  '''
    CMIM(data, labels, n_select)

    This function implements the conditional mutual information
    maximization feature selection algorithm. Note that this 
    implementation does not allow for the weighting of the 
    redundancy terms that BetaGamma will allow you to do.

    Input 
      :data - data in a Numpy array such that len(data) = 
        n_observations, and len(data.transpose()) = n_features
        (REQUIRED)
      :labels - labels represented in a numpy list with 
        n_observations as the number of elements. That is 
        len(labels) = len(data) = n_observations.
        (REQUIRED)
      :n_select - number of features to select. (REQUIRED)
    Output 
      :selected_features - returns a list containing the features
        in the order they were selected. 
  '''
  data, labels = check_data(data, labels)

  # python values
  n_observations, n_features = data.shape
  output = np.zeros(n_select)

  # cast as C types
  c_n_observations = c.c_int(n_observations)
  c_n_select = c.c_int(n_select)
  c_n_features = c.c_int(n_features)

  libFSToolbox.CMIM.restype = c.POINTER(c.c_double * n_select)
  features = libFSToolbox.CMIM(c_n_select,
                   c_n_observations,
                   c_n_features, 
                   data.ctypes.data_as(c.POINTER(c.c_double)),
                   labels.ctypes.data_as(c.POINTER(c.c_double)),
                   output.ctypes.data_as(c.POINTER(c.c_double))
                   )

  
  # turn our output into a list
  selected_features = []
  for i in features.contents:
    # recall that feast was implemented with Matlab in mind, so the 
    # authors assumed the indexing started a one; however, in Python 
    # the indexing starts at zero. 
    selected_features.append(i - 1)

  return selected_features



def CondMI(data, labels, n_select):
  '''
    CondMI(data, labels, n_select)

    This function implements the conditional mutual information
    maximization feature selection algorithm. 

    Input 
      :data - data in a Numpy array such that len(data) = 
        n_observations, and len(data.transpose()) = n_features
        (REQUIRED)
      :labels - labels represented in a numpy list with 
        n_observations as the number of elements. That is 
        len(labels) = len(data) = n_observations.
        (REQUIRED)
      :n_select - number of features to select. (REQUIRED)
    Output 
      :selected_features - returns a list containing the features
        in the order they were selected. 
  '''
  data, labels = check_data(data, labels)

  # python values
  n_observations, n_features = data.shape
  output = np.zeros(n_select)

  # cast as C types
  c_n_observations = c.c_int(n_observations)
  c_n_select = c.c_int(n_select)
  c_n_features = c.c_int(n_features)

  libFSToolbox.CondMI.restype = c.POINTER(c.c_double * n_select)
  features = libFSToolbox.CondMI(c_n_select,
                   c_n_observations,
                   c_n_features, 
                   data.ctypes.data_as(c.POINTER(c.c_double)),
                   labels.ctypes.data_as(c.POINTER(c.c_double)),
                   output.ctypes.data_as(c.POINTER(c.c_double))
                   )

  
  # turn our output into a list
  selected_features = []
  for i in features.contents:
    # recall that feast was implemented with Matlab in mind, so the 
    # authors assumed the indexing started a one; however, in Python 
    # the indexing starts at zero. 
    selected_features.append(i - 1)

  return selected_features


def Condred(data, labels, n_select):
  '''
    Condred(data, labels, n_select)

    This function implements the Condred feature selection algorithm.
    beta = 0; gamma = 1;

    Input 
      :data - data in a Numpy array such that len(data) = 
        n_observations, and len(data.transpose()) = n_features
        (REQUIRED)
      :labels - labels represented in a numpy list with 
        n_observations as the number of elements. That is 
        len(labels) = len(data) = n_observations.
        (REQUIRED)
      :n_select - number of features to select. (REQUIRED)
    Output 
      :selected_features - returns a list containing the features
        in the order they were selected. 
  '''
  data, labels = check_data(data, labels)

  return BetaGamma(data, labels, n_select, beta=0.0, gamma=1.0)



def DISR(data, labels, n_select):
  '''
    DISR(data, labels, n_select)

    This function implements the double input symmetrical relevance
    feature selection algorithm. 

    Input 
      :data - data in a Numpy array such that len(data) = 
        n_observations, and len(data.transpose()) = n_features
        (REQUIRED)
      :labels - labels represented in a numpy list with 
        n_observations as the number of elements. That is 
        len(labels) = len(data) = n_observations.
        (REQUIRED)
      :n_select - number of features to select. (REQUIRED)
    Output 
      :selected_features - returns a list containing the features
        in the order they were selected. 
  '''
  data, labels = check_data(data, labels)

  # python values
  n_observations, n_features = data.shape
  output = np.zeros(n_select)

  # cast as C types
  c_n_observations = c.c_int(n_observations)
  c_n_select = c.c_int(n_select)
  c_n_features = c.c_int(n_features)

  libFSToolbox.DISR.restype = c.POINTER(c.c_double * n_select)
  features = libFSToolbox.DISR(c_n_select,
                   c_n_observations,
                   c_n_features, 
                   data.ctypes.data_as(c.POINTER(c.c_double)),
                   labels.ctypes.data_as(c.POINTER(c.c_double)),
                   output.ctypes.data_as(c.POINTER(c.c_double))
                   )

  
  # turn our output into a list
  selected_features = []
  for i in features.contents:
    # recall that feast was implemented with Matlab in mind, so the 
    # authors assumed the indexing started a one; however, in Python 
    # the indexing starts at zero. 
    selected_features.append(i - 1)

  return selected_features




def ICAP(data, labels, n_select):
  '''
    ICAP(data, labels, n_select)

    This function implements the interaction capping feature 
    selection algorithm. 

    Input 
      :data - data in a Numpy array such that len(data) = 
        n_observations, and len(data.transpose()) = n_features
        (REQUIRED)
      :labels - labels represented in a numpy list with 
        n_observations as the number of elements. That is 
        len(labels) = len(data) = n_observations.
        (REQUIRED)
      :n_select - number of features to select. (REQUIRED)
    Output 
      :selected_features - returns a list containing the features
        in the order they were selected. 
  '''
  data, labels = check_data(data, labels)

  # python values
  n_observations, n_features = data.shape
  output = np.zeros(n_select)

  # cast as C types
  c_n_observations = c.c_int(n_observations)
  c_n_select = c.c_int(n_select)
  c_n_features = c.c_int(n_features)

  libFSToolbox.ICAP.restype = c.POINTER(c.c_double * n_select)
  features = libFSToolbox.ICAP(c_n_select,
                   c_n_observations,
                   c_n_features, 
                   data.ctypes.data_as(c.POINTER(c.c_double)),
                   labels.ctypes.data_as(c.POINTER(c.c_double)),
                   output.ctypes.data_as(c.POINTER(c.c_double))
                   )

  
  # turn our output into a list
  selected_features = []
  for i in features.contents:
    # recall that feast was implemented with Matlab in mind, so the 
    # authors assumed the indexing started a one; however, in Python 
    # the indexing starts at zero. 
    selected_features.append(i - 1)

  return selected_features





def JMI(data, labels, n_select):
  '''
    JMI(data, labels, n_select)

    This function implements the joint mutual information feature
    selection algorithm. 

    Input 
      :data - data in a Numpy array such that len(data) = 
        n_observations, and len(data.transpose()) = n_features
        (REQUIRED)
      :labels - labels represented in a numpy list with 
        n_observations as the number of elements. That is 
        len(labels) = len(data) = n_observations.
        (REQUIRED)
      :n_select - number of features to select. (REQUIRED)
    Output 
      :selected_features - returns a list containing the features
        in the order they were selected. 
  '''
  data, labels = check_data(data, labels)

  # python values
  n_observations, n_features = data.shape
  output = np.zeros(n_select)

  # cast as C types
  c_n_observations = c.c_int(n_observations)
  c_n_select = c.c_int(n_select)
  c_n_features = c.c_int(n_features)

  libFSToolbox.JMI.restype = c.POINTER(c.c_double * n_select)
  features = libFSToolbox.JMI(c_n_select,
                   c_n_observations,
                   c_n_features, 
                   data.ctypes.data_as(c.POINTER(c.c_double)),
                   labels.ctypes.data_as(c.POINTER(c.c_double)),
                   output.ctypes.data_as(c.POINTER(c.c_double))
                   )

  
  # turn our output into a list
  selected_features = []
  for i in features.contents:
    # recall that feast was implemented with Matlab in mind, so the 
    # authors assumed the indexing started a one; however, in Python 
    # the indexing starts at zero. 
    selected_features.append(i - 1)

  return selected_features



def MIFS(data, labels, n_select):
  '''
    MIFS(data, labels, n_select)

    This function implements the MIFS algorithm.
    beta = 1; gamma = 0;

    Input 
      :data - data in a Numpy array such that len(data) = 
        n_observations, and len(data.transpose()) = n_features
        (REQUIRED)
      :labels - labels represented in a numpy list with 
        n_observations as the number of elements. That is 
        len(labels) = len(data) = n_observations.
        (REQUIRED)
      :n_select - number of features to select. (REQUIRED)
    Output 
      :selected_features - returns a list containing the features
        in the order they were selected. 
  '''

  return BetaGamma(data, labels, n_select, beta=0.0, gamma=0.0)


def MIM(data, labels, n_select):
  '''
    MIM(data, labels, n_select)

    This function implements the MIM algorithm.
    beta = 0; gamma = 0;

    Input 
      :data - data in a Numpy array such that len(data) = 
        n_observations, and len(data.transpose()) = n_features
        (REQUIRED)
      :labels - labels represented in a numpy list with 
        n_observations as the number of elements. That is 
        len(labels) = len(data) = n_observations.
        (REQUIRED)
      :n_select - number of features to select. (REQUIRED)
    Output 
      :selected_features - returns a list containing the features
        in the order they were selected. 
  '''
  data, labels = check_data(data, labels)

  return BetaGamma(data, labels, n_select, beta=0.0, gamma=0.0)



def mRMR(data, labels, n_select):
  '''
    mRMR(data, labels, n_select)

    This funciton implements the max-relevance min-redundancy feature
    selection algorithm. 

    Input 
      :data - data in a Numpy array such that len(data) = 
        n_observations, and len(data.transpose()) = n_features
        (REQUIRED)
      :labels - labels represented in a numpy list with 
        n_observations as the number of elements. That is 
        len(labels) = len(data) = n_observations.
        (REQUIRED)
      :n_select - number of features to select. (REQUIRED)
    Output 
      :selected_features - returns a list containing the features
        in the order they were selected. 
  '''
  data, labels = check_data(data, labels)

  # python values
  n_observations, n_features = data.shape
  output = np.zeros(n_select)

  # cast as C types
  c_n_observations = c.c_int(n_observations)
  c_n_select = c.c_int(n_select)
  c_n_features = c.c_int(n_features)

  libFSToolbox.mRMR_D.restype = c.POINTER(c.c_double * n_select)
  features = libFSToolbox.mRMR_D(c_n_select,
                   c_n_observations,
                   c_n_features, 
                   data.ctypes.data_as(c.POINTER(c.c_double)),
                   labels.ctypes.data_as(c.POINTER(c.c_double)),
                   output.ctypes.data_as(c.POINTER(c.c_double))
                   )

  
  # turn our output into a list
  selected_features = []
  for i in features.contents:
    # recall that feast was implemented with Matlab in mind, so the 
    # authors assumed the indexing started a one; however, in Python 
    # the indexing starts at zero. 
    selected_features.append(i - 1)

  return selected_features

def check_data(data, labels):
  '''
    check_data(data, labels)

    Check dimensions of the data and the labels.  Raise and exception
    if there is a problem.

    Data and Labels are automatically cast as doubles before calling the 
    feature selection functions

    Input
      :data
      :labels
    Output
      :data
      :labels
  '''

  if isinstance(data, np.ndarray) is False:
    raise Exception("data must be an numpy ndarray.")
  if isinstance(labels, np.ndarray) is False:
    raise Exception("labels must be an numpy ndarray.")

  if len(data) != len(labels):
    raise Exception("data and labels must be the same length")
  
  return 1.0*data, 1.0*labels
