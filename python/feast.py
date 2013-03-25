import numpy as np
from ctypes import * 

try:
  libFSToolbox = CDLL("libFSToolbox.so"); 
except:
  print "Error: could not find libFSToolbox"
  exit()



def BetaGamma(data, labels, n_select, beta=2.0, gamma=2.0):

  # python values
  n_observations, n_features = data.shape
  output = np.zeros(n_select)

  # cast as C types
  c_n_observations = c_int(n_observations)
  c_n_select = c_int(n_select)
  c_n_features = c_int(n_features)
  c_beta = c_double(beta)
  c_gamma = c_double(gamma)

  libFSToolbox.BetaGamma.restype = POINTER(c_double * n_select)
  features = libFSToolbox.BetaGamma(c_n_select,
                   c_n_observations,
                   c_n_features, 
                   data.ctypes.data_as(POINTER(c_double)),
                   labels.ctypes.data_as(POINTER(c_double)),
                   output.ctypes.data_as(POINTER(c_double)),
                   c_beta,
                   c_gamma
                   )

  # turn our output into a list
  selected_features = []
  for i in features.contents:
    selected_features.append(i - 1)

  return selected_features

def JMI(data, labels, n_select):

  # python values
  n_observations, n_features = data.shape
  output = np.zeros(n_select)

  # cast as C types
  c_n_observations = c_int(n_observations)
  c_n_select = c_int(n_select)
  c_n_features = c_int(n_features)

  libFSToolbox.JMI.restype = POINTER(c_double * n_select)
  features = libFSToolbox.JMI(c_n_select,
                   c_n_observations,
                   c_n_features, 
                   data.ctypes.data_as(POINTER(c_double)),
                   labels.ctypes.data_as(POINTER(c_double)),
                   output.ctypes.data_as(POINTER(c_double))
                   )

  
  # turn our output into a list
  selected_features = []
  for i in features.contents:
    selected_features.append(i - 1)

  return selected_features

def mRMR_D(data, labels, n_select):

  # python values
  n_observations, n_features = data.shape
  output = np.zeros(n_select)

  # cast as C types
  c_n_observations = c_int(n_observations)
  c_n_select = c_int(n_select)
  c_n_features = c_int(n_features)

  libFSToolbox.mRMR_D.restype = POINTER(c_double * n_select)
  features = libFSToolbox.mRMR_D(c_n_select,
                   c_n_observations,
                   c_n_features, 
                   data.ctypes.data_as(POINTER(c_double)),
                   labels.ctypes.data_as(POINTER(c_double)),
                   output.ctypes.data_as(POINTER(c_double))
                   )

  
  # turn our output into a list
  selected_features = []
  for i in features.contents:
    selected_features.append(i - 1)

  return selected_features

def CMIM(data, labels, n_select):

  # python values
  n_observations, n_features = data.shape
  output = np.zeros(n_select)

  # cast as C types
  c_n_observations = c_int(n_observations)
  c_n_select = c_int(n_select)
  c_n_features = c_int(n_features)

  libFSToolbox.CMIM.restype = POINTER(c_double * n_select)
  features = libFSToolbox.CMIM(c_n_select,
                   c_n_observations,
                   c_n_features, 
                   data.ctypes.data_as(POINTER(c_double)),
                   labels.ctypes.data_as(POINTER(c_double)),
                   output.ctypes.data_as(POINTER(c_double))
                   )

  
  # turn our output into a list
  selected_features = []
  for i in features.contents:
    selected_features.append(i - 1)

  return selected_features

def DISR(data, labels, n_select):

  # python values
  n_observations, n_features = data.shape
  output = np.zeros(n_select)

  # cast as C types
  c_n_observations = c_int(n_observations)
  c_n_select = c_int(n_select)
  c_n_features = c_int(n_features)

  libFSToolbox.DISR.restype = POINTER(c_double * n_select)
  features = libFSToolbox.DISR(c_n_select,
                   c_n_observations,
                   c_n_features, 
                   data.ctypes.data_as(POINTER(c_double)),
                   labels.ctypes.data_as(POINTER(c_double)),
                   output.ctypes.data_as(POINTER(c_double))
                   )

  
  # turn our output into a list
  selected_features = []
  for i in features.contents:
    selected_features.append(i - 1)

  return selected_features

def ICAP(data, labels, n_select):

  # python values
  n_observations, n_features = data.shape
  output = np.zeros(n_select)

  # cast as C types
  c_n_observations = c_int(n_observations)
  c_n_select = c_int(n_select)
  c_n_features = c_int(n_features)

  libFSToolbox.ICAP.restype = POINTER(c_double * n_select)
  features = libFSToolbox.ICAP(c_n_select,
                   c_n_observations,
                   c_n_features, 
                   data.ctypes.data_as(POINTER(c_double)),
                   labels.ctypes.data_as(POINTER(c_double)),
                   output.ctypes.data_as(POINTER(c_double))
                   )

  
  # turn our output into a list
  selected_features = []
  for i in features.contents:
    selected_features.append(i - 1)

  return selected_features

def CondMI(data, labels, n_select):

  # python values
  n_observations, n_features = data.shape
  output = np.zeros(n_select)

  # cast as C types
  c_n_observations = c_int(n_observations)
  c_n_select = c_int(n_select)
  c_n_features = c_int(n_features)

  libFSToolbox.CondMI.restype = POINTER(c_double * n_select)
  features = libFSToolbox.CondMI(c_n_select,
                   c_n_observations,
                   c_n_features, 
                   data.ctypes.data_as(POINTER(c_double)),
                   labels.ctypes.data_as(POINTER(c_double)),
                   output.ctypes.data_as(POINTER(c_double))
                   )

  
  # turn our output into a list
  selected_features = []
  for i in features.contents:
    selected_features.append(i - 1)

  return selected_features
