import numpy as np
from ctypes import * 

def select(data, labels, n_observations, n_features, n_select, method):
  
  selected_features = []

  try:
    libFSToolbox = CDLL("libFSToolbox.so"); 
  except:
    print "Error: could not find libFSToolbox"
    exit()

# JMI(n_features_to_ret, int n_samples, int n_feats, double *featureMatrix, double *classcol, outputFeatures);
  c_output = (c_double * n_select)
  c_n_observations = c_int(n_observations)
  c_n_select = c_int(n_select)
  c_n_features = c_int(n_features)
  c_labels = (c_double * len(labels))(*labels)

  combined_list = [] 
  map(combined_list.extend, data.tolist())
  c_data = (c_double * len(combined_list))(*combined_list)




  # right now just call only JMI, work out the rest later
  libFSToolbox.JMI(c_n_select, c_n_observations, c_n_features, repr(c_data), repr(c_labels), repr(c_output))

  return selected_features
