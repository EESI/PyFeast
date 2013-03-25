#!/usr/bin/env python 
import feast
import numpy as np
import import_data 


print '---> Loading digit data'

data_source = 'uniform'


if data_source == 'uniform':
	data, labels = import_data.uniform_data()
elif data_source == 'digits':
	data, labels = import_data.read_digits('digit.txt')

print data


n_observations = len(data)					# number of samples in the data set
n_features = len(data.transpose())	# number of features in the data set
n_select = 15												# how many features to select
method = 'JMI'											# feature selection algorithm


print '---> Information'
print '     :n_observations - ' + str(n_observations)
print '     :n_features     - ' + str(n_features)
print '     :n_select       - ' + str(n_select)
print '     :algorithm      - ' + str(method)

selected_features = feast.JMI(data, labels, n_select)

print selected_features
