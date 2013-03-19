#!/usr/bin/env python 

import numpy as np

##################################################################
##################################################################
##################################################################
def read_digits(fname='digit.txt'):
	'''
		read_digits(fname='digit.txt')

		read a data file that contains the features and class labels. 
		each row of the file is a feature vector with the class 
		label appended. 
	'''
	import csv

	fw = csv.reader(open(fname,'rb'), delimiter='\t')
	data = []
	for line in fw: 
		data.append( [float(x) for x in line] )
	data = np.array(data)
	labels = data[:,len(data.transpose())-1]
	data = data[:,:len(data.transpose())-1]
	return data, labels
##################################################################
##################################################################
##################################################################



print '---> Loading digit data'
data, labels = read_digits('digit.txt')
n_observations = len(data)					# number of samples in the data set
n_features = len(data.transpose())	# number of features in the data set
n_select = 15												# how many features to select
method = 'jmi'											# feature selection algorithm


print '---> Information'
print '     :n_observations - ' + str(n_observations)
print '     :n_features     - ' + str(n_features)
print '     :n_select       - ' + str(n_select)
print '     :algorithm      - ' + str(method)


# Calvin: feature selection wrapper goes here
#selected_features = feast(data, labels, n_observations, n_features, n_select, \
#	method)