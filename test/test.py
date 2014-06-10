#!/usr/bin/env python 
from feast import *
import numpy as np
import csv


def check_result(selected_features, n_relevant):
	selected_features = sorted(selected_features)
	success = True
	for k in range(n_relevant):
		if k != selected_features[k]:
			success = False
	return success

def read_digits(fname='digit.txt'):
	'''
		read_digits(fname='digit.txt')

		read a data file that contains the features and class labels. 
		each row of the file is a feature vector with the class 
		label appended. 
	'''

	fw = csv.reader(open(fname,'rb'), delimiter='\t')
	data = []
	for line in fw: 
		data.append( [float(x) for x in line] )
	data = np.array(data)
	labels = data[:,len(data.transpose())-1]
	data = data[:,:len(data.transpose())-1]
	return data, labels

def uniform_data(n_observations = 1000, n_features = 50, n_relevant = 5):
	import numpy as np
	xmax = 10
	xmin = 0
	data = 1.0*np.random.randint(xmax + 1, size = (n_features, n_observations))
	labels = np.zeros(n_observations)
	delta = n_relevant * (xmax - xmin) / 2.0

	for m in range(n_observations):
		zz = 0.0
		for k in range(n_relevant):
			zz += data[k, m]
		if zz > delta:
			labels[m] = 1
		else:
			labels[m] = 2
	data = data.transpose()
	
	return data, labels





n_relevant = 5
data_source = 'uniform'    # set the data set we want to test


if data_source == 'uniform':
	data, labels = uniform_data(n_relevant = n_relevant)
elif data_source == 'digits':
	data, labels = read_digits('digit.txt')

n_observations = len(data)					# number of samples in the data set
n_features = len(data.transpose())	# number of features in the data set
n_select = 15												# how many features to select
method = 'MIM'											# feature selection algorithm


print '---> Information'
print '     :n_observations - ' + str(n_observations)
print '     :n_features     - ' + str(n_features)
print '     :n_select       - ' + str(n_select)
print '     :algorithm      - ' + str(method)
print ' '
print '---> Running unit tests on FEAST 4 Python... '


#################################################################
#################################################################
print '       Running BetaGamma... '
sf = BetaGamma(data, labels, n_select, beta=0.5, gamma=0.5)
if check_result(sf, n_relevant) == True:
	print '          BetaGamma passed!'
else:
	print '          BetaGamma failed!'


#################################################################
#################################################################
print '       Running CMIM... '
sf = CMIM(data, labels, n_select)
if check_result(sf, n_relevant) == True:
	print '          CMIM passed!'
else:
	print '          CMIM failed!'


#################################################################
#################################################################
print '       Running CondMI... '
sf = CondMI(data, labels, n_select)
if check_result(sf, n_relevant) == True:
	print '          CondMI passed!'
else:
	print '          CondMI failed!'


#################################################################
#################################################################
print '       Running DISR... '
sf = DISR(data, labels, n_select)
if check_result(sf, n_relevant) == True:
	print '          DISR passed!'
else:
	print '          DISR failed!'


#################################################################
#################################################################
print '       Running ICAP... '
sf = ICAP(data, labels, n_select)
if check_result(sf, n_relevant) == True:
	print '          ICAP passed!'
else:
	print '          ICAP failed!'


#################################################################
#################################################################
print '       Running JMI... '
sf = JMI(data, labels, n_select)
if check_result(sf, n_relevant) == True:
	print '          JMI passed!'
else:
	print '          JMI failed!'


#################################################################
#################################################################
print '       Running mRMR... '
sf = mRMR(data, labels, n_select)
if check_result(sf, n_relevant) == True:
	print '          mRMR passed!'
else:
	print '          mRMR failed!'

#################################################################
#################################################################
print '       Running MIM...'
sf = MIM(data, labels, n_select)
if check_result(sf, n_relevant) == True:
	print '          MIM passed!'
else:
	print '          MIM failed!'

p
print '---> Done unit tests!'




