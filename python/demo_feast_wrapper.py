#!/usr/bin/env python 
from feast import *
import numpy as np
import import_data 


def check_result(selected_features, n_select):
	selected_features = sorted(selected_features)
	success = True
	for k in range(n_select):
		if k != selected_features[k]:
			success = False
	return success




data_source = 'uniform'    # set the data set we want to test


if data_source == 'uniform':
	data, labels = import_data.uniform_data()
elif data_source == 'digits':
	data, labels = import_data.read_digits('digit.txt')

n_observations = len(data)					# number of samples in the data set
n_features = len(data.transpose())	# number of features in the data set
n_select = 15												# how many features to select
method = 'JMI'											# feature selection algorithm


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
if check_result(sf) == True:
	print '          BetaGamma passed!'
else:
	print '          BetaGamma failed!'


#################################################################
#################################################################
print '       Running CMIM... '
sf = CMIM(data, labels, n_select)
if check_result(sf) == True:
	print '          CMIM passed!'
else:
	print '          CMIM failed!'


#################################################################
#################################################################
print '       Running CondMI... '
sf = CondMI(data, labels, n_select)
if check_result(sf) == True:
	print '          CondMI passed!'
else:
	print '          CondMI failed!'


#################################################################
#################################################################
print '       Running DISR... '
sf = DISR(data, labels, n_select)
if check_result(sf) == True:
	print '          DISR passed!'
else:
	print '          DISR failed!'


#################################################################
#################################################################
print '       Running ICAP... '
sf = ICAP(data, labels, n_select)
if check_result(sf) == True:
	print '          ICAP passed!'
else:
	print '          ICAP failed!'


#################################################################
#################################################################
print '       Running JMI... '
sf = JMI(data, labels, n_select)
if check_result(sf) == True:
	print '          JMI passed!'
else:
	print '          JMI failed!'


#################################################################
#################################################################
print '       Running mRMR... '
sf = mRMR(data, labels, n_select)
if check_result(sf) == True:
	print '          mRMR passed!'
else:
	print '          mRMR failed!'

print '---> Done unit tests!'




