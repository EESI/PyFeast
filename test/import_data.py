



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
	import numpy as np

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



##################################################################
##################################################################
##################################################################
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

##################################################################
##################################################################
##################################################################
