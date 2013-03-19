#!/usr/bin/env python 
from sklearn import datasets

digits = datasets.load_digits()   # load the data from scikits
data = digits.images.reshape((digits.images.shape[0], -1))
labels = digits.target  # extract the labels

fw = open('digit.txt', 'w')

for n in range(len(data)):
	mstr = ''
	for x in data[n]:
		mstr += str(x) + '\t'
	fw.write(mstr + str(labels[n]) + '\n')

fw.close()