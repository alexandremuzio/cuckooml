#!/usr/bin/python

import argparse
import os
import json
import sys
import logging

from sklearn.feature_extraction import DictVectorizer
from sklearn import cluster

from core import *

log = logging.getLogger('cuckoo_ml')
log.setLevel(logging.WARN)

def load_directory_files(dir):
	files = os.listdir(dir)

	reports_data = []
	for file in files:
		file_path = os.path.join(dir, file)

		with open(file_path) as report_file:
			report_data = json.load(report_file)
	    	reports_data.append(report_data)

	return reports_data

def main():
	parser = argparse.ArgumentParser(description='ClusterML demo.')
	parser.add_argument('-d', '--data-dir', default='data', type=str)
	args = parser.parse_args()

	data_files = load_directory_files(args.data_dir)

	behavioral_profiles = [BehavioralProfile(data_file) for data_file in data_files]

	# print behavioral_profiles[0].get_operations()[0:10]
	feature_extractor = FeatureExtractor(behavioral_profiles)
	vectorized_data = feature_extractor.get_vectorized_data()

	k_means = KMeans(vectorized_data)
	k_means.run_clustering()

	#TODO - verify results
	import numpy as np
	np.savetxt('results.txt', k_means.labels())


if __name__ == '__main__':
	print('CuckooML demo')
	main()