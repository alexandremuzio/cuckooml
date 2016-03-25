#!/usr/bin/python3

import argparse
import os
import json
import sys
import logging

from sklearn.feature_extraction import DictVectorizer
from sklearn import cluster

from core import *
from cluster import *

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

def cluster_with_os_operations(data_files):
	behavioral_profiles = [OSOperationsProfile(data_file) for data_file in data_files]

	feature_extractor = OSOperationsFeatureExtractor(behavioral_profiles)
	vectorized_data = feature_extractor.get_vectorized_data()
	reduced_data = feature_extractor.get_reduced_data()

	k_means = KMeans(reduced_data, 6)
	k_means.run_clustering()

	#plot data
	k_means.vizualize_results_in_2d(reduced_data)

	#Save results
	# import numpy as np
	# np.savetxt('results.txt', k_means.labels())


def cluster_with_api_calls(data_files):
	behavioral_profiles = [OSOperationsProfile(data_file, data_files) for data_file in data_files]

	feature_extractor = APIFeatureExtractor(behavioral_profiles)
	vectorized_data = feature_extractor.get_vectorized_data()
	reduced_data = feature_extractor.get_reduced_data()

	k_means = KMeans(reduced_data, 6)
	k_means.run_clustering()

	#plot data
	k_means.vizualize_results_in_2d(reduced_data)



def main():
	parser = argparse.ArgumentParser(description='ClusterML demo.')
	parser.add_argument('-d', '--data-dir', default='data', type=str)
	parser.add_argument('-m', '--method-type', default='os_operations')
	args = parser.parse_args()

	#Reports loaded in memory
	data_files = load_directory_files(args.data_dir)

	if (args.method_type == 'os_operations'):
		cluster_with_os_operations(data_files)

	else:
		cluster_with_api_calls(data_files)

if __name__ == '__main__':
	print('CuckooML demo')
	main()