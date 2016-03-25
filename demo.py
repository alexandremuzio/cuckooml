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

IMAGES_DIR='images/'
OS_OPERATIONS_IMGS_DIR = IMAGES_DIR + 'os_operations'
API_CALLS_IMGS_DIR = IMAGES_DIR + 'api_calls'

def create_directories():
	try:
		os.makedirs(OS_OPERATIONS_IMGS_DIR)
	except:
		pass
	try:
		os.makedirs(API_CALLS_IMGS_DIR)
	except:
		pass

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

	for i in range(2, 8):
		k_means = KMeans(reduced_data, i)
		k_means.run_clustering()

		#plot data
		k_means.vizualize_results_in_2d(reduced_data, no_display=True, img_name='images/os_operations/kmeans_'+ str(i))
		k_means.plot_silhouette(reduced_data, no_display=True, img_name='images/os_operations/kmeans_sil_'+ str(i))

	#Save results
	# import numpy as np
	# np.savetxt('results.txt', k_means.labels())


def cluster_with_api_calls(data_files):
	behavioral_profiles = [APIProfile(data_file) for data_file in data_files]

	feature_extractor = APIFeatureExtractor(behavioral_profiles, data_files)
	vectorized_data = feature_extractor.get_vectorized_data()
	reduced_data = feature_extractor.get_reduced_data()

	for i in range(2, 8):
		k_means = KMeans(reduced_data, i)
		k_means.run_clustering()

		#plot data
		k_means.vizualize_results_in_2d(reduced_data, no_display=True, img_name='images/api_calls/kmeans_'+ str(i))
		k_means.plot_silhouette(reduced_data, no_display=True, img_name='images/api_calls/kmeans_sil_'+ str(i))


def main():
	create_directories()
	parser = argparse.ArgumentParser(description='ClusterML demo.')
	parser.add_argument('-d', '--data-dir', default='data', type=str)
	parser.add_argument('-m', '--method-type', default='os_operations')
	args = parser.parse_args()

	#Reports loaded in memory
	data_files = load_directory_files(args.data_dir)

	if (args.method_type == 'os_operations'):
		cluster_with_os_operations(data_files)

	elif (args.method_type == 'api_calls'):
		cluster_with_api_calls(data_files)

	else:
		print ('Doing nothing. Choose a correct method (os_operations or api_calls')

if __name__ == '__main__':
	print('CuckooML demo')
	main()