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

def access_tree_elem(tree ,feature_name):
	tree_elem = tree

	args = feature_name.split('_')
	for arg in args:
		tree_elem = tree_elem[arg]

	# print tree_elem
	return tree_elem

def extract_features(data_files):
	features_names = ['info_score', 'info_duration']

	features_list = []
	for data_file in data_files:
		feature_i = {}
		for feature_name in features_names:
			elem = access_tree_elem(data_file, feature_name)
			feature_i[feature_name] = elem

		features_list.append(feature_i)

	vec = DictVectorizer()
	vectored_data = vec.fit_transform(features_list)
	return vectored_data


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