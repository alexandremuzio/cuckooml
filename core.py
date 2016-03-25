from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn import decomposition
from sklearn.feature_extraction import DictVectorizer


def extract_section_from_data_file(data_file, section):
	return data_file[section]

def dict_as_json(dict_x):
	import json
	return json.dumps(dict_x, indent=1)

class OSOperation(object):
	"""Base abstract class for OS operations."""
	def __init__(self, name, entry):
		self.name = name.lower()
		self.entry = entry.lower()

	def __eq__(self, other):
		if isinstance(other, self.__class__):
			return self.__dict__ == other.__dict__
		else:
			return False

	def __hash__(self):
		return hash((self.name, self.entry))

	def __repr__(self):
		return self.name +  self.entry

	#Methods to be overrriden
	def get_name(self):
		raise NotImplementedError

	def get_operation(self):
		raise NotImplementedError

	def get_type(self):
		raise NotImplementedError

class ReadOperation(OSOperation):
	def get_type(self):
		return 'file'

	def get_operation(self):
		return self.name + "|" + self.entry

class VirusTotalOperation(OSOperation):
	def get_type(self):
		return 'virus_total'

	def get_operation(self):
		return self.name + '|' + self.entry

class NetworkOperation(OSOperation):
	def get_type(self):
		return 'network'

	def get_operation(self):
		return self.name + '|' + self.entry

class BehavioralProfile(object):
	def __init__(self, data_file):
		self.data_file = data_file
		self.os_operations = []
		self._create_profile()

	def _access_tree_elem(self, tree ,feature_name):
		tree_elem = tree

		args = feature_name.split('_')
		for arg in args:
			if not tree_elem.has_key(arg): 
				return dict()

			tree_elem = tree_elem[arg]

		return tree_elem

	def _create_profile(self):
		#behavior data from files
		sum_elem = self._access_tree_elem(self.data_file, "behavior_summary")
		# print dict_as_json(sum_elem)
		for sys_call_name, sys_calls in sum_elem.iteritems():
			for sys_call in sys_calls:
				if (sys_call_name == 'dll_loaded' or \
					sys_call_name == 'file_opened' or \
					sys_call_name == 'regkey_opened' or \
					sys_call_name == 'regkey_read'):
						self.os_operations.append(ReadOperation(sys_call_name, sys_call))

		#behavior data from virustotal
		scan_elem = self._access_tree_elem(self.data_file, "virustotal_scans")
		# print dict_as_json(scan_elem)
		for av_name, av_scan in scan_elem.iteritems():
			if av_scan['result'] is None: continue 
			self.os_operations.append(VirusTotalOperation(av_name, av_scan['result']))

		#TODO - behavior data from networking 

	def get_operations(self):
		return self.os_operations

class FeatureExtractor(object):
	def __init__(self, behavioral_profiles):
		self.behavioral_profiles = behavioral_profiles
		self._create_feature_set()

	def _create_feature_set(self):
		self.feature_set = set()

		#ignore features that appear only once
		feature_dic = {}
		for profile in self.behavioral_profiles:
			for op in profile.get_operations():
				if not feature_dic.get(op):
				 	feature_dic[op] = 0

				feature_dic[op] += 1

		# print feature_dic

		features_list = []
		for profile in self.behavioral_profiles:
			feature_elem = {}
			for op in profile.get_operations():
				if feature_dic[op] <= 1: continue
				feature_elem[op.get_operation()] = 1

			# print (repr(op) + " times: " + str(op_num))
			self.feature_set.add(op.get_operation())
			features_list.append(feature_elem)

		# print ('Number of features: ' + str(len(features_list)))
		# print dict_as_json(features_list)
		#create vectorized data
		vec = DictVectorizer()

		#TODO - sparse vector
		self.vectorized_data = vec.fit_transform(features_list).toarray()

		#Apply PCA
		pca = decomposition.PCA(n_components=2)
		self.reduced_data = pca.fit_transform(self.vectorized_data)

		# print (pca.explained_variance_)
		# print self.vectorized_data.shape

	def get_feature_set(self):
		return self.feature_set

	def get_vectorized_data(self):
		return self.vectorized_data

	def get_reduced_data(self):
		return self.reduced_data

class KMeans(object):
	def __init__(self, vectorized_data, n_clusters):
		self.vectorized_data = vectorized_data
		self.n_clusters = n_clusters

		self.cluster = cluster.KMeans(init='k-means++', n_clusters=self.n_clusters, n_init=10)

	def run_clustering(self):
		self.cluster.fit(self.vectorized_data)

	def labels(self):
		return self.cluster.labels_

	def vizualize_results_in_2d(self, reduced_data):	
		# Step size of the mesh. Decrease to increase the quality of the VQ.
		h = .015     # point in the mesh [x_min, m_max]x[y_min, y_max].

		# Plot the decision boundary. For that, we will assign a color to each
		x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
		y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

		# Obtain labels for each point in mesh. Use last trained model.
		Z = self.cluster.predict(np.c_[xx.ravel(), yy.ravel()])

		# Put the result into a color plot
		Z = Z.reshape(xx.shape)
		plt.figure(1)
		plt.clf()
		plt.imshow(Z, interpolation='nearest',
		           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
		           cmap=plt.cm.Paired,
		           aspect='auto', origin='lower')

		plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
		# Plot the centroids as a white X
		centroids = self.cluster.cluster_centers_
		plt.scatter(centroids[:, 0], centroids[:, 1],
		            marker='x', s=169, linewidths=3,
		            color='w', zorder=10)
		plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
		          'Centroids are marked with white cross')
		plt.xlim(x_min, x_max)
		plt.ylim(y_min, y_max)
		plt.xticks(())
		plt.yticks(())
		plt.show()