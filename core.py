from sklearn.feature_extraction import DictVectorizer
from sklearn import cluster

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
		for op, op_num in feature_dic.iteritems():
			if op_num == 1: continue

			self.feature_set.add(op.get_operation())
			features_list.append({op.get_type(): op.get_operation()})

		#print dict_as_json(features_list)
		#create vectorized data
		vec = DictVectorizer()

		#TODO - sparse vector
		self.vectorized_data = vec.fit_transform(features_list).toarray()

		# import numpy as np
		# np.savetxt('test.txt', self.vectorized_data.toarray())


	def get_feature_set(self):
		return self.feature_set

	def get_vectorized_data(self):
		return self.vectorized_data

class KMeans(object):
	def __init__(self, vectorized_data):
		self.cluster = cluster.KMeans(init='k-means++', n_clusters=3, n_init=10)
		self.vectorized_data = vectorized_data

	def run_clustering(self):
		self.cluster.fit(self.vectorized_data)

	def labels(self):
		return self.cluster.labels_