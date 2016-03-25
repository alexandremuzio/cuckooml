import json
from collections import defaultdict

from sklearn import cluster
from sklearn import decomposition
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#Utils
def extract_section_from_data_file(data_file, section):
	return data_file[section]

def dict_as_json(dict_x):
	return json.dumps(dict_x, indent=1)

#Access a section in data file (tree)
def access_data_elem(tree ,section_name):
	tree_elem = tree

	args = section_name.split('_')
	for arg in args:
		if arg not in tree_elem: 
			return dict()

		tree_elem = tree_elem[arg]
	return tree_elem

#################################################################
#Abstract behavioral profile
class BehavioralProfile(object):
	def __init__(self, data_file):
		self.data_file = data_file

	def _create_profile(self):
		raise NotImplementedError

#################################################################
#Behavioral profile based on idea 1 (abstraction of system calls)
class OSOperationsProfile(BehavioralProfile):
	def __init__(self, data_file):
		super().__init__(data_file)
		self.os_operations = []
		self._create_profile()

	def _create_profile(self):
		#behavior summary data from files
		summary_elem = access_data_elem(self.data_file, 'behavior_summary')
		# print dict_as_json(summary_elem)
		for sys_call_name, sys_calls in summary_elem.items():
			for sys_call in sys_calls:
				if (sys_call_name == 'dll_loaded' or \
					sys_call_name == 'file_opened' or \
					sys_call_name == 'regkey_opened' or \
					sys_call_name == 'regkey_read'):
						self.os_operations.append(ReadOperation(sys_call_name, sys_call))

		#behavior data from virustotal
		scan_elem = access_data_elem(self.data_file, 'virustotal_scans')
		# print dict_as_json(scan_elem)
		for av_name, av_scan in scan_elem.items():
			if av_scan['result'] is None: continue 
			self.os_operations.append(VirusTotalOperation(av_name, av_scan['result']))

		#TODO - behavior data from networking

	def get_operations(self):
		return self.os_operations

class OSOperation(object):
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

#Feature extraction
class OSOperationsFeatureExtractor(object):
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

			# print (repr(op) + ' times: ' + str(op_num))
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


#################################################################
#Behavioral profile based on idea 2
class APIProfile(BehavioralProfile):
	def __init__(self, data_file):
		super().__init__(data_file)
		self.api_dict = defaultdict(int)
		self._create_profile()

	def _create_profile(self):
		#td-idf counts list
		apistats_counts = []
		#behavior apistats data from files
		apistats_elem = access_data_elem(self.data_file, 'behavior_apistats')

		for pid, api_calls in apistats_elem.items():
			for api_call_name, api_call_val in api_calls.items():
				self.api_dict[api_call_name] += api_call_val

class APIFeatureExtractor(object):
	def __init__(self, behavioral_profiles, data_files):
		self.behavioral_profiles = behavioral_profiles
		self.data_files = data_files
		self._create_feature_set()

	def _create_feature_set(self):
		apistats_counts = []

		total_api_calls = defaultdict(int)
		#behavior apistats data from all files
		for data_file in self.data_files:
			apistats_elem = access_data_elem(data_file, 'behavior_apistats')
			for pid, api_calls in apistats_elem.items():
				for api_call_name, api_call_val in api_calls.items():
					total_api_calls[api_call_name] += api_call_val

		# print (dict_as_json(total_api_calls))

		#Create inverse lookup dic
		inv_total_api_calls = {}
		idx = 0
		for api_call in total_api_calls.keys():
			inv_total_api_calls[api_call] = idx
			idx += 1

		#create td-idf counts vector
		apistats_counts = []
		vector_len = len(total_api_calls)
		for behavioral_profile in self.behavioral_profiles:
			apistats_count_vector = [0] * vector_len
			for api_call, v in behavioral_profile.api_dict.items():
				apistats_count_vector[inv_total_api_calls[api_call]] += v
			apistats_counts.append(apistats_count_vector)

		# print(dict_as_json(apistats_counts))
		transformer = TfidfTransformer()
		tfidf = transformer.fit_transform(apistats_counts)

		self.vectorized_data = tfidf.toarray()

		#Apply PCA
		pca = decomposition.PCA(n_components=2)
		self.reduced_data = pca.fit_transform(self.vectorized_data)

	def get_feature_set(self):
		return self.feature_set

	def get_vectorized_data(self):
		return self.vectorized_data

	def get_reduced_data(self):
		return self.reduced_data