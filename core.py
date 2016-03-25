import json
from collections import defaultdict

from sklearn import cluster
from sklearn import decomposition
from sklearn.feature_extraction import DictVectorizer

#Utils
def extract_section_from_data_file(data_file, section):
	return data_file[section]

def dict_as_json(dict_x):
	return json.dumps(dict_x, indent=1)

#################################################################
#Abstract behavioral profile
class BehavioralProfile(object):
	def __init__(self, data_file):
		self.data_file = data_file

	#Access a section in data file (tree)
	def _access_data_elem(self, tree ,section_name):
		tree_elem = tree

		args = section_name.split('_')
		for arg in args:
			if arg not in tree_elem: 
				return dict()

			tree_elem = tree_elem[arg]

		return tree_elem

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
		summary_elem = self._access_data_elem(self.data_file, 'behavior_summary')
		# print dict_as_json(summary_elem)
		for sys_call_name, sys_calls in summary_elem.items():
			for sys_call in sys_calls:
				if (sys_call_name == 'dll_loaded' or \
					sys_call_name == 'file_opened' or \
					sys_call_name == 'regkey_opened' or \
					sys_call_name == 'regkey_read'):
						self.os_operations.append(ReadOperation(sys_call_name, sys_call))

		#behavior data from virustotal
		scan_elem = self._access_data_elem(self.data_file, 'virustotal_scans')
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
	def __init__(self, data_file, data_files):
		super().__init__(data_file)
		self.data_files = data_files
		self._create_profile()

	def _create_profile(self):
		#td-idf counts list
		apistats_counts = []
		#behavior apistats data from files
		for data_file in self.data_files:
			apistats_elem = self._access_data_elem(self.data_file, 'behavior_apistats')

			curr_api_dict = defaultdict(int)
			for pid, api_calls in apistats_elem():
				for api_call_name, api_call_val in api_calls.items():
					curr_api_dict[api_call_name] += api_call_val

			print

			#convert curr_api_dict to curr_api_list - remember array positioning

			apistats_counts.append(curr_api_dict)

		transformer = TfidfTransformer()
		tfidf = transformer.fit_transform(apistats_counts)

	def get_feature_set(self):
		return self.feature_set

	def get_vectorized_data(self):
		return self.vectorized_data

	def get_reduced_data(self):
		return self.reduced_data

class APIFeatureExtractor(object):
	def __init__(self, behavioral_profiles):
		self.behavioral_profiles = behavioral_profiles
		self._create_feature_set()

	def _create_feature_set(self):
		pass