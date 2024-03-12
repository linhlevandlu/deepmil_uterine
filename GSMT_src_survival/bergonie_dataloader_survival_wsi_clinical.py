import os
import torch
import numpy as np
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from collections import Counter
import pickle as pl

class Patient(data_utils.Dataset):
	"""docstring for MNISTBags"""
	def __init__(self, root_folder, 
						patient_list, 
						labels_list, 
						clinical_list,
						patients_wsi, 
						transf = None, 
						n_tiles = 0, 
						seed = 2334,
						root2 = False):
		super(Patient, self).__init__()
		self.root_folder = root_folder
		self.patient_list = patient_list
		self.outcome_list = labels_list
		self.clinical_list = clinical_list
		self.transforms = transf
		self.seed = seed
		self.n_tiles = n_tiles
		self.patients_wsi = patients_wsi
		self.sampling_model = SMOTE(k_neighbors = 1, random_state = seed)
		#self.list_of_bags = self.get_all_bags()
		if root2:
			print("With SMOTE sampling...")
			self.list_of_bags = self.get_all_bags_sampling()
		else:
			print("Without SMOTE sampling....")
			self.list_of_bags = self.get_all_bags()
		# if root2 != None:
		# 	self.root2 = root2
		# 	self.list_of_bags = self.get_all_bags_2roots()

	def _read_folder(self, npz_file, label = 0):
		npz_array = np.load(str(npz_file), allow_pickle=True)['arr_0']
		list_labels = [label] * npz_array.shape[0]
		#print(npz_array.shape, np.array(list_labels).shape, npz_array.shape[0])
		return npz_array, np.array(list_labels), npz_array.shape[0]

	def random_select_tiles_wsi(self, np_array, num_tiles = 10000):
		n_rows = np_array.shape[0]
		rd_indices = np.random.choice(n_rows, size = num_tiles, replace = True)
		selected_array = np_array[rd_indices,:]

		return selected_array
	
	def _setup_bag_2(self, pname, n_tiles = 10000): 
		#print("Get bag of {}".format(self.region))
		list_wsis = self.patients_wsi[pname]
		all_wsi = []
		for wsi in list_wsis:
			wsi_path = os.path.join(self.root_folder, wsi)
			c_files, _, _ = self._read_folder(wsi_path)
			all_wsi.append(c_files)
		wsi_file = np.concatenate(all_wsi, axis = 0)
		wsi_file= shuffle(wsi_file, random_state = self.seed)
		selected_tiles = self.random_select_tiles_wsi(wsi_file, num_tiles = n_tiles)
		print('Shape: ', pname, wsi_file.shape, selected_tiles.shape)
		return selected_tiles

	def get_all_bags(self):
		all_bags = []
		for idx in range(len(self.patient_list)):
			patient = self.patient_list[idx]
			status, surv_time = self.outcome_list[idx]
			clinical = self.clinical_list[idx]
			# Get data and load 
			wsi_file = self._setup_bag_2(patient, self.n_tiles)
			#print(wsi_file.shape, status.shape, clinical.shape)
			status = torch.Tensor([status])
			surv_time_tensor = torch.Tensor([surv_time])
			input_tensor = torch.from_numpy(wsi_file)
			clinical_tensor = torch.from_numpy(clinical)

			sample = {'itensor': input_tensor, 
						'istatus': status,
						'isurvtime':surv_time_tensor,
						'iclinical': clinical_tensor}

			if self.transforms is not None:
				sample = self.transforms(sample)
			all_bags.append((sample['itensor'], 
								sample['istatus'],
								sample['isurvtime'],
								sample['iclinical']))
		return all_bags

	def _setup_bag_2roots(self, root, pname, n_tiles = 10000): 
		#print("Get bag of {}".format(self.region))
		list_wsis = self.patients_wsi[pname]
		all_wsi = []
		for wsi in list_wsis:
			wsi_path = os.path.join(root, wsi)
			c_files, _, _ = self._read_folder(wsi_path)
			all_wsi.append(c_files)
		wsi_file = np.concatenate(all_wsi, axis = 0)
		wsi_file= shuffle(wsi_file, random_state = self.seed)
		selected_tiles = self.random_select_tiles_wsi(wsi_file, num_tiles = n_tiles)
		print('Shape: ', pname, wsi_file.shape, selected_tiles.shape)
		return selected_tiles

	def get_all_bags_2roots(self):
		all_bags = []
		for idx in range(len(self.patient_list)):
			patient = self.patient_list[idx]
			status = self.labels_list[idx]
			clinical = self.clinical_list[idx]
			# Get data and load 
			wsi_file1 = self._setup_bag_2roots(self.root_folder, patient, self.n_tiles)
			wsi_file2 = self._setup_bag_2roots(self.root2, patient, self.n_tiles)
			
			# from root 1
			status = torch.Tensor([status])
			input_tensor = torch.from_numpy(wsi_file1)
			clinical_tensor = torch.from_numpy(clinical)

			sample = {'itensor': input_tensor, 
						'istatus': status,
						'iclinical': clinical_tensor}

			if self.transforms is not None:
				sample = self.transforms(sample)

			all_bags.append((sample['itensor'], 
								sample['istatus'],
								sample['iclinical']))
			# from root 2
			input_tensor2 = torch.from_numpy(wsi_file2)

			sample = {'itensor': input_tensor2, 
						'istatus': status,
						'iclinical': clinical_tensor}

			if self.transforms is not None:
				sample = self.transforms(sample)

			all_bags.append((sample['itensor'], 
								sample['istatus'],
								sample['iclinical']))

			#if idx > 10:
			#	break
		return all_bags
	
	def get_all_bags_sampling(self):
		all_bags = []
		all_status = []
		for idx in range(len(self.patient_list)):
			patient = self.patient_list[idx]
			status = self.labels_list[idx]
			clinical = self.clinical_list[idx]

			# Get data and load 
			wsi_file = self._setup_bag_2(patient, self.n_tiles)
			wsi_file = wsi_file.reshape(-1)
			wsi_features = np.append(wsi_file, clinical)

			all_bags.append(wsi_features)
			all_status.append(status)

		all_bags = np.array(all_bags)
		all_status = np.array(all_status)
		print("Before sampling: ", all_bags.shape, np.unique(all_status, return_counts = True))
		#print(Counter(all_status).items())

		all_bags_resampled, status_resampled = self.sampling_model.fit_resample(all_bags, all_status)
		print("After sampling: ",all_bags_resampled.shape,status_resampled.shape, np.unique(status_resampled, return_counts = True))
		#print(Counter(status_resampled).items())

		all_bags_tensor = []
		for i in range(status_resampled.shape[0]):
			wsi_feature = all_bags_resampled[i,:]
			wsi_file = wsi_feature[:-1].reshape(self.n_tiles, 2048)
			clinical = wsi_feature[-1:]
			
			print("Round", clinical)
			status = status_resampled[i]
			#print(wsi_file.shape, status.shape, clinical.shape)

			status = torch.Tensor([status])
			input_tensor = torch.from_numpy(wsi_file)
			clinical_tensor = torch.from_numpy(clinical)

			sample = {'itensor': input_tensor, 
						'istatus': status,
						'iclinical': clinical_tensor}

			if self.transforms is not None:
				sample = self.transforms(sample)
			all_bags_tensor.append((sample['itensor'], 
								sample['istatus'],
								sample['iclinical']))

		return all_bags_tensor

	def __len__(self):
		return len(self.list_of_bags)

	def __getitem__(self, index):
		return self.list_of_bags[index]


def read_row(row, nb_clinicals = 1):
	if nb_clinicals == 0:
		name, label, time = row
		cli_list = []
	elif nb_clinicals == 1:
		name, label, time, mit = row
		cli_list = [float(mit)]
	elif nb_clinicals == 2:
		name, label, time, mit, tc = row
		cli_list = [float(mit), float(tc)]
	elif nb_clinicals == 3:
		name, label, time, mit, tc, at = row
		cli_list = [float(mit), float(tc), float(at)]
	elif nb_clinicals == 4:
		name, label, time, mit, tc, at, lvi = row
		cli_list = [float(mit), float(tc), float(at), float(lvi)]
	return name, label, time, cli_list

def read_csv(csv_file, wsi_folder):
	# Read WSI folder
	list_files = sorted(list(os.listdir(wsi_folder)))
	# Read CSV file
	file = open(csv_file)
	csvreader = csv.reader(file)
	header = next(csvreader)
	nclinical = len(header) - 3
	print("Number of clinicals: ", nclinical)
	
	label_patients_dict = {}
	patients_wsi_dict = {}
	patients_clinical = {}
	labels = []
	surv_times = []
	patients = []
	patient_outcomes = {}
	for row in csvreader:
		name, label, time, cli_list = read_row(row, nb_clinicals = nclinical)
		labels.append(int(label))
		surv_times.append(float(time))
		patients.append(name)
		#patients[name] = int(label)
		wsi_files = [f_name for f_name in list_files if name in f_name]
		patients_wsi_dict[name] = wsi_files
		
		patients_clinical[name] = np.array(cli_list)
		patient_outcomes[name] = (int(label), float(time))

	assert len(labels) == len(patients)

	# group the patients by label 
	arr_labels = np.array(labels)
	arr_labels_un = np.unique(arr_labels)
	#print(arr_labels_un)
	for lbl in arr_labels_un:
		lbl_patients = []
		for idx in range(len(labels)):
			slbl = labels[idx]
			if lbl == slbl:
				lbl_patients.append(patients[idx])
		label_patients_dict[lbl] = lbl_patients

	file.close()

	return label_patients_dict, patients_wsi_dict, patients_clinical, patient_outcomes

def search_keys(npz_name, labels_keys):
	for x_key in labels_keys:
		if x_key in npz_name:
			return x_key 
	return None

def weighted_sampler(target_labels):
	labels_count = np.unique(target_labels, return_counts = True)[1]
	class_weight = 1./ labels_count
	
	samples_weight = class_weight[target_labels]
	samples_weight = torch.from_numpy(samples_weight)
	samples_weight = samples_weight.double()
	sampler = data_utils.WeightedRandomSampler(samples_weight, len(samples_weight))
	return sampler

def random_select_tiles_wsi(np_array, num_tiles = 200):
	n_rows = np_array.shape[0]
	rplace = False
	if n_rows < num_tiles:
		rplace =  True
	rd_indices = np.random.choice(n_rows, size = num_tiles, replace = rplace)
	selected_array = np_array[rd_indices,:]
	remain_array = np.delete(np_array, rd_indices, 0)
	return selected_array, remain_array

def get_per_patient2(patient_array, 
					label, 
					n_tiles = 200):
	train_array, _ = random_select_tiles_wsi(patient_array, num_tiles = n_tiles)
	train_labels = [label] * train_array.shape[0]
	train_labels = np.array(train_labels)

	return train_array, train_labels

def get_bag2(root_folder, 
				list_patients, 
				list_patients_labels, 
				patients_wsi, 
				n_stiles = 100,
				seed = 2452):
	np_labels = np.array(list_patients_labels)
	np_unique, np_counts = np.unique(np_labels, return_counts = True)
	print(np_unique, np_counts)
	np_ratios = np.around(np.max(np_counts) / np_counts)
	dict_labels = {}
	for ilbl in range(len(np_unique)):
		lbl = np_unique[ilbl]
		dict_labels[lbl] = np_ratios[ilbl]
	print(dict_labels)

	list_tiles = []
	list_labels = []
	#r_state = np.random.RandomState(seed)
	for pidx in range(len(list_patients)):
		patient = list_patients[pidx]
		label = list_patients_labels[pidx]
		list_wsis = patients_wsi[patient]

		n_select_tiles = n_stiles * dict_labels[label]
		patient_tiles = load_wsis(list_wsis, root_folder, seed)
		print(patient, label, patient_tiles.shape, n_select_tiles)

		train_array, train_labels = get_per_patient2(patient_tiles,
													label, 
													n_tiles = int(n_select_tiles))
		patient_ids = [pidx] * train_array.shape[0]
		

		list_tiles.append(train_array)
		list_labels.append(train_labels)

	list_tiles = np.concatenate(tuple(list_tiles))
	list_labels = np.concatenate(tuple(list_labels), axis = 0)

	list_tiles, list_labels, list_pids = shuffle(list_tiles, list_labels, random_state = seed)

	return list_tiles, list_labels

def split_patients(list_patients, list_labels, val_per = 0.2, seed = 2452):
	list_patients = np.array(list_patients)
	list_labels = np.array(list_labels)
	st_split = StratifiedShuffleSplit(n_splits = 1, test_size = val_per, random_state = seed)
	for train_idx, valid_idx in st_split.split(list_patients, list_labels):
		train_patients, valid_patients = list_patients[train_idx], list_patients[valid_idx]
		train_labels, valid_labels = list_labels[train_idx], list_labels[valid_idx]

	# train_patients = train_patients[:5]
	# train_labels = train_labels[:5]
	# valid_patients = valid_patients[:5]
	# valid_labels = valid_labels[:5]
	return train_patients, train_labels, valid_patients, valid_labels

def split_patients_from_csv(root_folder, 
					csv_labels,
					n_tiles = 10000,
					batch_size = 256,
					val_per = 0.2,
					seed = 2334):
	label_patients_dict, patients_wsi_dict, patients_clinical, patient_outcomes = read_csv(csv_labels, root_folder)
	list_labels = list(label_patients_dict)

	all_train_patients = []
	all_train_labels = []
	all_valid_patients = []
	all_valid_labels = []
	for lbl in list_labels:
		lbl_patients = label_patients_dict[lbl]
		lbl_labels = [lbl] * len(lbl_patients)
		train_p, train_l, valid_p, valid_l = split_patients(lbl_patients, 
															lbl_labels, 
															val_per = val_per, 
															seed = seed)
		all_train_patients += list(train_p)
		all_train_labels += list(train_l)
		all_valid_patients += list(valid_p)
		all_valid_labels += list(valid_l)

	all_train_clinicals = []
	all_valid_clinicals = []
	all_train_outcomes = []
	all_valid_outcomes = []
	for tpatient in all_train_patients:
		all_train_clinicals.append(patients_clinical[tpatient])
		all_train_outcomes.append(patient_outcomes[tpatient])
	all_train_clinicals = np.array(all_train_clinicals).astype(np.float32)

	for vpatient in all_valid_patients:
		all_valid_clinicals.append(patients_clinical[vpatient])
		all_valid_outcomes.append(patient_outcomes[vpatient])

	all_valid_clinicals = np.array(all_valid_clinicals).astype(np.float32)

	print("Traing/valid: ", len(all_train_labels), len(all_valid_labels))
	train_valid_dict = {}
	train_valid_dict['train'] = (all_train_patients, all_train_labels, all_train_clinicals, all_train_outcomes)
	train_valid_dict['val'] = (all_valid_patients, all_valid_labels, all_valid_clinicals, all_valid_outcomes)

	return train_valid_dict, patients_wsi_dict

def get_data_by_patient(root_folder, 
					csv_labels,
					n_tiles = 10000,
					batch_size = 256,
					val_per = 0.2,
					seed = 2334):
	train_valid_dict, patients_wsi_dict = split_patients_from_csv(root_folder, 
														csv_labels,
														n_tiles,
														batch_size,
														val_per,
														seed)
	
	# Create train loader
	all_train_patients, _, all_train_clinicals, all_train_outcomes = train_valid_dict['train']
	print('Training patients: ', all_train_patients)
	train_transf = None
	train_dataset = Patient(root_folder,
							all_train_patients, 
							all_train_outcomes,
							all_train_clinicals,
							patients_wsi_dict,
							transf = train_transf, 
							n_tiles = n_tiles,
							seed = seed)

	train_dataloader = torch.utils.data.DataLoader(train_dataset, 
													batch_size = batch_size, 
													num_workers=20,)
	# Create data loader
	all_valid_patients, _, all_valid_clinicals, all_valid_outcomes = train_valid_dict['val']
	print('Validation patients: ', all_valid_patients)
	valid_transf = None
	valid_dataset = Patient(root_folder,
							all_valid_patients, 
								all_valid_outcomes,
								all_valid_clinicals,
								patients_wsi_dict,
								transf = valid_transf,
								n_tiles = n_tiles,
								seed = seed)

	valid_dataloader = torch.utils.data.DataLoader(valid_dataset, 
													batch_size = batch_size, 
													shuffle = False,
													num_workers=20,)
	
	print("Total training tiles....: {}".format(len(train_dataset)))
	print("Total validation tiles....: {}".format(len(valid_dataset)))
	
	return train_dataloader, valid_dataloader

########################################### Cross-Validation ####################################################
def split_patients_per_label_CV(list_patients, list_labels, patients_clinical, patient_outcomes, n_folds = 5, seed = 2452):
	list_clinicals = [patients_clinical[patient]for patient in list_patients]
	patient_outcomes = [patient_outcomes[patient]for patient in list_patients]

	list_patients = np.array(list_patients)
	list_labels = np.array(list_labels)
	list_clinicals = np.array(list_clinicals)
	patient_outcomes = np.array(patient_outcomes)

	kf = StratifiedKFold(n_splits = n_folds, shuffle=True, random_state = seed)

	train_on_one_label = []
	valid_on_one_label = []

	for train_index, valid_index in kf.split(list_patients, list_labels):
		train_patients, valid_patients = list_patients[train_index], list_patients[valid_index]
		train_labels, valid_labels = list_labels[train_index], list_labels[valid_index]
		train_clinicals, valid_clinicals = list_clinicals[train_index], list_clinicals[valid_index]
		train_outcomes, valid_outcomes = patient_outcomes[train_index], patient_outcomes[valid_index]
		
		train_on_one_label.append((train_patients, train_labels, train_clinicals, train_outcomes))
		valid_on_one_label.append((valid_patients, valid_labels, valid_clinicals, valid_outcomes))

	return train_on_one_label, valid_on_one_label

def split_patients_per_fold_CV(root_folder, 
					csv_labels,
					n_folds = 5,
					seed = 2452):
	label_patients_dict, patients_wsi_dict, patients_clinical, patients_outcomes = read_csv(csv_labels, root_folder)
	list_labels = list(label_patients_dict)

	all_classes_dict = {}
	for lbl in list_labels:
		lbl_patients = label_patients_dict[lbl]
		lbl_labels = [lbl] * len(lbl_patients)
		train_data, valid_data = split_patients_per_label_CV(lbl_patients, 
															lbl_labels, 
															patients_clinical,
															patients_outcomes,
															n_folds = n_folds, 
															seed = seed)
		all_classes_dict[str(lbl)] = (train_data, valid_data)
	return all_classes_dict, list_labels, patients_wsi_dict

def get_data_cross_validation_SKFold(csv_labels, 
									root_folder, 
									k_folds = 5, 
									seed = 2452):
	all_classes_dict, list_labels, patients_wsi_dict = split_patients_per_fold_CV(root_folder,
																					csv_labels,
																					k_folds,
																					seed)
	train_patients_dict = {}
	valid_patients_dict = {}
	for idx in range(k_folds):
		all_train_patients = []
		all_train_labels = []
		all_train_clinicals = []
		all_train_outcomes = []
		all_valid_patients = []
		all_valid_labels = []
		all_valid_clinicals = []
		all_valid_outcomes = []
		for lbl in list_labels:
			train_data, valid_data = all_classes_dict[str(lbl)]
			# Get train data
			train_data_idx = train_data[idx]
			train_p, train_l, train_clin, train_out = train_data_idx[0], train_data_idx[1], train_data_idx[2], train_data_idx[3]
			#Get valid data
			valid_data_idx = valid_data[idx]
			valid_p, valid_l, valid_clin, valid_out = valid_data_idx[0], valid_data_idx[1], valid_data_idx[2], valid_data_idx[3]

			all_train_patients += list(train_p)
			all_train_labels += list(train_l)
			all_train_clinicals += list(train_clin)
			all_train_outcomes += list(train_out)

			all_valid_patients += list(valid_p)
			all_valid_labels += list(valid_l)
			all_valid_clinicals += list(valid_clin)
			all_valid_outcomes += list(valid_out)

		print("Fold ", idx, 
				"Train:{}/{}".format(len(all_train_patients), len(all_train_labels)), 
				"Valid: {}/{}".format(len(all_valid_patients), len(all_valid_labels)))

		train_patients_dict['fold_' + str(idx)] = (all_train_patients, all_train_labels, all_train_clinicals, all_train_outcomes)
		valid_patients_dict['fold_' + str(idx)] = (all_valid_patients, all_valid_labels, all_valid_clinicals, all_valid_outcomes)

	return train_patients_dict, valid_patients_dict, patients_wsi_dict

def get_data_from_cross_validation(root_folder, 
	train_dict,
	valid_dict, 
	patients_wsi,
	n_tiles = 10000,
	batch_size = 1, 
	seed = 2334):
	
	# Read sample files and split
	# print("Load the loaders .........")
	train_patients, train_labels, train_clinicals, train_outcomes = train_dict[0], train_dict[1], train_dict[2], train_dict[3]
	valid_patients, valid_labels, valid_clinicals, valid_outcomes = valid_dict[0], valid_dict[1], valid_dict[2], valid_dict[3]

	train_transf = None
	train_dataset = Patient(root_folder, 
							train_patients, 
							train_outcomes,
							train_clinicals,
							patients_wsi, 
							transf = train_transf, 
							n_tiles = n_tiles, 
							seed = seed,
							root2 = False)

	#train_status = list(list(zip(*train_labels))[0])
	#train_status = list(list(zip(*train_labels)))
	#train_status = list(map(int, train_status)) # convert list of float number to int
	#train_status = train_labels
	#train_weighted_sampler = weighted_sampler(train_status)
	train_dataloader = torch.utils.data.DataLoader(train_dataset, 
													batch_size = batch_size, 
													num_workers=1,
													shuffle = True,)

	#valid_transf = transforms.Compose([RandomNormalize(prob = 1.0)])
	valid_transf = None
	valid_dataset = Patient(root_folder, 
							valid_patients, 
							valid_outcomes, 
							valid_clinicals,
							patients_wsi,
							transf = valid_transf, 
							n_tiles = n_tiles, 
							seed = seed,
							root2 = False)
	#valid_status = list(list(zip(*valid_labels))[0])
	#valid_status = list(map(int, valid_status))
	#valid_weighted_sampler = weighted_sampler(valid_status)
	valid_dataloader = torch.utils.data.DataLoader(valid_dataset, 
													batch_size = batch_size, 
													num_workers = 1, 
													shuffle = False,)
													#sampler = valid_weighted_sampler)

	print("Total training patients....: {}".format(len(train_dataset)))
	print("Total validation patients....: {}".format(len(valid_dataset)))

	return train_dataloader, valid_dataloader

def split_patients_with_clinicals(list_patients, list_labels, list_clinicals, list_outcomes, val_per = 0.2, seed = 2452):
	list_patients = np.array(list_patients)
	list_labels = np.array(list_labels)
	list_clinicals = np.array(list_clinicals)
	list_outcomes = np.array(list_outcomes)
	st_split = StratifiedShuffleSplit(n_splits = 1, test_size = val_per, random_state = seed)
	for train_idx, valid_idx in st_split.split(list_patients, list_labels):
		train_patients, valid_patients = list_patients[train_idx], list_patients[valid_idx]
		train_labels, valid_labels = list_labels[train_idx], list_labels[valid_idx]
		train_clinicals, valid_clinicals = list_clinicals[train_idx], list_clinicals[valid_idx]
		train_outs, valid_outs = list_outcomes[train_idx], list_outcomes[valid_idx]

	#train_patients = train_patients[:5]
	#train_labels = train_labels[:5]
	#valid_patients = valid_patients[:5]
	#valid_labels = valid_labels[:5]
	return train_patients, train_labels, train_clinicals, train_outs, valid_patients, valid_labels, valid_clinicals, valid_outs


def split_train_data(root_folder, train_dict, patients_wsi_dict, n_tiles = 100000, batch_size = 1,
					val_per = 0.2,
					seed = 2334):

	#label_patients_dict, patients_wsi_dict, patients_clinical = read_csv(csv_labels, root_folder)
	#list_labels = list(label_patients_dict)
	print("Take 20per as validation set")
	train_patients, train_labels, train_clinicals, train_outcomes = train_dict[0], train_dict[1], train_dict[2], train_dict[3]
	list_labels = np.unique(train_labels)
	print(list_labels)
	all_train_patients = []
	all_train_labels = []
	all_valid_patients = []
	all_valid_labels = []
	all_train_clinicals = []
	all_valid_clinicals = []
	all_train_outcomes = []
	all_valid_outcomes = []
	for lbl in list_labels:
		list_patients = []
		list_labels2 = []
		list_clinicals = []
		list_outcomes = []
		for idx in range(len(train_labels)):
			if train_labels[idx] == lbl:
				list_patients.append(train_patients[idx])
				list_labels2.append(train_labels[idx])
				list_clinicals.append(train_clinicals[idx])
				list_outcomes.append(train_outcomes[idx])
		train_p, train_l, train_c, train_out, valid_p, valid_l, valid_c, valid_out = split_patients_with_clinicals(list_patients, 
															list_labels2, 
															list_clinicals,
															list_outcomes,
															val_per = val_per, 
															seed = seed)
		print("Train: ", len(train_l), ", Valid: ", len(valid_l))
		all_train_patients += list(train_p)
		all_train_labels += list(train_l)
		all_train_clinicals += list(train_c)
		all_train_outcomes += list(train_out)

		all_valid_patients += list(valid_p)
		all_valid_labels += list(valid_l)
		all_valid_clinicals += list(valid_c)
		all_valid_outcomes += list(valid_out)
		
	all_train_clinicals = np.array(all_train_clinicals).astype(np.float32)
	all_valid_clinicals = np.array(all_valid_clinicals).astype(np.float32)

	#print("Traing/valid: ", len(all_train_labels), len(all_valid_labels))
	#train_valid_dict = {}
	#train_valid_dict['train'] = (all_train_patients, all_train_labels, all_train_clinicals)
	#train_valid_dict['val'] = (all_valid_patients, all_valid_labels, all_valid_clinicals)

	train_transf = None
	train_dataset = Patient(root_folder,
							all_train_patients, 
							all_train_outcomes,
							all_train_clinicals,
							patients_wsi_dict,
							transf = train_transf, 
							n_tiles = n_tiles,
							seed = seed,
							root2 = False)

	train_dataloader = torch.utils.data.DataLoader(train_dataset, 
													batch_size = batch_size, 
													num_workers=1,)
	# Create data loader
	valid_transf = None
	valid_dataset = Patient(root_folder,
							all_valid_patients, 
								all_valid_outcomes,
								all_valid_clinicals,
								patients_wsi_dict,
								transf = valid_transf,
								n_tiles = n_tiles,
								seed = seed,
								root2 = False)

	valid_dataloader = torch.utils.data.DataLoader(valid_dataset, 
													batch_size = batch_size, 
													shuffle = False,
													num_workers=1,)
	
	print("Total training tiles....: {}".format(len(train_dataset)))
	print("Total validation tiles....: {}".format(len(valid_dataset)))
	
	return train_dataloader, valid_dataloader
########################################### End Cross-Validation ####################################################
