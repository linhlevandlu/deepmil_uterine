import os
import torch
import numpy as np
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE, ADASYN
#from collections import Counter

class Patient(data_utils.Dataset):
	"""docstring for MNISTBags"""
	def __init__(self, root_folder, 
						patient_list, 
						labels_list, 
						patients_wsi, 
						transf = None, 
						n_tiles = 0, 
						seed = 2334):
		super(Patient, self).__init__()
		self.root_folder = root_folder
		self.patient_list = patient_list
		self.labels_list = labels_list
		self.transforms = transf
		self.seed = seed
		self.n_tiles = n_tiles
		self.patients_wsi = patients_wsi
		self.list_of_bags = self.get_all_bags()

	def _read_folder(self, npz_file, label = 0):
		npz_array = np.load(npz_file)['arr_0']
		list_labels = [label] * npz_array.shape[0]
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
		
		return selected_tiles

	def get_all_bags(self):
		all_bags = []
		for idx in range(len(self.patient_list)):
			p_tensor, p_status = self._get_patient_idx(idx)
			all_bags.append((p_tensor, p_status))
		return all_bags

	def _get_patient_idx(self, pidx):
		patient = self.patient_list[pidx]
		status = self.labels_list[pidx]
		# Get data and load 
		wsi_file = self._setup_bag_2(patient, self.n_tiles)
		status = torch.Tensor([status])
		input_tensor = torch.from_numpy(wsi_file)
		sample = {'itensor': input_tensor, 'istatus': status,}

		if self.transforms is not None:
			sample = self.transforms(sample)
		return sample['itensor'], sample['istatus']
		
	def __len__(self):
		return len(self.list_of_bags)

	def __getitem__(self, index):
		return self.list_of_bags[index]
		#return self._get_patient_idx(index)

def read_csv(csv_file, wsi_folder):
	# Read WSI folder
	list_files = sorted(list(os.listdir(wsi_folder)))
	# Read CSV file
	file = open(csv_file)
	csvreader = csv.reader(file)
	header = next(csvreader)

	label_patients_dict = {}
	patients_wsi_dict = {}
	labels = []
	patients = []
	for row in csvreader:
		name, label = row
		labels.append(int(label))
		patients.append(name)
		#patients[name] = int(label)
		wsi_files = [f_name for f_name in list_files if name in f_name]
		patients_wsi_dict[name] = wsi_files

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

	return label_patients_dict, patients_wsi_dict

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
	label_patients_dict, patients_wsi_dict = read_csv(csv_labels, root_folder)
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

	print("Traing/valid: ", len(all_train_labels), len(all_valid_labels))
	train_valid_dict = {}
	train_valid_dict['train'] = (all_train_patients, all_train_labels)
	train_valid_dict['val'] = (all_valid_patients, all_valid_labels)

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
	all_train_patients, all_train_labels = train_valid_dict['train']
	train_transf = None
	train_dataset = Patient(root_folder,
							all_train_patients, 
							all_train_labels,
							patients_wsi_dict,
							transf = train_transf, 
							n_tiles = n_tiles,
							seed = seed)

	train_dataloader = torch.utils.data.DataLoader(train_dataset, 
													batch_size = batch_size, 
													num_workers=1,)
	# Create data loader
	all_valid_patients, all_valid_labels = train_valid_dict['val']
	valid_transf = None
	valid_dataset = Patient(root_folder,
							all_valid_patients, 
								all_valid_labels,
								patients_wsi_dict,
								transf = valid_transf,
								n_tiles = n_tiles,
								seed = seed)

	valid_dataloader = torch.utils.data.DataLoader(valid_dataset, 
													batch_size = batch_size, 
													shuffle = False,
													num_workers=1,)
	
	print("Total training tiles....: {}".format(len(train_dataset)))
	print("Total validation tiles....: {}".format(len(valid_dataset)))
	
	return train_dataloader, valid_dataloader

########################################### Cross-Validation ####################################################

def split_patients_per_label_CV(list_patients, list_labels, n_folds = 5, seed = 2452):
	list_patients = np.array(list_patients)
	list_labels = np.array(list_labels)
	kf = StratifiedKFold(n_splits = n_folds, shuffle=True, random_state = seed)

	train_on_one_label = []
	valid_on_one_label = []

	for train_index, valid_index in kf.split(list_patients, list_labels):
		train_patients, valid_patients = list_patients[train_index], list_patients[valid_index]
		train_labels, valid_labels = list_labels[train_index], list_labels[valid_index]
		
		train_on_one_label.append((train_patients, train_labels))
		valid_on_one_label.append((valid_patients, valid_labels))

	return train_on_one_label, valid_on_one_label

def split_patients_per_fold_CV(root_folder, 
					csv_labels,
					n_folds = 5,
					seed = 2452):
	label_patients_dict, patients_wsi_dict = read_csv(csv_labels, root_folder)
	list_labels = list(label_patients_dict)

	all_classes_dict = {}
	for lbl in list_labels:
		lbl_patients = label_patients_dict[lbl]
		lbl_labels = [lbl] * len(lbl_patients)
		train_data, valid_data = split_patients_per_label_CV(lbl_patients, 
															lbl_labels, 
															n_folds = n_folds, 
															seed = seed)
		all_classes_dict[str(lbl)] = (train_data, valid_data)
	return all_classes_dict, list_labels, patients_wsi_dict

def get_data_cross_validation_SKFold(csv_labels, 
									root_folder, 
									k_folds = 5, 
									seed = 2334):
	all_classes_dict, list_labels, patients_wsi_dict = split_patients_per_fold_CV(root_folder,
													csv_labels,
													k_folds,
													seed)
	train_patients_dict = {}
	valid_patients_dict = {}
	for idx in range(k_folds):
		all_train_patients = []
		all_train_labels = []
		all_valid_patients = []
		all_valid_labels = []
		for lbl in list_labels:
			train_data, valid_data = all_classes_dict[str(lbl)]
			# Get train data
			train_data_idx = train_data[idx]
			train_p, train_l = train_data_idx[0], train_data_idx[1]
			#Get valid data
			valid_data_idx = valid_data[idx]
			valid_p, valid_l = valid_data_idx[0], valid_data_idx[1]

			all_train_patients += list(train_p)
			all_train_labels += list(train_l)
			all_valid_patients += list(valid_p)
			all_valid_labels += list(valid_l)
		print("Fold ", idx, 
				"Train:{}/{}".format(len(all_train_patients), len(all_train_labels)), 
				"Valid: {}/{}".format(len(all_valid_patients), len(all_valid_labels)))

		# all_train_patients = all_train_patients[:25]
		# all_train_labels = all_train_labels[:25]
		# all_valid_patients = all_valid_patients[:25]
		# all_valid_labels = all_valid_labels[:25]

		train_patients_dict['fold_' + str(idx)] = (all_train_patients, all_train_labels)
		valid_patients_dict['fold_' + str(idx)] = (all_valid_patients, all_valid_labels)
	return train_patients_dict, valid_patients_dict, patients_wsi_dict

def get_data_from_cross_validation(root_folder, 
	train_dict,
	valid_dict, 
	patients_wsi,
	n_tiles = 10000,
	batch_size = 1, 
	seed = 2334):
	
	# Read sample files and split
	print("Load the loaders .........")
	train_patients, train_labels = train_dict[0], train_dict[1]
	valid_patients, valid_labels = valid_dict[0], valid_dict[1]

	train_transf = None
	train_dataset = Patient(root_folder, 
							train_patients, 
							train_labels,
							patients_wsi, 
							transf = train_transf, 
							n_tiles = n_tiles, 
							seed = seed)
	
	train_dataloader = torch.utils.data.DataLoader(train_dataset, 
													batch_size = batch_size, 
													num_workers=1,
													shuffle = True,)

	#valid_transf = transforms.Compose([RandomNormalize(prob = 1.0)])
	valid_transf = None
	valid_dataset = Patient(root_folder, 
							valid_patients, 
							valid_labels, 
							patients_wsi,
							transf = valid_transf, 
							n_tiles = n_tiles, 
							seed = seed)
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

def split_train_data_in_CV(root_folder, train_dict, patients_wsi_dict, n_tiles = 100000, batch_size = 1,
					val_per = 0.2,
					seed = 2334):

	#label_patients_dict, patients_wsi_dict, patients_clinical = read_csv(csv_labels, root_folder)
	#list_labels = list(label_patients_dict)
	train_patients, train_labels = train_dict[0], train_dict[1]
	list_labels = np.unique(train_labels)
	print(list_labels)
	all_train_patients = []
	all_train_labels = []
	all_valid_patients = []
	all_valid_labels = []
	
	for lbl in list_labels:
		list_patients = []
		list_labels = []
		for idx in range(len(train_labels)):
			if train_labels[idx] == lbl:
				list_patients.append(train_patients[idx])
				list_labels.append(train_labels[idx])
		train_p, train_l, valid_p, valid_l = split_patients(list_patients, 
															list_labels, 
															val_per = val_per, 
															seed = seed)
		print("Train: ", len(train_l), ", Valid: ", len(valid_l))
		all_train_patients += list(train_p)
		all_train_labels += list(train_l)
		
		all_valid_patients += list(valid_p)
		all_valid_labels += list(valid_l)
		

	#print("Traing/valid: ", len(all_train_labels), len(all_valid_labels))
	#train_valid_dict = {}
	#train_valid_dict['train'] = (all_train_patients, all_train_labels, all_train_clinicals)
	#train_valid_dict['val'] = (all_valid_patients, all_valid_labels, all_valid_clinicals)
	train_transf = None
	train_dataset = Patient(root_folder,
							all_train_patients, 
							all_train_labels,
							patients_wsi_dict,
							transf = train_transf, 
							n_tiles = n_tiles,
							seed = seed)

	train_dataloader = torch.utils.data.DataLoader(train_dataset, 
													batch_size = batch_size, 
													num_workers=20,)
	# Create data loader
	valid_transf = None
	valid_dataset = Patient(root_folder,
							all_valid_patients, 
								all_valid_labels,
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
########################################### End Cross-Validation ####################################################