import sys
import os
import torch.nn
import torch 
from torchvision import transforms

from model_attention_survival import NN_Model2aplus_Clinical, NN_Model2aplus
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import auc, RocCurveDisplay
from sklearn.utils import shuffle
from PIL import Image
from datetime import datetime
import pickle
import csv
import sklearn.metrics as sm
from loss_2 import c_index_np
from predictor import Predictor

def inference_bags(model, root_folder, patient_wsi_dict, patient, 
                   n_bags = 5, n_tiles = 10000, seed = 2452):
    y_probs = []
    r_state = np.random.RandomState(seed)
    list_wsis = patient_wsi_dict[patient]
    all_tiles = _load_wsis(root_folder,list_wsis)
    for idx in range(n_bags):
        indices0 = r_state.randint(0, all_tiles.shape[0], n_tiles)
        bag_i = all_tiles[indices0,:]
        print('Bag ' , idx, 'Shape: ', all_tiles.shape, bag_i.shape)
        input_tensor = torch.from_numpy(bag_i)
        input_tensor = input_tensor.unsqueeze(0)
        inf_patient = input_tensor

        # predict
        y_prob, _ = model(inf_patient)
        y_sigmoid = torch.sigmoid(y_prob.float())
        
        y_probs.append(y_sigmoid.squeeze().item())
    print(y_probs)
    return sum(y_probs)/ len(y_probs)

def inference_validation(predictor, root_folder, labels_dict, patients_clinical, patient_wsi_dict, bags = 1, threshold = 0.5):
    output_intervals = torch.arange(0., 31., 1.).to('cuda:0')
    print("List of patients:")
    patient_list = list(labels_dict)
    print(FOLD_IDX + ' = ',patient_list)

    predictions = []
    #predict_y_hat = []
    #y_trues = []
    #surv_funcs = []
    for ptx in range(len(patient_list)):
        patient = patient_list[ptx]
        #label = labels_dict[patient]
        #y_trues.append(label)

        if bags == 1:
            #wsi_file, wsi_labels = _read_npz(root_folder, patient)
            wsi_file = _setup_bag_2(root_folder, patient_wsi_dict, patient, n_tiles = 10000, seed = 2452)
            input_tensor = torch.from_numpy(wsi_file)
            input_tensor = input_tensor.unsqueeze(0)
            inf_patient = input_tensor
            clinical = patients_clinical[patient]
            clinical_tensor = torch.from_numpy(clinical)
            #clinical_tensor = clinical_tensor.to(torch.float32)
            clinical_tensor = clinical_tensor.unsqueeze(0)
    	       
    	   # predict
            surv_probs, x_att = predictor.predict((inf_patient, clinical_tensor),
                                                    prediction_year = None,
                                                    intervals = output_intervals,
                                                    nb_clinical = 0)
            
        elif bags > 1:
            y_sigmoid = inference_bags(model,
                            root_folder, 
                            patient_wsi_dict,
                            patient,
                            n_bags = bags,
                            n_tiles = 10000,
                            seed = 2452)
            y_hat = int(y_sigmoid >= threshold)

        predictions.append(surv_probs)
        #predict_y_hat.append(y_hat)

    return predictions#, predict_y_hat, y_trues

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

def read_row_noGT(row, nb_clinicals = 1):
    if nb_clinicals == 0:
        name= row[0]
        cli_list = []
    elif nb_clinicals == 1:
        name, mit = row
        cli_list = [float(mit)]
    elif nb_clinicals == 2:
        name, mit, tc = row
        cli_list = [float(mit), float(tc)]
    elif nb_clinicals == 3:
        name, mit, tc, at = row
        cli_list = [float(mit), float(tc), float(at)]
    elif nb_clinicals == 4:
        name, mit, tc, at, lvi = row
        cli_list = [float(mit), float(tc), float(at), float(lvi)]
    return name, cli_list, 0, 0.0

def read_csv(csv_file, wsi_folder, with_GT = False):
    print("Testing patients from CSV")
    # Read WSI folder
    list_files = sorted(list(os.listdir(wsi_folder)))
    # Read CSV file
    file = open(csv_file)
    csvreader = csv.reader(file)
    header = next(csvreader)
    if with_GT:
        nclinical = len(header) - 3
    else:
        nclinical = len(header) - 1
    print("Number of clinicals: ", nclinical)
    
    label_patients_dict = {}
    patients_wsi_dict = {}
    patients_clinical = {}
    surv_times = []
    patients = []
    patient_outcomes = {}
    for row in csvreader:
        if with_GT:
            name, label, time, cli_list = read_row(row, nb_clinicals = nclinical)
        else:
            name, cli_list, label, time = read_row_noGT(row, nb_clinicals = nclinical)
        
        surv_times.append(float(time))
        patients.append(name)
        #patients[name] = int(label)
        wsi_files = [f_name for f_name in list_files if name in f_name]
        patients_wsi_dict[name] = wsi_files
        
        patients_clinical[name] = np.array(cli_list)
        patient_outcomes[name] = (int(label), float(time))

    file.close()

    return patients, patients_wsi_dict, patients_clinical, patient_outcomes

def _read_npz(npz_file, label = 0):
    npz_array = np.load(npz_file)['arr_0']
    return npz_array

def _load_wsis(root_folder, list_wsis, seed = 2452):
    all_wsi = []
    for wsi in list_wsis:
        wsi_path = os.path.join(root_folder, wsi)
        c_files= _read_npz(wsi_path)
        all_wsi.append(c_files)
    all_wsi = np.concatenate(all_wsi, axis = 0)
    wsi_file= shuffle(all_wsi, random_state = seed)
    return all_wsi

def _setup_bag_2(root, patient_wsi_dict, pname, n_tiles = 10000, seed = 2452): 
    #print("Get bag of {}".format(self.region))
    list_wsis = patient_wsi_dict[pname]
    wsi_file = _load_wsis(root, list_wsis)
    selected_tiles = random_select_tiles_wsi(wsi_file, num_tiles = n_tiles)
    #print('Shape: ', wsi_file.shape, selected_tiles.shape)

    return selected_tiles

def random_select_tiles_wsi(np_array, num_tiles = 10000):
    n_rows = np_array.shape[0]
    rd_indices = np.random.choice(n_rows, size = num_tiles, replace = True)
    selected_array = np_array[rd_indices,:]

    return selected_array

def read_ckpt(ckpt_file, wsi_folder):
    list_files = sorted(list(os.listdir(wsi_folder)))

    valid_data = torch.load(ckpt_file)
    valid_dict = valid_data[FOLD_IDX]
    valid_patients, valid_labels, valid_clinicals, valid_outcomes = valid_dict[0], valid_dict[1], valid_dict[2], valid_dict[3]

    assert(len(valid_patients) == len(valid_labels))
    
    label_patients_dict = {}
    patients_wsi_dict = {}
    patients_clinical = {}
    patients_outcomes = {}
    for idx in range(len(valid_patients)):
        ptx_name = valid_patients[idx]
        label = valid_labels[idx]
        clinicals = valid_clinicals[idx]
        outcome = valid_outcomes[idx]

        label_patients_dict[ptx_name] = int(label)
        wsi_files = [f_name for f_name in list_files if ptx_name in f_name]
        patients_wsi_dict[ptx_name] = wsi_files

        patients_clinical[ptx_name] = np.array(clinicals).astype(np.float32)
        patients_outcomes[ptx_name] = outcome
    return label_patients_dict, patients_wsi_dict, patients_clinical, patients_outcomes

def load_model(ckpt_path):
    # Load model and create the predictor
    checkpoint = torch.load(ckpt_path)
    for key in list(checkpoint['state_dict'].keys()):
        new_key = key[6:]
        checkpoint['state_dict'][new_key] = checkpoint['state_dict'].pop(key)
    model = NN_Model2aplus(fc_1= 256, fc_2 = 128, fc_output = 30)
    #model = NN_Model2aplus_Clinical(n_clinical = 3, fc_1= 256, fc_2 = 128, fc_output = 30)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def pred_at_time_point(list_tensor_preds, time_point = 1):
    '''
        Get the survival probability at a time points from the predictions
    '''
    #np_preds = tensor_preds.cpu().numpy()
    all_patients_probs = []
    all_patients_at_one_point = []
    for pred in list_tensor_preds:
        pred = pred.numpy()
        all_patients_probs.append(pred)
        all_patients_at_one_point.append(pred[time_point])
    return all_patients_probs, all_patients_at_one_point

def export_to_csv(filename, patients, predictions):
    with open(filename, 'w') as f:
        write = csv.writer(f)
        head = ['Patient']
        for i in range(30):
            head += ['year_' + str(i)]
        write.writerow(head)

        for idx in range(len(patients)):
            patient = patients[idx]
            pred = list(predictions[idx])
            row = [patient] + pred
            write.writerow(row)

def load_patient_dict_and_inference(ckpt_path, root_folder, csv_patient, with_GT = False):
    ext = csv_patient.split('.')[-1]
    if ext == 'csv':
       label_patients_dict, patients_wsi_dict, patients_clinical, patient_outcomes = read_csv(csv_patient, root_folder, with_GT)
    elif ext == 'ckpt':
       label_patients_dict, patients_wsi_dict, patients_clinical, patient_outcomes = read_ckpt(csv_patient, root_folder)
       with_GT = True
    #print(patient_outcomes)

    model = load_model(ckpt_path)
    output_intervals = torch.arange(0., 31, 1.).to('cuda:0')
    predictor = Predictor(model, output_intervals)

    y_preds  = inference_validation(predictor, 
                                            root_folder, 
                                            label_patients_dict,
                                            patients_clinical,
                                            patients_wsi_dict,
                                            bags = 1,
                                            threshold = 0.5)
    
    print('Total patients: ', len(y_preds))
    all_patients_probs, all_patients_one_point = pred_at_time_point(y_preds, time_point = 4)
    #print(all_patients_one_point)
    #print(y_preds)
    if with_GT:
        patient_list = list(label_patients_dict)
        y_trues = []
        y_survtimes = []
        for pat in patient_list:
            surv = patient_outcomes[pat]
            y_trues.append(torch.Tensor([surv[0]]))
            y_survtimes.append(torch.tensor([surv[1]]))

        y_trues = np.array(y_trues, dtype = object)
        y_survtimes = np.array(y_survtimes, dtype = object)
        y_preds = torch.stack(y_preds)
        #y_survtimes = torch.from_numpy(y_survtimes)
        #y_trues = torch.from_numpy(y_trues)
        #y_preds = torch.from_numpy(y_preds)
        cindex = c_index_np(y_preds, y_survtimes, y_trues)
        print("C-index: ", cindex)
    else:
        cindex = -1.
    return list(label_patients_dict), all_patients_one_point, all_patients_probs, cindex
    
root = '/beegfs/vle/Sabrina_Croce/GSMT_Survival/lightning_logs_MIDL/PFS_5CV_no_clinical_noSTUMP_2year/default/'
folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
ckpt_list = ['version_0/checkpoints/epoch=144-step=19429.ckpt',
            'version_1/checkpoints/epoch=110-step=14984.ckpt',
            'version_2/checkpoints/epoch=109-step=14849.ckpt',
            'version_3/checkpoints/epoch=205-step=28015.ckpt',
            'version_4/checkpoints/epoch=135-step=18495.ckpt']
            
testing_patients = '/beegfs/vle/Sabrina_Croce/GSMT_Survival/exports_midl/survival_v2_SEED_2452_model2aplus_PFS_5CV_no_clinical_noSTUMP_2year_skfold_5_folds_20240115171903.ckpt'
#testing_patients = '/beegfs/vle/Sabrina_Croce/GSMT_data/v6_MIDL_csv/survival_pfs_STUMP_no_clinical.csv'
root_folder = '/beegfs/vle/Sabrina_Croce/GSMT_data/tiles_features'
#testing_patients = '/beegfs/vle/Sabrina_Croce/GSMT_data/v6_MIDL_csv/external_cohort/ex_survival_pfs_withSTUMP_clinical_norm_2year.csv'
#root_folder = '/beegfs/vle/Sabrina_Croce/GSMT_data/external_cohort/tiles_features'

print(root)
print(testing_patients)
all_test_patients = []
all_test_surv_prob_preds = []
all_test_surv_prob_preds_one_point = []
all_test_cindex = []
for idx in range(len(folds)):
    FOLD_IDX = folds[idx]
    ckpt_path = root + ckpt_list[idx]
    print(FOLD_IDX, ckpt_list[idx])
    patients, surv_probs_one_point, surv_probs, cindex = load_patient_dict_and_inference(ckpt_path, root_folder, testing_patients, with_GT = True)
    all_test_patients += patients
    all_test_surv_prob_preds_one_point += surv_probs_one_point
    all_test_surv_prob_preds += surv_probs
    all_test_cindex.append(cindex)

# Compute mean value for each time-point
output_intervals = torch.arange(0., 30, 1.)
all_test_surv_prob_preds = np.array(all_test_surv_prob_preds)
mean_probs = [all_test_surv_prob_preds[:,i] for i in range(len(output_intervals))]
mean_probs = np.array(mean_probs).mean(axis = 1)

print(all_test_patients)
print(all_test_surv_prob_preds_one_point)
#print(mean_probs)
print('Average C-index: ', sum(all_test_cindex)/len(all_test_cindex))

export_to_csv('exports_csv_v6_midl/IB_survival_PFS_5CV_no_clinical_noSTUMP_2year.csv', all_test_patients, all_test_surv_prob_preds)
#export_to_csv('exports_csv_v4/DFS_all_35_testing_patients_Ctd_0_clinicals.csv', all_test_patients, all_test_surv_prob_preds)
print("Finish internal testing !!!!")

'''
print("External testing")
testing_patients = '/beegfs/vle/Sabrina_Croce/GSMT_data/external_cohort/csv/survival_35patients_0clinical.csv'
root_folder = '/beegfs/vle/Sabrina_Croce/GSMT_data/external_cohort/tiles_features'
print(root)
print(testing_patients)
all_test_patients = []
all_test_surv_prob_preds = []
all_test_surv_prob_preds_one_point = []
all_test_cindex = []
for idx in range(len(folds)):
    FOLD_IDX = folds[idx]
    ckpt_path = root + ckpt_list[idx]
    print(FOLD_IDX, ckpt_list[idx])
    patients, surv_probs_one_point, surv_probs, cindex = load_patient_dict_and_inference(ckpt_path, root_folder, testing_patients, with_GT = True)
    all_test_patients += patients
    all_test_surv_prob_preds_one_point += surv_probs_one_point
    all_test_surv_prob_preds += surv_probs
    all_test_cindex.append(cindex)

# Compute mean value for each time-point
output_intervals = torch.arange(0., 30, 1.)
all_test_surv_prob_preds = np.array(all_test_surv_prob_preds)
mean_probs = [all_test_surv_prob_preds[:,i] for i in range(len(output_intervals))]
mean_probs = np.array(mean_probs).mean(axis = 1)

print(all_test_patients)
print(all_test_surv_prob_preds_one_point)
#print(mean_probs)
print('Average C-index: ', sum(all_test_cindex)/len(all_test_cindex))
export_to_csv('exports_csv_v4/PFS_all_35_testing_patients_Ctd_0_clinicals_noSTUMP.csv', all_test_patients, all_test_surv_prob_preds)
print("Finish external testing !!!!")
'''
# Test the STUMPs
# print("Testing on the STUMPs of Bergonie testing")
# testing_patients = '/beegfs/vle/Sabrina_Croce/GSMT_data/v4_350_20092023/os_218patients_0clinical_STUMP.csv'
# root_folder = '/beegfs/vle/Sabrina_Croce/GSMT_data/tiles_features'
# print(root)
# print(testing_patients)
# all_test_patients = []
# all_test_surv_prob_preds = []
# all_test_surv_prob_preds_one_point = []
# all_test_cindex = []
# for idx in range(len(folds)):
#     FOLD_IDX = folds[idx]
#     ckpt_path = root + ckpt_list[idx]
#     print(FOLD_IDX, ckpt_list[idx])
#     patients, surv_probs_one_point, surv_probs, cindex = load_patient_dict_and_inference(ckpt_path, root_folder, testing_patients, with_GT = True)
#     all_test_patients += patients
#     all_test_surv_prob_preds_one_point += surv_probs_one_point
#     all_test_surv_prob_preds += surv_probs
#     all_test_cindex.append(cindex)

# # Compute mean value for each time-point
# output_intervals = torch.arange(0., 30, 1.)
# all_test_surv_prob_preds = np.array(all_test_surv_prob_preds)
# mean_probs = [all_test_surv_prob_preds[:,i] for i in range(len(output_intervals))]
# mean_probs = np.array(mean_probs).mean(axis = 1)

# print(all_test_patients)
# print(all_test_surv_prob_preds_one_point)
# #print(mean_probs)
# print('Average C-index: ', sum(all_test_cindex)/len(all_test_cindex))
# export_to_csv('exports_csv_v4/OS_39_STUMP_patients_Ctd_0_clinicals.csv', all_test_patients, all_test_surv_prob_preds)
# print("Finish external testing !!!!")