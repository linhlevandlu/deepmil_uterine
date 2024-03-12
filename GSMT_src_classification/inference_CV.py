import sys
import os
import torch.nn
import torch 
from torchvision import transforms

from model_attention_survival import NN_Model2aplus
from model_survivals_other import NN_Model3a, NN_Model3b
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import auc, RocCurveDisplay
from sklearn.utils import shuffle
from PIL import Image
from datetime import datetime
import pickle
import csv
import sklearn.metrics as sm

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

def inference_validation(ckpt_path, root_folder, labels_dict, patient_wsi_dict, bags = 1, threshold = 0.5):
    checkpoint = torch.load(ckpt_path)
    #print(checkpoint)
    for key in list(checkpoint['state_dict'].keys()):
    	new_key = key[6:]
    	checkpoint['state_dict'][new_key] = checkpoint['state_dict'].pop(key)

    model = NN_Model3b(fc_1= 256, fc_2 = 128, fc_output = 1)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print("List of patients:")
    patient_list = list(labels_dict)
    print(FOLD_IDX + ' = ',patient_list)

    predictions = []
    predict_y_hat = []
    y_trues = []
    #surv_funcs = []
    for ptx in range(len(patient_list)):
        patient = patient_list[ptx]
        label = labels_dict[patient]
        y_trues.append(label)

        if bags == 1:
            wsi_file = _setup_bag_2(root_folder, patient_wsi_dict, patient, n_tiles = 10000, seed = 2452)
            input_tensor = torch.from_numpy(wsi_file)
            input_tensor = input_tensor.unsqueeze(0)
            inf_patient = input_tensor
        
           # predict
            y_prob, x_att = model(inf_patient)
            y_sigmoid = torch.sigmoid(y_prob.float())
            y_hat = torch.ge(y_sigmoid, threshold).float().squeeze().item()
            y_sigmoid = y_sigmoid.squeeze().item()
        elif bags > 1:
            y_sigmoid = inference_bags(model,
                            root_folder, 
                            patient_wsi_dict,
                            patient,
                            n_bags = bags,
                            n_tiles = 10000,
                            seed = 2452)
            y_hat = int(y_sigmoid >= threshold)

        predictions.append(y_sigmoid)
        predict_y_hat.append(y_hat)

    return predictions, predict_y_hat, y_trues

def read_csv(csv_file, wsi_folder):
    # Read WSI folder
    list_files = sorted(list(os.listdir(wsi_folder)))
    # Read CSV file
    file = open(csv_file)
    csvreader = csv.reader(file)
    header = next(csvreader)

    label_patients_dict = {}
    patients_wsi_dict = {}
    for row in csvreader:
        name, label = row
        label_patients_dict[name] = int(label)
        wsi_files = [f_name for f_name in list_files if name in f_name]
        patients_wsi_dict[name] = wsi_files

    file.close()
    return label_patients_dict, patients_wsi_dict

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

def _compute_metrics(y_probs, y_hats, y_trues):
    acc_score = sm.balanced_accuracy_score(y_trues, y_hats)
    f1_score = sm.f1_score(y_trues, y_hats)
    print("Balanced ACC: ", acc_score)
    print('F1 score: ', f1_score)

    fpr, tpr, _ = sm.roc_curve(y_trues, y_probs)
    roc_auc = sm.auc(fpr, tpr)
    print('AUC score: ', roc_auc)
    # ROC Curve
    # fig, ax = plt.subplots()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonale
    # plt.legend()
    # plt.show()

    # Confusion matrix
    cm = sm.confusion_matrix(y_trues, y_hats)
    print(cm)
    print(cm.ravel())
    tn, fp, fn, tp = cm.ravel()
    spec = tn/(tn + fp)
    sen = tp/(tp + fn)
    print('Sensitivity score: ', sen)
    print('Specificity score: ', spec)
    return acc_score, roc_auc

def read_ckpt(ckpt_file, wsi_folder):
    list_files = sorted(list(os.listdir(wsi_folder)))

    valid_data = torch.load(ckpt_file)
    valid_dict = valid_data[FOLD_IDX]
    valid_patients, valid_labels = valid_dict[0], valid_dict[1]

    assert(len(valid_patients) == len(valid_labels))
    
    label_patients_dict = {}
    patients_wsi_dict = {}
    
    for idx in range(len(valid_patients)):
        ptx_name = valid_patients[idx]
        label = valid_labels[idx]

        label_patients_dict[ptx_name] = int(label)
        wsi_files = [f_name for f_name in list_files if ptx_name in f_name]
        patients_wsi_dict[ptx_name] = wsi_files

    return label_patients_dict, patients_wsi_dict

def load_patient_dict_and_inference(ckpt_path, root_folder, csv_patient):
    ext = csv_patient.split('.')[-1]
    if ext == 'csv':
       label_patients_dict, patients_wsi_dict = read_csv(csv_patient, root_folder)
    elif ext == 'ckpt':
       label_patients_dict, patients_wsi_dict = read_ckpt(csv_patient, root_folder)
       
    y_preds, y_hats, y_trues  = inference_validation(ckpt_path, 
                                            root_folder, 
                                            label_patients_dict,
                                            patients_wsi_dict,
                                            bags = 1,
                                            threshold = 0.5)
    
    print(FOLD_IDX + '_t =', y_trues)
    print(FOLD_IDX + '_yhat =', y_hats)
    print('Total patients: ', len(y_trues))
    print(_compute_metrics(y_preds, y_hats, y_trues))

root = '/beegfs/vle/Sabrina_Croce/GSMT_Classification/lightning_logs_MIDL/CV_5CV_MinMax_noclinical_noSTUMP/default/'
folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
ckpt_list = ['version_0/checkpoints/epoch=25-step=6057.ckpt',
            'version_1/checkpoints/epoch=61-step=14445.ckpt',
            'version_2/checkpoints/epoch=20-step=4913.ckpt',
            'version_3/checkpoints/epoch=17-step=4211.ckpt',
            'version_4/checkpoints/epoch=16-step=3977.ckpt']
#testing_patients = '/beegfs/vle/Sabrina_Croce/GSMT_Classification/exports/v5/random_SEED_no_clinical_2452_model2aplus_CV_5folds_patient_level_skfold_5_folds_20231204094517.ckpt'
#root_folder = '/beegfs/vle/Sabrina_Croce/GSMT_data/tiles_features'
testing_patients = '/beegfs/vle/Sabrina_Croce/GSMT_data/v6_MIDL_csv/external_cohort/ex_classification_2classes_withSTUMP_no_clinical.csv'
root_folder = '/beegfs/vle/Sabrina_Croce/GSMT_data/external_cohort/tiles_features'

print(root)
print(testing_patients)
for idx in range(len(folds)):
    FOLD_IDX = folds[idx]
    ckpt_path = root + ckpt_list[idx]
    print(FOLD_IDX, ckpt_list[idx])
    load_patient_dict_and_inference(ckpt_path, root_folder, testing_patients)
print("Finish !!!!")

