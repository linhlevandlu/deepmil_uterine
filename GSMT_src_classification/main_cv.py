import sys
import os
print(sys.path)

import math
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import numpy as np
import socket
import torch.distributed as dist
import time
import random
from datetime import datetime

from model_pl import DeepNN_Model

import bergonie_dataloader_survival_wsi as pdata

def cross_validation(hparams):

    config = "CV_5CV_AvgPooling_noclinical_noSTUMP"
    if hparams.m_get_data == 'SKFold':
        print("Get data by SKFold CV.")
        # train_patients_dict, valid_patients_dict = pdata.get_data_cross_validation_SKFold(hparams.npz_labels, 
        #                                                                                             hparams.npz_train,
        #                                                                                             k_folds = hparams.n_folds,
        #                                                                                             n_tiles = hparams.n_tiles,
        #                                                                                             seed = hparams.seed)
        train_patients_dict, valid_patients_dict, patients_wsi_dict = pdata.get_data_cross_validation_SKFold(hparams.npz_labels, 
                                                                                                    hparams.npz_train,
                                                                                                    k_folds = hparams.n_folds,
                                                                                                    seed = hparams.seed)
    
    torch.save(valid_patients_dict, 'exports_midl/no_clinical_{}_model2aplus_{}_skfold_{}_folds_'.format(hparams.seed, config, hparams.n_folds) + datetime.now().strftime("%Y%m%d%H%M%S") + '.ckpt')
    #print(valid_patients_dict)
    assert(len(train_patients_dict) == len(valid_patients_dict))
    
    for idx in range(hparams.n_folds):
        # if idx < 4:
        #    continue
        train_dict = train_patients_dict['fold_' + str(idx)]
        valid_dict = valid_patients_dict['fold_' + str(idx)]
        train_loader_idx, valid_loader_idx = pdata.get_data_from_cross_validation(hparams.npz_train,
                                                                           train_dict,
                                                                           valid_dict, 
                                                                           patients_wsi_dict,
                                                                           n_tiles = hparams.n_tiles, 
                                                                           batch_size = hparams.batch_size, 
                                                                           seed = hparams.seed)

        #train_loader_idx, valid_loader_idx = 1, 1
        model = DeepNN_Model(hparams, train_loader_idx, valid_loader_idx)
        estop_callback = EarlyStopping(monitor = 'val_loss', mode ='min', min_delta = 0.0000, patience = 20, verbose = True)
        chp_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min",)
        #tb_logger = pl_loggers.TensorBoardLogger(save_dir = 'lightning_logs_OS/cv_5_new_R1_TNormal/OS/Muscle/{}/'.format(hparams.area))
        tb_logger = pl_loggers.TensorBoardLogger(save_dir = 'lightning_logs_MIDL/{}/'.format(config))
        trainer = pl.Trainer(max_epochs=hparams.epochs, \
                            weights_summary='top', \
                            num_nodes = hparams.n_nodes, \
                            gpus = hparams.n_gpus, \
                            #accelerator = 'gpu', \
                            #strategy= "ddp", \
                            #amp_level = "O1", \
                            #precision = 16, \
                            callbacks = [chp_callback, estop_callback],
                            num_sanity_val_steps = 0,
                            #replace_sampler_ddp=False,
                            logger = tb_logger,)
        trainer.fit(model, train_loader_idx, valid_loader_idx)
        
def main(hparams):
    # use a random seed
    if hparams.seed == -1:
        hparams.seed = random.randint(0,5000)
        print('The SEED number was randomly set to {}'.format(hparams.seed))

    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)
    random.seed(hparams.seed)

    print(hparams)
    torch.cuda.empty_cache()
    cross_validation(hparams)

if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    main_arg_parser = argparse.ArgumentParser(description="parser for observation generator", add_help=False)
    main_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    main_arg_parser.add_argument("--checkpoint-interval", type=int, default=500,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    main_arg_parser.add_argument('--epochs', default = 500, type=int)
    main_arg_parser.add_argument('--n_nodes', default = 1, type=int)
    main_arg_parser.add_argument('--n_gpus', default = 1, type=int)
    main_arg_parser.add_argument('--learning_rate', default=0.001, type=float)
    main_arg_parser.add_argument('--w_decay', default=1e-4, type=float)
    main_arg_parser.add_argument('--batch_size', default=1, type=int)
    main_arg_parser.add_argument('--n_folds', default= 5, type=int)

    #main_arg_parser.add_argument('--npz_train', default = '/media/monc/LaCie10TB/Sabrina/Tiles_20X/tiles_features', type=str)
    main_arg_parser.add_argument('--npz_train', default = '/beegfs/vle/Sabrina_Croce/GSMT_data/tiles_features', type=str) # Clusters_features Centroids_77 selection
    #
    # classification_2classes_noSTUMP_clinical.csv                                          
    # classification_2classes_noSTUMP_clinical_norm.csv                                     
    # classification_2classes_noSTUMP_no_clinical.csv
    #
    main_arg_parser.add_argument('--npz_labels', default = '/beegfs/vle/Sabrina_Croce/GSMT_data/v6_MIDL_csv/classification_2classes_noSTUMP_no_clinical.csv', type=str)
    main_arg_parser.add_argument('--init_features', default = 2048, type=int)
    main_arg_parser.add_argument('--m_get_data', default = 'SKFold', type=str) # KFold or SKFold

    main_arg_parser.add_argument('--seed', default = 2452, type=int)
    main_arg_parser.add_argument('--n_tiles', default = 10000, type=int)

    main_arg_parser.add_argument('--fc_1', default = 256, type=int)
    main_arg_parser.add_argument('--fc_2', default = 128, type=int)
    main_arg_parser.add_argument('--num_classes', default = 1, type=int) 
    
    # add model specific args i
    parser = DeepNN_Model.add_model_specific_args(main_arg_parser, os.getcwd())
    hyperparams = parser.parse_args()

    main(hyperparams)

