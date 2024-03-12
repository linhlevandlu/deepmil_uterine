import os
import torch
import torch.utils.data as data_utils
from torch.nn import functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
import sklearn.metrics as sm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

from model_attention_survival import NN_Model2aplus, NN_Model2aplus_Clinical

import bergonie_dataloader_survival_wsi_clinical as pro
from loss_2 import Survival_Loss, c_index, integrated_brier_score, c_index_td
from sksurv.metrics import cumulative_dynamic_auc
from datetime import datetime


class DeepNN_Model(pl.LightningModule):
    def __init__(self, hparams, train_loader = None, valid_loader = None):
        super().__init__()
        
        self.save_hyperparameters(hparams)
        print("global seed: ", os.environ.get("PL_GLOBAL_SEED"))
        self.nb_clinical = hparams.nb_clinical
        #pl.seed_everything(hparams.seed)
        # Create the model
        if hparams.nb_clinical == 0:
            self.model = NN_Model2aplus(in_features = hparams.init_features,
                    fc_1= hparams.fc_1, 
                    fc_2 = hparams.fc_2, 
                    fc_output = hparams.num_classes)
        else:
            self.model = NN_Model2aplus_Clinical(in_features = hparams.init_features,
                    n_clinical = hparams.nb_clinical,
                    fc_1= hparams.fc_1, 
                    fc_2 = hparams.fc_2, 
                    fc_output = hparams.num_classes)

        # if hparams.pretrained != None:
        #     print("Load the model from a pre-trained model")
        #     self.model = self.load_model(self.model, hparams.pretrained)

        # if hparams.frozen:
        #     print("Freeze the layers")
        #     self.model = self.freeze_model(self.model)

        ## DataLoader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        if self.train_loader == None and self.valid_loader == None:
            self.train_loader, self.valid_loader = pro.get_data_by_patient(hparams.npz_train, 
                                                            hparams.npz_labels, 
                                                            n_tiles = hparams.n_tiles,
                                                            batch_size = hparams.batch_size, 
                                                            val_per = hparams.val_per,
                                                            seed = hparams.seed)
            
            #self.train_loader, self.valid_loader,_ = prop.get_data_npz_folder_patient_level(hparams.npz_train, 
            #                                                 hparams.npz_labels, 
            #                                                 tmp_folder = hparams.tmp_folder,
            #                                                 n_tiles = hparams.n_tiles,
            #                                                 batch_size = hparams.batch_size, 
            #                                                 val_per = hparams.val_per,
            #                                                 seed = hparams.seed)

            # To save the list of validation patients
            #torch.save(self.valid_patients_dict, 'exports_v4_finetuning/validation_patients_freeze_{}_'.format(hparams.frozen) + datetime.now().strftime("%Y%m%d%H%M%S") + '.ckpt')
        

        self.output_intervals = torch.arange(0., hparams.num_classes + 1, 1.).to('cuda:1')
        self.loss_f = Survival_Loss()
        self.iter = 0
        self.lbl_pred_each = []
        self.survtime_all = []
        self.status_all = []
        self.automatic_optimization = False
        

    def load_model(self, model, pretrained_chkpoint):
        checkpoint = torch.load(pretrained_chkpoint)
        print(checkpoint['state_dict'].keys())
        for key in list(checkpoint['state_dict'].keys()):
            new_key = key[6:]
            checkpoint['state_dict'][new_key] = checkpoint['state_dict'].pop(key)    
        
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def freeze_model(self, model):
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        return model

    # Delegate forward to underlying model
    def forward(self, x, x_clinical):
        if self.nb_clinical == 0:
            results_dict = self.model(x)
            y_prob = results_dict['logits']
            x_att = None
        else:
            y_prob, x_att = self.model(x, x_clinical)
        return y_prob, x_att

    def training_step(self, batch, batch_idx):
        x, y_status, y_survtime, x_clinical = batch
        if self.nb_clinical == 0:
            #y_prob, x_att = self.model(x)
            results_dict = self.model(x)
            y_prob = results_dict['logits']
        else:
            y_prob, x_att = self.forward(x, x_clinical)

        # (risk_pred, y, e, model)
        loss = self.loss_f(y_prob, y_survtime, y_status, self.output_intervals)
        #self.manual_backward(loss, retain_graph=True)

        tensorboard_logs = {'train_loss':loss}
        self.log('train_loss', loss)

        return {'loss': loss, 
                'log': tensorboard_logs,
                'time': y_survtime,
                'status':y_status,
                'y_pred': y_prob, 
                'batch_idx': batch_idx}
    
    def training_step_end(self, train_step_output):
        #print(train_step_output)
        y_survtime = train_step_output['time']
        y_status = train_step_output['status']
        y_prob = train_step_output['y_pred']
        batch_idx = train_step_output['batch_idx']

        if self.iter == 0:
            self.lbl_pred_each = y_prob
            self.status_all = y_status
            self.survtime_all = y_survtime
        else:
            self.lbl_pred_each = torch.cat([self.lbl_pred_each, y_prob])
            self.status_all = torch.cat([self.status_all, y_status])
            self.survtime_all = torch.cat([self.survtime_all, y_survtime])

        self.iter +=1

        if self.iter % 32 == 0 or batch_idx == len(self.train_loader) - 1:
        #if batch_idx == len(self.train_loader) - 1:
            #print(self.lbl_pred_each.size(), self.status_all.size())
            #self.survtime_all = torch.stack(self.survtime_all)
            self.survtime_all = self.survtime_all.view(self.survtime_all.size(0),-1)
            #self.status_all = torch.stack(self.status_all)
            self.status_all = self.status_all.view(self.status_all.size(0), -1)
            
            #(risk_pred, y, e, model)
            loss = self.loss_f(self.lbl_pred_each, self.survtime_all, self.status_all, self.output_intervals)
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss, retain_graph=True)
            opt.step()

            torch.cuda.empty_cache()
            self.lbl_pred_each = []
            self.survtime_all = []
            self.status_all = []
            self.iter = 0
            self.log('train_loss', loss)

            return {'loss': loss,
                    'time':y_survtime,
                    'status': y_status,
                    'pred_score': y_prob}

    def compute_auc(self, y_true_tensors, y_pred_tensors):
        y_pred_tensors = [torch.sigmoid(y_prob.float()) for y_prob in y_pred_tensors]
        y_hats = [torch.ge(y_sigmoid, 0.5).float() for y_sigmoid in y_pred_tensors]
        preds = []
        for y_hat in y_hats:
            preds = preds + y_hat.cpu().numpy().reshape(-1).tolist()

        targets = []
        for y in y_true_tensors:
            targets = targets + y.cpu().numpy().reshape(-1).tolist()

        acc_score = sm.balanced_accuracy_score(targets, preds)
        f1_score = sm.f1_score(targets, preds)

        y_probs = []
        for ypr in y_pred_tensors:
            y_probs = y_probs + ypr.cpu().numpy().reshape(-1).tolist()
    
        fpr, tpr, _ = sm.roc_curve(targets, y_probs)
        roc_auc = sm.auc(fpr, tpr)

        # ROC Curve
        fig, ax = plt.subplots()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonale
        plt.legend()
        self.logger.experiment.add_figure("ROC_curve", fig, self.current_epoch)

        # Confusion matrix
        cm = sm.confusion_matrix(targets, preds)
        cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap = plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set_axis_off()
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, "{}\n {}".format(format(cm2[i, j], '.2f'),cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()


            
        self.logger.experiment.add_figure("confusion_matrix", fig, self.current_epoch)
            
        # Calibration curve
        fig, ax = plt.subplots()
        prob_true, prob_pred = calibration_curve(targets, y_probs, n_bins = 10)
        ax.plot(prob_pred, prob_true, marker='o')
        ax.set_xlabel('Average Predicted value')
        ax.set_ylabel('Proportion of positive records')
        ax.plot([0,1], [0,1], linestyle='--')
        self.logger.experiment.add_figure("Calibration curve", fig, self.current_epoch)

        fig, ax = plt.subplots()
        ax.hist(prob_true, bins = 10)
        self.logger.experiment.add_figure("Calibration histogram", fig, self.current_epoch)

        brief_score = sm.brier_score_loss(targets, y_probs)
        # Precision Recall Curve
        #self.logger.experiment.add_pr_curve('pr_curve', targets, y_probs, self.current_epoch)
        return roc_auc, acc_score, f1_score, brief_score
    
    # Validate on one batch
    
    def validation_step(self, batch, batch_idx):
        x, y_status, y_survtime, x_clinical = batch
        if self.nb_clinical == 0:
            #y_prob, x_att = self.model(x)
            results_dict = self.model(x)
            y_prob = results_dict['logits']
        else:
            y_prob, x_att = self.forward(x, x_clinical)

        return {'time': y_survtime,
                'status': y_status,
                'pred_score': y_prob}   

    def validation_epoch_end(self, outputs):
        y_survtimes = [x['time'] for x in outputs]
        y_trues = [x['status'] for x in outputs]
        y_preds = [x['pred_score'] for x in outputs]

        y_survtimes = torch.stack(y_survtimes)
        y_trues = torch.stack(y_trues)
        y_preds = torch.stack(y_preds)

        y_preds = y_preds.view(y_preds.size(0), -1)
        y_trues = y_trues.view(y_trues.size(0), -1)
        y_survtimes = y_survtimes.view(y_survtimes.size(0), -1)

        #print(y_preds.shape, y_survtimes.shape, y_trues.shape)
        avg_loss = self.loss_f(y_preds, y_survtimes, y_trues, self.output_intervals)

        cindex = c_index(y_preds, y_survtimes, y_trues)
        cindex_td = c_index_td(y_preds, y_survtimes, y_trues)

        # To compute IBS scores
        survival_data = []
        for i in range(len(y_trues)):
            e = y_trues[i].cpu()
            t = y_survtimes[i].cpu()
            survival_data.append((e,t))
        survival_data = np.array(survival_data, dtype = np.dtype('bool, float'))
        ibs = integrated_brier_score(survival_data, survival_data, y_preds.cpu()[:,4:8],self.output_intervals.cpu()[4:8])
        _, mean_auc = cumulative_dynamic_auc(survival_data, survival_data, y_preds.cpu()[:,4:8], self.output_intervals.cpu()[4:8])

        tensorboard_logs = {'loss': avg_loss, 
                            'auc': mean_auc,
                            'c-index':c_index,
                            'c-index-td':cindex_td,
                            'ibs': ibs}

        self.log('val_loss',avg_loss)
        self.log('val_ibs',ibs)
        self.log('val_cindex',cindex)
        self.log('val_cindex_td', cindex_td)
        self.log('val_auc', mean_auc)

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # Setup optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr = self.hparams.learning_rate,
                                    weight_decay = self.hparams.w_decay)
        #scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 20, verbose = True)
        lr_schedulers = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                                mode = 'min', 
                                                                                patience = 5, 
                                                                                factor = 0.1, 
                                                                                min_lr = 1e-7, 
                                                                                verbose = True), 
                        "monitor": "val_loss"}
        return [optimizer],[lr_schedulers]
    
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader


    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--name', default='Deep NN model for classification', type=str)
        return parser


