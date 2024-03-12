import os
import torch
import torch.utils.data as data_utils
from torch.nn import functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
import sklearn.metrics as sm
from sklearn.calibration import calibration_curve
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

from model_attention_survival import NN_Model2aplus
#from model_survivals_other import NN_Model3a, NN_Model3b
import bergonie_dataloader_survival_wsi as pro

from datetime import datetime

class DeepNN_Model(pl.LightningModule):
    def __init__(self, hparams, train_loader = None, valid_loader = None):
        super().__init__()
        
        self.save_hyperparameters(hparams)
        print("global seed: ", os.environ.get("PL_GLOBAL_SEED"))
        #pl.seed_everything(hparams.seed)
        
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
            
            # self.train_loader, self.valid_loader,_ = prop.get_data_npz_folder_patient_level(hparams.npz_train, 
            #                                                 hparams.npz_labels, 
            #                                                 tmp_folder = hparams.tmp_folder,
            #                                                 n_tiles = hparams.n_tiles,
            #                                                 batch_size = hparams.batch_size, 
            #                                                 val_per = hparams.val_per,
            #                                                 seed = hparams.seed)

        self.model = NN_Model2aplus(in_features = hparams.init_features,
                    fc_1= hparams.fc_1, 
                    fc_2 = hparams.fc_2, 
                    fc_output = hparams.num_classes)
        # self.model = NN_Model3a(in_features = hparams.init_features,
        #             fc_1= hparams.fc_1, 
        #             fc_2 = hparams.fc_2, 
        #             fc_output = hparams.num_classes)

        self.loss_f = torch.nn.BCEWithLogitsLoss()
        self.iter = 0
        self.lbl_pred_each = []
        self.survtime_all = []
        self.status_all = []
        self.automatic_optimization = False
        self.cpt_loss = 16

    # Delegate forward to underlying model
    def forward(self, x):
        y_prob, x_att = self.model(x)
        return y_prob, x_att

    def training_step(self, batch, batch_idx):
        x, y_status = batch
        y_prob, x_att = self.forward(x)

        if self.iter == 0:
            self.lbl_pred_each = y_prob
            self.status_all = y_status
        else:
            self.lbl_pred_each = torch.cat([self.lbl_pred_each, y_prob])
            self.status_all = torch.cat([self.status_all, y_status])

        self.iter +=1

        if self.iter % self.cpt_loss == 0 or batch_idx == len(self.train_loader) - 1:
            self.status_all = self.status_all.view(self.status_all.size(0), -1)
            
            loss = self.loss_f(self.lbl_pred_each, self.status_all)
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss, retain_graph=True)
            opt.step()

            torch.cuda.empty_cache()
            self.lbl_pred_each = []
            self.status_all = []
            self.iter = 0
            self.log('train_loss', loss)

            return {'loss': loss,
                    'batch_idx': batch_idx}

    def compute_auc(self, y_true_tensors, y_pred_tensors):
        y_pred_tensors = [torch.sigmoid(y_prob.float()) for y_prob in y_pred_tensors]
        y_hats = [torch.ge(y_sigmoid, 0.5).float() for y_sigmoid in y_pred_tensors]
        #preds = [y_hat.cpu().numpy().reshape(-1).tolist() for y_hat in y_hats]
        preds = []
        for y_hat in y_hats:
            preds = preds + y_hat.cpu().numpy().reshape(-1).tolist()

        targets = []
        for y in y_true_tensors:
            targets = targets + y.cpu().numpy().reshape(-1).tolist()

        
        acc_score = sm.balanced_accuracy_score(targets, preds)
        f1_score = sm.f1_score(targets, preds)

        #y_probs = [ypr.cpu().numpy().reshape(-1) for ypr in y_pred_tensors]
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

        tn, fp, fn, tp = sm.confusion_matrix(targets, preds).ravel()
        spec = tn/(tn + fp)
        sen = tp/(tp + fn)

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
        return torch.as_tensor(roc_auc), \
                torch.as_tensor(acc_score), \
                torch.as_tensor(f1_score), \
                torch.as_tensor(brief_score), \
                torch.as_tensor(spec), \
                torch.as_tensor(sen), \
    
    # Validate on one batch
    
    def validation_step(self, batch, batch_idx):
        x, y_status = batch
        y_prob, x_att = self.forward(x)

        if self.iter == 0:
            self.lbl_pred_each = y_prob
            self.status_all = y_status
        else:
            self.lbl_pred_each = torch.cat([self.lbl_pred_each, y_prob])
            self.status_all = torch.cat([self.status_all, y_status])

        self.iter +=1

        if batch_idx == len(self.valid_loader) - 1:
            self.status_all = self.status_all.view(self.status_all.size(0), -1)
            loss = self.loss_f(self.lbl_pred_each, self.status_all)

            auc_score, acc_score, f1_score, brief_score, spec, sen = self.compute_auc(self.status_all, self.lbl_pred_each)

            torch.cuda.empty_cache()
            self.lbl_pred_each = []
            self.status_all = []
            self.iter = 0
            
            return {'val_loss': loss,
                    'val_auc': auc_score,
                    'val_acc': acc_score,
                    'val_f1': f1_score,
                    'val_brief': brief_score,
                    'val_sen': sen,
                    'val_spec': spec,
                    'batch_idx': batch_idx} 

    def validation_epoch_end(self, outputs):
        print(outputs)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_auc = torch.stack([x['val_auc'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        avg_brief = torch.stack([x['val_brief'] for x in outputs]).mean()
        avg_spec = torch.stack([x['val_spec'] for x in outputs]).mean()
        avg_sen = torch.stack([x['val_sen'] for x in outputs]).mean()

        tensorboard_logs = {'loss': avg_loss, 
                            'auc':avg_auc,
                            'acc': avg_acc,
                            'f1': avg_f1,
                            'sensitivity': avg_sen,
                            'specificility': avg_spec,
                            'brier_score': avg_brief}

        self.log('val_loss',avg_loss)
        #self.log('val_cindex',cindex)
        self.log('val_auc',avg_auc)
        self.log('val_acc', avg_acc)
        self.log('val_f1', avg_f1)
        self.log('val_brier_score', avg_brief)
        self.log('val_sensitivity', avg_sen)
        self.log('val_specificility', avg_spec)

        sch = self.lr_schedulers()
        sch.step(self.trainer.callback_metrics['val_loss'])

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # Setup optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr = self.hparams.learning_rate,
                                    weight_decay = self.hparams.w_decay)
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


