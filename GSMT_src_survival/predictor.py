import os
import sys
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

class Predictor:
	def __init__(self, model, intervals = None , device = 'cuda:0'):
		self.model = model
		self.device = device
		self.model = self.model.to(self.device)
		self.intervals = intervals

	def _convert_to_survival(self, conditional_probs):
		return np.cumprod(conditional_probs)

	def predict(self, patient_data, prediction_year = None, intervals = None, nb_clinical = 4):
		# Predict patient survival probability at a provided time point
		wsi_tensor, clinical_tensor = patient_data
		wsi_tensor = wsi_tensor.to(self.device)
		clinical_tensor = clinical_tensor.to(self.device)
		self.model.eval()
		with torch.set_grad_enabled(False):
			if nb_clinical > 0:
				probs, x_attention = self.model(wsi_tensor, clinical_tensor)
			else:
				probs, x_attention = self.model(wsi_tensor)
		survival_prob = self._convert_to_survival(probs.cpu())
		
		if prediction_year is not None:
			survival_prob = np.interp(prediction_year, intervals.cpu(), torch.cat((torch.tensor([1]).float(), survival_prob)))
		return survival_prob, x_attention

	def format_output_intervals(self, intervals):
		# Convert intervals to time points for plotting
		# parameters: interval = torch.Tensor
		time_points = np.array(intervals.cpu())
		time_points[1:] = time_points[1:] - np.diff(time_points)[0] / 2
		return time_points

	def plot_survival_curve_a_patient(self, surv_probs, intervals, save = False, file_name = None):
		fig = plt.figure(figsize = (4,2.75))
		ax = fig.add_subplot(1,1,1)

		survival_time_points = self.format_output_intervals(intervals)
		ax.plot(survival_time_points[:20], surv_probs[:20], 'o-', label = file_name.replace('.pdf',''))
		ax.set_ylim(0,1.1)
		ax.set_xlim(None, 21)
		ax.legend(loc = 'center left')
		ax.set_xlabel("Time in years")
		ax.set_ylabel("Survival probability")
		if save and file_name != None:
			file_name += '.pdf'
			fig.savefig(file_name, dpi = 300, bbox_inches = 'tight', transparent = True)

	def plot_survival_curve_all_patients(self, patients, all_surv_probs, intervals, save = False):
		fig = plt.figure(figsize = (25,20))
		ax = fig.add_subplot(1,1,1)

		survival_time_points = self.format_output_intervals(intervals)
		for idx in range(len(all_surv_probs)):
			surv_probs = all_surv_probs[idx]
			p_label = 'Patient = ' + patients[idx]
			ax.plot(survival_time_points[:20], surv_probs[:20], 'o-', label = p_label)

		ax.set_ylim(0,1.1)
		ax.set_xlim(None, 21)
		ax.legend(loc = 'center left')
		ax.set_xlabel("Time in years")
		ax.set_ylabel("Survival probability")
		if save:
			file_name = 'exports_pdf/test_STUMP_141patients_3clinical_test_MIT_TC_AT_inference.pdf'
			fig.savefig(file_name, dpi = 300, bbox_inches = 'tight', transparent = True)