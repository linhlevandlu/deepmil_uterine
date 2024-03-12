import torch
import os, warnings
from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import integrated_brier_score as sk_ibs
import pandas as pd
from pycox.evaluation import EvalSurv

class Survival_Loss(torch.nn.Module):
	def __init__(self):
		super(Survival_Loss, self).__init__()

	def _convert_labels(self, time, event, breaks):
		"""Convert event and time labels to label array.
			Each patient label array has dimensions number of intervals x 2:
			* First half is 1 if patient survived interval, 0 if not.
			* Second half is for non-censored times and is 1 for time interval 
			in which event time falls and 0 for other intervals.
		"""
		#print(time.device)
		n_intervals = len(breaks) - 1
		timegap = breaks[1:] - breaks[:-1]
		breaks_midpoint = breaks[:-1] + 0.5*timegap
		out = torch.zeros(len(time), n_intervals * 2)
		for i, (t,e) in enumerate(zip(time,event)):
			if e: # if not censored, survived time intervals where time >= upper limit
				out[i, 0: n_intervals] = 1.0 * (t >= breaks[1:])
				# if time is greater than end of last time interval, no interval is marked
				if t < breaks[-1]:
					idx = torch.nonzero(t<breaks[1:]).squeeze()
					if idx.shape:
						idx = idx[0]
					out[i, n_intervals + idx] = 1
			else: # if censored, if lived more than half-way through interval, give credit for surviving the interval
				out[i, 0:n_intervals] = 1.0 * (t >= breaks_midpoint)
		
		return out

	def _neg_log_likelihood(self, risk, label, break_list):
		n_intervals = len(break_list) - 1
		all_patients = 1. + label[:,0:n_intervals] * (risk - 1.)
		noncensored = 1. - label[:, n_intervals: 2*n_intervals] * risk
		neg_log_like = -torch.log(torch.clamp(torch.cat((all_patients, noncensored), dim = 1), 1e-07, None))
		return neg_log_like.mean()

	def forward(self, risk, times = None, events = None, breaks = None):
		#print(risk.get_device(), times.get_device(), events.get_device(), breaks.get_device())
		label_array = self._convert_labels(times, events, breaks).to("cuda:1")
		loss = self._neg_log_likelihood(risk, label_array, breaks)
		return loss

def c_index(risk_preds, y, e):
	'''
		- y = survival time
		- e = event (status)
	'''
	probs_by_interval = risk_preds.permute(1,0)
	c_index = [concordance_index(event_times = y.cpu(), predicted_scores = interval_probs.cpu(), event_observed = e.cpu()) for interval_probs in probs_by_interval]
	return c_index[0]

def c_index_np(risk_preds, y, e):
	'''
		- risk_preds: tensor gpu
		- y = survival time (numpy array)
		- e = event (status) (numpy array)
	'''
	probs_by_interval = risk_preds.permute(1,0)
	c_index = [concordance_index(event_times = y, predicted_scores = interval_probs.cpu(), event_observed = e) for interval_probs in probs_by_interval]
	return c_index[0]
	# if not isinstance(risk_pred, np.ndarray):
	# 	risk_pred = risk_pred.detach().cpu().numpy()
	# if not isinstance(y, np.ndarray):
	# 	y = y.detach().cpu().numpy()
	# if not isinstance(e, np.ndarray):
	# 	e = e.detach().cpu().numpy()
	# return concordance_index(y, risk_pred, e)

def integrated_brier_score(survival_train, survival_test, estimation, times):
	return sk_ibs(survival_train, survival_test, estimation, times)

def prediction_to_pycox(predictions, time_points = None):
	predictions = {k: predictions[k] for k in range(len(predictions))}
	df = pd.DataFrame.from_dict(predictions)
	#df = pd.DataFrame(predictions).T

	if time_points is None:
		time_points = torch.arange(0.5,30,1)
	df.insert(0,'time', time_points)
	df = df.set_index('time')
	return df

def c_index_td(risk_preds, y, e):
	'''
		- risk_preds:
		- y = survival time (GT, numpy array)
		- e = survival event (status) (CT, numpy array)
	'''
	#probs_by_interval = risk_preds.permute(1,0)
	probs_by_interval = risk_preds.cpu()
	#print(probs_by_interval.size())
	y = y.cpu().numpy().reshape(-1).astype('float32')
	e = e.cpu().numpy().reshape(-1).astype('float32')
	predicts = prediction_to_pycox(probs_by_interval)
	#print(predicts.shape, y.shape, e.shape)
	ev = EvalSurv(predicts.astype('float32'), y, e, censor_surv = 'km')
	c_index_td = ev.concordance_td('antolini')
	return c_index_td

'''

class Regularization(object):
	def __init__(self, order, weight_decay):
		super(Regularization, self).__init__()
		self.order = order
		self.weight_decay = weight_decay

	def __call__(self, model):
		reg_loss = 0.
		for name, w in model.named_parameters():
			if 'weight' in name:
				reg_loss = reg_loss + torch.norm(w, p = self.order)
		reg_loss = self.weight_decay * reg_loss
		return reg_loss

class NegativeLogLikelihood(torch.nn.Module):
	# https://github.com/czifan/DeepSurv.pytorch/blob/f572a8a7d3ce5ad10609bd273ce3accaf7ea4b66/networks.py#L76
	def __init__(self):
		super(NegativeLogLikelihood, self).__init__()
		self.reg = Regularization(order = 2, weight_decay = 0)# 1e-5

	def forward(self, risk_pred, y, e, model):
		#print(y.shape)
		mask = torch.ones(y.shape[0], y.shape[0])
		mask[(y.T - y) > 0] = 0
		if torch.cuda.is_available():
			mask = mask.cuda()
		log_loss = torch.exp(risk_pred) * mask
		#log_loss = torch.sum(log_loss, dim = 0) / torch.sum(mask, dim = 0)
		log_loss = torch.sum(log_loss, dim = 0)
		log_loss = torch.log(log_loss).reshape(-1,1)
		neg_log_loss = -torch.sum((risk_pred - log_loss) * e)/torch.sum(e)

		l2_loss = self.reg(model)
		return neg_log_loss + l2_loss


def c_index(risk_pred, y, e):
	#	- y = survival time
	#	- e = event (status)
	if not isinstance(risk_pred, np.ndarray):
		risk_pred = risk_pred.detach().cpu().numpy()
	if not isinstance(y, np.ndarray):
		y = y.detach().cpu().numpy()
	if not isinstance(e, np.ndarray):
		e = e.detach().cpu().numpy()
	return concordance_index(y, risk_pred, e)
'''