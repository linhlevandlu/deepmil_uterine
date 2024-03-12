import torch
import torch.nn as nn
from MIL_attention import *
import torch.nn.functional as F

class NN_Model3a(nn.Module):
	def __init__(self, 
					in_features = 2048,
					fc_1= 200, 
					fc_2 = 100, 
					fc_output = 1):
		print('Model without attention, avg or max pooling layer')
		super(NN_Model3a, self).__init__()
		self.in_fc = in_features
		self.fc1 = fc_1
		self.fc2 = fc_2
		self.output = fc_output

		self.conv_extractor = nn.Sequential(nn.Conv1d(self.in_fc, 512, 1),
											nn.PReLU(),
											nn.Dropout(p = 0.5),
											nn.Conv1d(512, self.fc1, 1),
											nn.PReLU(),
											nn.Dropout(p = 0.5),
											nn.Conv1d(self.fc1, self.fc2, 1),
											nn.PReLU(),
											)

		#self.pool = nn.AvgPool1d(kernel_size = 10000)
		self.pool = nn.MaxPool1d(kernel_size = 10000)

		self.classifier = nn.Sequential(nn.Dropout(p = 0.5),
			nn.Linear(self.fc2, self.output))

	def forward(self, x): 
		x = x.unsqueeze(3)
		x = x.squeeze(0)
		#print(x.size())
		x = self.conv_extractor(x)
		x = x.squeeze(2)
		x = torch.transpose(x,1,0)
		#print(x.size())

		x = self.pool(x)
		#print('x extractor: ',x.size())
		x = torch.transpose(x,1,0)
		y_prob = self.classifier(x)
		return y_prob, y_prob



class MinMaxLayer(nn.Module):
	def __init__(self, k_min_max = 1, max_only = False):
		super(MinMaxLayer, self).__init__()
		self.k = k_min_max
		self.max_only = max_only

	def forward(self, x_features, x_attention):
		#print(x_features.size(), x_attention.size())
		x_att_1d = x_attention.squeeze()
		indices_sorted = torch.argsort(x_att_1d, descending = True)
		sorted_att_scores = x_attention[indices_sorted, :]
		sorted_features = x_features[indices_sorted, :]
		#print(sorted_features.size(), sorted_att_scores.size())

		if self.max_only:
			selected_att = sorted_att_scores[:self.k, :]
			selected_features = sorted_features[:self.k, :]
			return selected_att, selected_features
		selected_att = torch.cat((sorted_att_scores[:self.k, :],sorted_att_scores[-self.k:, :]))
		selected_features = torch.cat((sorted_features[:self.k,:], sorted_features[-self.k:,:]))
		return selected_att, selected_features

class NN_Model3b(nn.Module):
	def __init__(self, 
					in_features = 2048,
					fc_1= 200, 
					fc_2 = 100, 
					fc_output = 1):
		print('Model without attention and Min-Max layer')
		super(NN_Model3b, self).__init__()
		self.in_fc = in_features
		self.fc1 = fc_1
		self.fc2 = fc_2
		self.output = fc_output

		self.conv_extractor = nn.Sequential(nn.Conv1d(self.in_fc, 512, 1),
											nn.PReLU(),
											nn.Conv1d(512, self.fc1, 1),
											nn.PReLU(),
											nn.Conv1d(self.fc1, self.fc2, 1),
											nn.PReLU(),)

		self.scorer = nn.Sequential(nn.Conv1d(self.fc2, 1, 1),)

		self.k_min_max = 50
		self.max_only = False
		self.k_infc = self.k_min_max
		if not self.max_only:
			self.k_infc *= 2
		self.minmax = MinMaxLayer(k_min_max = self.k_min_max, max_only = self.max_only)

		#self.classifier = nn.Sequential(nn.Linear(self.k_infc * self.output, self.output))
		self.classifier = nn.Sequential(nn.Linear(self.k_infc, self.output))

	def forward(self, x): 
		x = x.unsqueeze(3)
		x = x.squeeze(0)
		#print(x.size())
		x = self.conv_extractor(x)
		#print(x.size())
		x_att = self.scorer(x)

		x_att, x_features = self.minmax(x, x_att)
		#print('x features: ',x_features.size())
		x_att = x_att.unsqueeze(0)
		x_att = x_att.view(x_att.size(0), -1)
		y_prob = self.classifier(x_att)
		return y_prob, y_prob
