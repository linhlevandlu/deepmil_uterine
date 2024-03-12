import torch
import torch.nn as nn
from MIL_attention import *
import torch.nn.functional as F

class NN_Model2aplus(nn.Module):
	def __init__(self, 
					in_features = 2048,
					fc_1= 200, 
					fc_2 = 100, 
					fc_output = 1):
		print('Attention model 2a plus')
		super(NN_Model2aplus, self).__init__()
		self.in_fc = in_features
		self.fc1 = fc_1
		self.fc2 = fc_2
		self.output = fc_output

		self.conv_extractor = nn.Sequential(nn.Conv1d(self.in_fc, 1024, 1),
											nn.PReLU(),
											nn.Dropout(p = 0.5),
											nn.Conv1d(1024, 512, 1),
											nn.PReLU(),
											nn.Dropout(p = 0.5))

		self.fc_extractor = nn.Sequential(nn.Linear(512, self.fc1),
											nn.PReLU(),
											nn.Dropout(p = 0.5))

		self.attention = Attention(self.fc1, self.fc2, 1) # GateAttention

		self.classifier = nn.Sequential(nn.Linear(self.fc1 * 1, self.fc2),
										nn.PReLU(),
										nn.Linear(self.fc2, self.fc2//2),
										nn.PReLU(),
										nn.Linear(self.fc2//2, self.output),
										nn.Sigmoid()
										)

	def forward(self, x): 
		x = x.unsqueeze(3)
		x = x.squeeze(0)
		x = self.conv_extractor(x)
		x = x.view(x.size(0), -1)
		#print(x.size())

		x = self.fc_extractor(x)
		#print('x extractor: ',x.size())
		x_attention = self.attention(x)

		x_2 = torch.mm(x_attention, x)
		y_prob = self.classifier(x_2)

		return y_prob, x_attention

class NN_Model2aplus_Clinical(nn.Module):
	def __init__(self, 
					in_features = 2048,
					n_clinical = 1,
					fc_1= 200, 
					fc_2 = 100, 
					fc_output = 1):
		print('Attention model 2a plus clinical')
		super(NN_Model2aplus_Clinical, self).__init__()
		self.in_fc = in_features
		self.fc1 = fc_1
		self.fc2 = fc_2
		self.output = fc_output
		self.n_clinical = n_clinical

		self.conv_extractor = nn.Sequential(nn.Conv1d(self.in_fc, 1024, 1),
											nn.PReLU(),
											nn.Dropout(p = 0.5),
											nn.Conv1d(1024, 512, 1),
											nn.PReLU(),
											nn.Dropout(p = 0.5))

		self.fc_extractor = nn.Sequential(nn.Linear(512, self.fc1),
											nn.PReLU(),
											nn.Dropout(p = 0.5))

		self.attention = Attention(self.fc1, self.fc2, 1) #self.output) # GateAttention

		
		self.clinical_path = nn.Sequential(nn.Linear(self.n_clinical,16),
										nn.ReLU(),
										nn.Linear(16,32),
										)
		self.classifier = nn.Sequential(nn.Linear((self.fc1 * 1) + 32, self.fc2),
										nn.PReLU(),
										nn.Linear(self.fc2, self.fc2//2),
										nn.PReLU(),
										nn.Linear(self.fc2//2, self.output),
										nn.Sigmoid()
										)

	def forward(self, x, x_clinical): 
		x = x.unsqueeze(3)
		x = x.squeeze(0)
		#x = x.to(torch.float16)
		x = self.conv_extractor(x)
		x = x.view(x.size(0), -1)
		#print(x.size())

		x = self.fc_extractor(x)
		x_attention = self.attention(x)
		x_2 = torch.mm(x_attention, x)

		# Clinical
		x_clinical = x_clinical.to(torch.float32)
		x_clinical = self.clinical_path(x_clinical)
		#print(x_clinical.size())
		#print(x_2.size())
		x_cat = torch.cat((x_2 * 0.1, x_clinical * 0.9), dim = 1)

		y_prob = self.classifier(x_cat)

		return y_prob, x_attention
