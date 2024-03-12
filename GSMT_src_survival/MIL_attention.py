import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
	def __init__(self, fc_extractor = 500, fc_middle = 128, fc_output = 1):
		super(Attention, self).__init__()
		self.L = fc_extractor
		self.D = fc_middle
		self.K = fc_output

		self.attention = nn.Sequential(nn.Linear(self.L, self.D),
										nn.Tanh(),
										nn.Linear(self.D, self.K))

	def forward(self, x): # x shape = N x K
		#print(x.shape)
		x = self.attention(x)
		#print(x.shape)
		x = torch.transpose(x,1,0)
		x = F.softmax(x, dim = 1)
		return x

class GateAttention(nn.Module):
	def __init__(self, fc_extractor = 500, fc_middle = 128, fc_output = 1):
		super(GateAttention, self).__init__()
		self.L = fc_extractor
		self.D = fc_middle
		self.K = fc_output

		self.attention_V = nn.Sequential(nn.Linear(self.L, self.D),
										nn.Tanh())
		self.attention_U = nn.Sequential(nn.Linear(self.L, self.D),
										nn.Sigmoid())
		self.attention_weights = nn.Linear(self.D, self.K)

	def forward(self, x): # x shape = N x K
		x_v = self.attention_V(x)
		x_u = self.attention_U(x)
		x_attention = self.attention_weights(x_v * x_u)

		x = torch.transpose(x_attention,1,0)
		x = F.softmax(x, dim = 1)
		return x
