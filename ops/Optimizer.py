import torch

class Optimizer:

	def __init__(self):

		self.opti = {
			"Adam" : torch.optim.Adam,
			"SGD"  : torch.optim.SGD,
			"Adadelta" : torch.optim.Adadelta,
			"Adagrad" : torch.optim.Adagrad,
			"RMSprop" : torch.optim.RMSprop
		}
	def call(self,option):
		if option in self.cri.keys():
			return opti[option](model.parameters(), lr=learning_rate) 
		else:
			print("try one of these optimizer: ",opti.keys()," Taking Adam as default")
			return torch.optim.Adam(model.parameters(), lr=learning_rate) 
