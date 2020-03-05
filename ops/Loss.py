from torch import nn
class Loss:
	def __init__(self):
		self.cri = {
			"CrossEntropyLoss" : nn.CrossEntropyLoss,
			"L1Loss" : nn.L1Loss,
			"MSELoss" : nn.MSELoss,
			"NLLLoss": nn.NLLLoss,
			"BinaryCrossEntropyLoss" : nn.BCELoss}
		
	def call(self, option):
		if option in self.cri.keys():
			return self.cri[option]()
		else:
			print("try one of these loss: ",self.cri.keys()," Taking CrossEntropyLoss as default")
			return nn.CrossEntropyLoss()
