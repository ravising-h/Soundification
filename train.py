import os
import librosa
from os.path import join
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Loader.Dataset import SoundCLassification as Emotion
from Model.Model import NeuralNet
import yaml
import logging
from tqdm import tqdm
import sys
from torch.utils.tensorboard import SummaryWriter
import warnings
from datetime import datetime
import time
from ops.Optimizer import Optimizer
from ops.utils import * 
from ops.Loss import Loss
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

warnings.filterwarnings("ignore")

def main(argv):
	with open(argv[0],"r") as f:
	    data = yaml.load(f, Loader=yaml.FullLoader)

	makedir([data["logging"],data["tensorboard"],data["checkpoint"]])

	logging.basicConfig(filename=data["logging"] + str(datetime.now()) + ".log", filemode='w', format='  %(message)s')
	logger(logging,"Loading configuration....")

	with open(argv[0],"r") as f:
	    string = f.read()
	    logger(logging, string)

	writer = SummaryWriter(data["tensorboard"])

	num_epochs = data["num_epochs"]
	batch_size = data["batch_size"]
	learning_rate = data["learning_rate"]
	# Device configuration
	device = data["device"]
	logger(logging, "Loading Model...")

	model = NeuralNet().to(device)
	logger(logging, model.state_dict)
	logger(logging, "Model Loaded")


	logger(logging, "Loading Dataset...")


	train_dataset = Emotion(data["path_train"],classes = data["classes"] , form = data["form"])
	test_dataset =  Emotion(data["path_test"] ,classes = data["classes"] , form = data["form"])


	if data["use_dataloader"]:
		train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
		                                           batch_size=batch_size, 
		                                           num_workers = data["num_workers"],
		                                           shuffle=False)

		test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
		                                          batch_size=batch_size, 
		                                          num_workers = data["num_workers"],
		                                          shuffle=False)
	else:
		for i in tqdm(range(data["total_train_size"])):
			train_loader = train_dataset.__getitem__(i)
		for i in tqdm(range(data["total_test_size"])):
			test_loader = test_dataset.__getitem__(i)


	optimizer = Optimizer().call(data["optimizer"])
	criterion = Loss().call(data["criterion"])

	logger(logging, "Dataset Loaded. Length of dataset is " + str(train_dataset.__len__()))
	total_step = len(train_loader)
	logger(logging, "training Started")
	for epoch in range(num_epochs):
	    logger(logging, "Running {} epoch".format(epoch))
	    running_loss = 0.0
	    for i,(feature, labels) in enumerate(train_loader):
	        start_time = time.time()

	        # Move tensors to the configured device
	        feature = torch.from_numpy(feature).float().to(device)
	        labels = labels.to(device)
	        
	        # Forward pass
	        outputs = model(feature)
	        loss = criterion(outputs, labels)
	        
	        # Backward and optimize
	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()
	        running_loss += loss.item()

	        if (i+1) % data["show"] == 0:
	            writer.add_scalar('training loss',running_loss/  data["show"],epoch * len(train_loader) + i)
	            logger(logging,'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, running_loss: {:.4f}, time: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item(), running_loss,(time.time()- start_time)/data["show"]))
	            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
	            print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
	            running_loss = 0.0
	    
	    torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, data["checkpoint"]+"epoch_{}.pth".format(epoch + 1))


if __name__ == '__main__':
	main(sys.argv)
