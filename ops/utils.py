import os
from os.path import join

def make_dir(folder):
	if not os.path.exists(folder):
		os.mkdir(folder)

def makedir(folders):
	[make_dir(folder) for folder in folders]

def logger(logging,string):
	logging.warning(string + "\n")
	print(string)
