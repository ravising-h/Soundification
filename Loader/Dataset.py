import os
import librosa
from os.path import join
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SoundCLassification(Dataset):
     

    def __init__(self, Folder, classes = ['disgust' ,'fear','happy','neutral','sad'], form = "mfcc"):

        self.root     = Folder
        self.form     = form
        self.classes  = classes 
        self.cate     = {i:x for x,i in enumerate(self.classes )}
        self.len_     = cpt = sum([len(files) for r, d, files in os.walk(self.root)])

    def listdir(self):
        Class = self.classes
        Class = Class if  isinstance(Class, list) else [Class]
        self.paths = list()
        for folder in Class:
            for i in os.listdir(join(self.root,folder)):
                self.paths.append(join(self.root, folder,i))
        return self.paths



    def getX(self, form, path):
        X, sample_rate = librosa.load(path, res_type='kaiser_fast')
        if form == "mfcc":
            return np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128).T,axis=0)
        elif form == "melspectrogram":
            return np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis = 0)
        elif form == "chroma":
            return np.mean(librosa.feature.chroma_cens(y=X, sr=sample_rate).T,axis = 0)

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        self.listdir()
        file_ = self.paths[idx]
        label = os.path.split(os.path.split(self.paths[idx])[0])[-1]
        label_ = np.zeros((len(self.classes)),dtype = np.float16)
        label_[self.cate[label]] = 1
        return self.getX(self.form, file_), self.cate[label]