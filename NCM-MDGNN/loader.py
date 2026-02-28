import numpy as np
from glob import glob
import os
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# precomputed mean and std of the dataset

mesh_folder = '/home/gnn/data_process/meshcnn'
class ClimateSegLoader(Dataset):
    """Data loader for Climate Segmentation dataset."""

    def __init__(self, partition="train"):
        """
        Args:
            data_dir: path to data directory
            partition: train or test
        """
        self.partition=partition
        load_path='/WdHeDisk/users/xuzhewen/re_train_retention'
        assert(partition in ["train", "test", "val"])
        if partition == "train":
            self.fin = np.load('/WdHeDisk/users/xuzhewen/re_train_retention/data_preprocess/x_interplot.npy')
            self.fout = np.load('/WdHeDisk/users/xuzhewen/re_train_retention/data_preprocess/y_interplot.npy')
        else:
            self.fin = np.load('/WdHeDisk/users/xuzhewen/re_train_retention/data_preprocess/x_interplot.npy')
            self.fout = np.load('/WdHeDisk/users/xuzhewen/re_train_retention/data_preprocess/y_interplot.npy')
        # self.fin = np.load('XX.npy')
        # self.fout = np.load('YY.npy')
        # self.fin = np.load('/home/gnn/capcity_tian/data_preprocess/XX_55.npy')
        # self.fout = np.load('/home/gnn/capcity_tian/data_preprocess/YY_55.npy')
        self.mean_list=np.load(load_path+'/data_preprocess/mean_list_inter.npy')
        self.std_list=np.load(load_path+'/data_preprocess/std_list_inter.npy')
        self.mean_list_X=np.load(load_path+'/data_preprocess/mean_list_inter_X.npy')
        self.std_list_X=np.load(load_path+'/data_preprocess/std_list_inter_X.npy')

    def __len__(self):
        length = self.fin.shape[0]
        return length

    def __getitem__(self, idx):

        ## load climatology
        # fin_norm=self.fin
        # fout_norm=self.fout
        for i in range(self.fin.shape[1]):
            self.fin[idx,i] = (self.fin[idx,i]-self.mean_list_X[i])/self.std_list_X[i] #self.std_list[j]
        for j in range(self.fout.shape[1]):
            self.fout[idx,j] = (self.fout[idx,j]-self.mean_list[j])/self.std_list[j]

        return self.fin[idx].astype(np.float32), self.fout[idx].astype(np.float32)

