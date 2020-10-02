import scipy.sparse as sp
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import fastrand
from numpy import random
import time
import pickle
from scipy.sparse import csr_matrix

class Dataset(object):
    def __init__(self, totalFilename, trainFilename, valFilename, testFilename, negativesFilename):
        print("Reading Dataset")
        self.totalData = pd.read_csv(totalFilename, sep='\t')[['uid','iid']]
        self.train = pd.read_csv(trainFilename, sep='\t')[['uid','iid']]
            
        self.trainMatrix = self.load_rating_file_as_matrix(trainFilename)
        self.valRatings = self.load_rating_file_as_list(valFilename)
        self.testRatings = self.load_rating_file_as_list(testFilename)
        self.negatives = self.load_negative_file(negativesFilename)

        assert len(self.testRatings) == len(self.negatives)
        self.numUsers, self.numItems = len(self.totalData.uid.unique()), len(self.totalData.iid.unique())
        
        self.userCache = self.getuserCache()
        self.itemCache = self.getitemCache()
        
        self.totalTrainUsers = set(self.train.uid.unique())
        self.totalTrainItems = set(self.train.iid.unique())
        
        print("[Rating] numUsers: %d, numItems: %d, numRatings: %d]" %(self.numUsers, self.numItems, len(self.trainMatrix)))
        
        # Free memory
        self.totalData.drop(self.totalData.index, inplace=True)
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                mat[user, item] = 1.0
                line = f.readline()    
        return mat
    
    
    def getuserCache(self):
        train = self.train
        totalItems = set(range(self.numItems))
        userCache = {}
        userCache_rev = {}
        for uid in train.uid.unique():
            items = train.loc[train.uid == uid]['iid'].values.tolist()
            userCache[uid] = items
        
        return userCache
    
    def getitemCache(self):
        train = self.train
        totalUsers = set(range(self.numUsers))
        itemCache = {}
        itemCache_rev = {}
        #for iid in train.iid.unique():
        for iid in range(self.numItems):
            users = train.loc[train.iid == iid]['uid'].values.tolist()
            if len(users) == 0:
                users = []
            itemCache[iid] = users
            
        return itemCache


class DatasetNew(object):
    def __init__(self, path):
        print("Reading Dataset")

        self.path = path

        train_file = path + '/train.pkl'
        test_file = path + '/test.pkl'

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        self.exist_users = []
        self.train_items, self.test_items = {}, {}

        self.train_items = pickle.load(open(train_file, "rb"))
        self.test_items = pickle.load(open(test_file, "rb"))

        train_new = []
        for user, items in self.train_items.items():
            for item in items:
                train_new.append([user, item])
        train = np.array(train_new).astype(int)

        self.train = train

        self.n_users = np.max(train[:, 0])
        self.n_items = np.max(train[:, 1])
        self.n_train = train.shape[0]

        self.exist_users = list(self.train_items.keys())

        test_new = []
        for user, items in self.test_items.items():
            for item in items:
                test_new.append([user, item])
        test = np.array(test_new).astype(int)

        self.test = test

        self.numUsers = max(self.n_users, np.max(test[:, 0])) + 1
        self.numItems = max(self.n_items, np.max(test[:, 1])) + 1
        self.n_test = test.shape[0]

        self.itemCache = self.getitemCache()

        self.totalTrainUsers = self.numUsers
        self.totalTrainItems = self.numItems

        print("[Rating] numUsers: %d, numItems: %d, numRatings: %d]" % (
        self.numUsers, self.numItems, len(self.train)))

        self.userCache = self.train_items
        self.itemCache = self.getitemCache()

        self.user_item_csr = self.generate_rating_matrix([*self.train_items.values()], self.numUsers, self.numItems)

    def getitemCache(self):
        itemCache = {}

        for i in range(self.numItems):
            itemCache[i] = []

        for u, i in self.train:
            itemCache[i].append(u)

        return itemCache

    def generate_rating_matrix(self, train_set, num_users, num_items):
        # three lists are used to construct sparse matrix
        row = []
        col = []
        data = []
        for user_id, article_list in enumerate(train_set):
            for article in article_list:
                row.append(user_id)
                col.append(article)
                data.append(1)

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

        return rating_matrix


class Dataset_TransCF(Dataset):
    def __init__(self, totalData):
        self.totalData = totalData
        
    def __len__(self):
        return len(self.totalData)
    
    def __getitem__(self, idx):
        result = {'u':self.totalData[idx,0],'i':self.totalData[idx,1],'j':self.totalData[idx,2]}
        return result               
