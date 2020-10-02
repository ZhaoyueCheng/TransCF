import time
import pandas as pd
import math
from Dataset import DatasetNew
import numpy as np
import torch
from torch.autograd import Variable
import heapq
import evaluation
import fastrand
from tqdm import trange

class Recommender(object):
    def __init__(self, args):
        self.cuda_available = torch.cuda.is_available()
        self.recommender = args.recommender
        self.numEpoch = args.numEpoch
        self.batch_size = args.batch_size
        self.embedding_dim = args.embedding_dim
        self.lRate = args.lRate
        self.Ks = [5, 10, 20]
        self.reg1 = args.reg1
        self.reg2 = args.reg2
        self.num_negatives = args.num_negatives
        self.dataset = args.dataset
        self.margin = args.margin
        self.rand_seed = args.rand_seed
        np.random.seed(self.rand_seed)
        self.mode = args.mode
        self.cuda = args.cuda
        self.batchSize_test = args.batchSize_test
        self.early_stop = args.early_stop
        self.path = args.path
        
        dataset = DatasetNew(self.path)

        self.train = dataset.train
        self.train_items = dataset.train_items
        self.test = dataset.test
        self.test_items = dataset.test_items
        self.user_item_csr = dataset.user_item_csr

        self.userCache = dataset.userCache
        self.itemCache = dataset.itemCache

        self.numUsers = dataset.numUsers
        self.numItems = dataset.numItems

        self.totalTrainUsers, self.totalTrainItems = dataset.totalTrainUsers, dataset.totalTrainItems                
            
        # Evaluation
        self.bestHR1 = 0; self.bestNDCG1 = 0; self.bestMRR1 = 0; self.bestHR5 = 0; self.bestNDCG5 = 0; self.bestMRR5 = 0; self.bestHR10 = 0; self.bestNDCG10 = 0; self.bestMRR10 = 0; self.bestHR20 = 0; self.bestNDCG20 = 0; self.bestMRR20 = 0; self.bestHR50 = 0; self.bestNDCG50 = 0; self.bestMRR50 = 0; 
        self.early_stop_metric = []
        
    def currentTime(self):
        now = time.localtime()
        s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        return s

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    
    def is_converged(self, model, epoch, totalLoss, topHits, topNdcgs, topMrrs):
        
        HR1 = topHits[1]
        NDCG1 = topNdcgs[1]
        MRR1 = topMrrs[1]
        if HR1 > self.bestHR1:
            self.bestHR1 = HR1
        if NDCG1 > self.bestNDCG1:
            self.bestNDCG1 = NDCG1
        if MRR1 > self.bestMRR1:
            self.bestMRR1 = MRR1
        
        HR5 = topHits[5]
        NDCG5 = topNdcgs[5]
        MRR5 = topMrrs[5]
        if HR5 > self.bestHR5:
            self.bestHR5 = HR5
        if NDCG5 > self.bestNDCG5:
            self.bestNDCG5 = NDCG5
        if MRR5 > self.bestMRR5:
            self.bestMRR5 = MRR5
        
        HR10 = topHits[10]
        NDCG10 = topNdcgs[10]
        MRR10 = topMrrs[10]
        if HR10 > self.bestHR10:
            self.bestHR10 = HR10            
        if NDCG10 > self.bestNDCG10:
            self.bestNDCG10 = NDCG10
        if MRR10 > self.bestMRR10:
            self.bestMRR10 = MRR10
        
        HR20 = topHits[20]
        NDCG20 = topNdcgs[20]
        MRR20 = topMrrs[20]
        if HR20 > self.bestHR20:
            self.bestHR20 = HR20
        if NDCG20 > self.bestNDCG20:
            self.bestNDCG20 = NDCG20
        if MRR20 > self.bestMRR20:
            self.bestMRR20 = MRR20
        
        HR50 = topHits[50]
        NDCG50 = topNdcgs[50]
        MRR50 = topMrrs[50]
        if HR50 > self.bestHR50:
            self.bestHR50 = HR50
        if NDCG50 > self.bestNDCG50:
            self.bestNDCG50 = NDCG50
        if MRR50 > self.bestMRR50:
            self.bestMRR50 = MRR50

        if epoch % 10 == 0:
            print("[%s] [iter=%d %s] Loss: %.2f, margin: %.3f | %.4f, %.4f, %.4f, %.4f, %.4f || %.4f, %.4f, %.4f, %.4f, %.4f || %.4f, %.4f, %.4f, %.4f, %.4f"%(self.recommender, epoch+1, self.currentTime(), totalLoss, self.margin, self.bestHR1, self.bestHR5, self.bestHR10, self.bestHR20, self.bestHR50, self.bestNDCG1, self.bestNDCG5, self.bestNDCG10, self.bestNDCG20, self.bestNDCG50, self.bestMRR1, self.bestMRR5, self.bestMRR10, self.bestMRR20, self.bestMRR50))

        self.early_stop_metric.append(self.bestHR10)
        if self.mode == 'Val' and epoch > self.early_stop and self.bestHR10 == self.early_stop_metric[epoch-self.early_stop]:
            print("[%s] [Final (Early Converged)] %.4f, %.4f, %.4f, %.4f, %.4f || %.4f, %.4f, %.4f, %.4f, %.4f || %.4f, %.4f, %.4f, %.4f, %.4f" %(self.recommender, self.bestHR1, self.bestHR5, self.bestHR10, self.bestHR20, self.bestHR50, self.bestNDCG1, self.bestNDCG5, self.bestNDCG10, self.bestNDCG20, self.bestNDCG50, self.bestMRR1, self.bestMRR5, self.bestMRR10, self.bestMRR20, self.bestMRR50))
            return True
        
    def printFinalResult(self):
        print("[%s] [Final] %.4f, %.4f, %.4f, %.4f, %.4f || %.4f, %.4f, %.4f, %.4f, %.4f || %.4f, %.4f, %.4f, %.4f, %.4f" %(self.recommender, self.bestHR1, self.bestHR5, self.bestHR10, self.bestHR20, self.bestHR50, self.bestNDCG1, self.bestNDCG5, self.bestNDCG10, self.bestNDCG20, self.bestNDCG50, self.bestMRR1, self.bestMRR5, self.bestMRR10, self.bestMRR20, self.bestMRR50))

    def evalScore(self, model):

        pred_matrix = self.predict_batch(model)
        results = self.eval_rec(pred_matrix)
            
        return results

    def eval_rec(self, pred_matrix):
        topk = 50
        pred_matrix[self.user_item_csr.nonzero()] = np.NINF
        ind = np.argpartition(pred_matrix, -topk)
        ind = ind[:, -topk:]
        arr_ind = pred_matrix[np.arange(len(pred_matrix))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_matrix)), ::-1]
        pred_list = ind[np.arange(len(pred_matrix))[:, None], arr_ind_argsort]

        precision, recall, MAP, ndcg = [], [], [], []

        # ranking = argmax_top_k(pred_matrix, topk)  # Top-K items

        for k in [5, 10, 20]:
            precision.append(precision_at_k(self.test_items, pred_list, k))
            recall.append(recall_at_k(self.test_items, pred_list, k))
            # MAP.append(mapk(data.test_dict, pred_list, k))

        all_ndcg = ndcg_lgcn([*self.test_items.values()], pred_list)
        ndcg = [all_ndcg[x - 1] for x in [5, 10, 20]]

        return precision, recall, ndcg

    def predict(self, model):
        num_users, num_items = self.numUsers, self.numItems
        probs_matrix = np.zeros((num_users, num_items))

        for i in trange(num_users):
            users = torch.LongTensor(np.array([i] * num_items)).cuda(self.cuda)
            items = torch.LongTensor(np.arange(num_items)).cuda(self.cuda)

            i_viewed_u_idx, i_viewed_u_offset, u_viewed_i_idx, u_viewed_i_offset = self.getNeighbors(users, items)

            vals, _ = model(users, items, u_viewed_i_idx, u_viewed_i_offset, i_viewed_u_idx, i_viewed_u_offset)
            vals *= -1.0

            if self.cuda_available == True:
                vals = vals.detach().cpu().numpy() * -1
            else:
                vals = vals.numpy() * -1

            probs_matrix[i] = np.reshape(vals, [-1, ])

        return probs_matrix

    def predict_batch(self, model):
        num_users, num_items = self.numUsers, self.numItems
        probs_matrix = np.zeros((num_users, num_items))

        all_users = np.repeat(np.arange(num_users), num_items)
        all_items = np.tile(np.arange(num_items), num_users)

        all_probs = []

        for i in trange(0, len(all_users), self.batchSize_test):
            start = i
            end = i + self.batchSize_test

            users = torch.LongTensor(all_users[start:end]).cuda(self.cuda)
            items = torch.LongTensor(all_items[start:end]).cuda(self.cuda)

            # users = torch.LongTensor(np.array([i] * num_items)).cuda(self.cuda)
            # items = torch.LongTensor(np.arange(num_items)).cuda(self.cuda)

            i_viewed_u_idx, i_viewed_u_offset, u_viewed_i_idx, u_viewed_i_offset = self.getNeighbors(users, items)

            vals, _ = model(users, items, u_viewed_i_idx, u_viewed_i_offset, i_viewed_u_idx, i_viewed_u_offset)
            vals *= -1.0

            if self.cuda_available == True:
                vals = vals.detach().cpu().numpy() * -1
            else:
                vals = vals.numpy() * -1

            # probs_matrix[i] = np.reshape(vals, [-1, ])
            all_probs.extend(list(vals))

        probs_matrix = np.array(all_probs).reshape(num_users, num_items)

        return probs_matrix

    
    def getNeighbors(self, uids, iids):
        uid_idxvec = []
        uid_offset = []
        prev_len = 0
        if self.cuda_available == True:
            iids = iids.cpu().data.numpy().tolist()
        else:
            iids = iids.data.numpy().tolist()
        
        for iid in iids:
            users = self.itemCache[iid]
            uid_idxvec += users
            uid_offset.append(prev_len)
            prev_len += len(users)
            
            
        iid_idxvec = []
        iid_offset = []
        prev_len = 0
        if self.cuda_available == True:
            uids = uids.cpu().data.numpy().tolist()
        else:
            uids = uids.data.numpy().tolist()
            
        for uid in uids:
            items = self.userCache[uid]
            iid_idxvec += items
            iid_offset.append(prev_len)
            prev_len += len(items)
        
        if self.cuda_available == True:
            return (torch.LongTensor(iid_idxvec)).cuda(self.cuda), (torch.LongTensor(iid_offset)).cuda(self.cuda), (torch.LongTensor(uid_idxvec)).cuda(self.cuda), Variable(torch.LongTensor(uid_offset)).cuda(self.cuda)
        else:
            return (torch.LongTensor(iid_idxvec)), (torch.LongTensor(iid_offset)), (torch.LongTensor(uid_idxvec)), (torch.LongTensor(uid_offset))
    
    def getTestInstances(self):
        trainItems = set(self.train.iid.unique())
        test=dict()
        # Make test data
        input = range(self.numUsers)
        bins = [input[i:i+self.batchSize_test] for i in range(0, len(input), self.batchSize_test)]

        for bin_idx, bin in enumerate(bins):
            userIdxs = []
            itemIdxs = []
            prevOffset = 0
            offset = [0]
            for uid in bin:
                if self.mode == 'Val':
                    rating = self.valRatings[uid]
                else:
                    rating = self.testRatings[uid]
                items = self.negatives[uid]
                items = list(trainItems.intersection(set(items)))
                u = rating[0]
                assert (uid == u)
                gtItem = rating[1]
                if gtItem not in trainItems:
                    continue
                items.append(gtItem)

                users = [u] * len(items)

                userIdxs += users
                itemIdxs += items
                offset.append(prevOffset + len(users))
                prevOffset += len(users)

            test.setdefault(bin_idx, dict())
            test[bin_idx]['offsets'] = offset
            if self.cuda_available == True:
                test[bin_idx]['u'] = torch.LongTensor(np.array(userIdxs)).cuda(self.cuda)
                test[bin_idx]['i'] = torch.LongTensor(np.array(itemIdxs)).cuda(self.cuda)

            else:
                test[bin_idx]['u'] = torch.LongTensor(np.array(userIdxs))
                test[bin_idx]['i'] = torch.LongTensor(np.array(itemIdxs))
                
        return test
    
    
    def getTrainInstances(self):
        totalData = []
        for s in range(self.numUsers * self.num_negatives):
            while True:
                u = fastrand.pcg32bounded(self.numUsers)
                cu = self.userCache[u]
                if len(cu) == 0:
                    continue

                t = fastrand.pcg32bounded(len(cu))
                
                #i = list(cu)[t]
                i = cu[t]
                j = fastrand.pcg32bounded(self.numItems)

                while j in cu:
                    j = fastrand.pcg32bounded(self.numItems)
                    
                break

            totalData.append([u, i, j])
                
        totalData = np.array(totalData)
        
        return totalData
        

def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(actual)
    for i, v in actual.items():
        act_set = set(v)
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)
    return sum_precision / num_users

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(actual)
    true_users = 0
    for i, v in actual.items():
        act_set = set(v)
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    assert num_users == true_users
    return sum_recall / true_users

def ndcg_lgcn(ground_truths, ranks):
    result = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)

        # calculate idcg
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len-1]

        # idcg = np.cumsum(1.0/np.log2(np.arange(2, len_rank+2)))
        dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        result += dcg / idcg
    return result / len(ranks)