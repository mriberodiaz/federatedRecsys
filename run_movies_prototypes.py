import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import random
import os
from utilsMovie import give_percentiles_all, avg_rank, process_data, train_eval, \
         give_percentiles, process_idx_users_items
from private_kmeans import clustering, eval_pred
import pickle

import warnings
warnings.filterwarnings('ignore')

from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

rand_seed = 7
random.seed(rand_seed)
np.random.seed(rand_seed)

data_all = pd.read_csv('/Users/monic/Box/Datasets_Federated_Learning/ml-1m/ratings.dat', header=None, names = ['uid','iid', 'count','Timestamp'], sep = '::')
users = pd.read_csv('/Users/monic/Box/Datasets_Federated_Learning/ml-1m/users.dat', header=None, names = ['uid','gender','age','occ','zip'], sep = '::', index_col= 0)

print(' Finished reading data \n')
# assign zone to user according to zip

data_all['entityid'] = users.loc[data_all.uid].zip.str.slice(stop=1).values.astype(np.int)

data_all.uid = data_all.uid-1
data_all = process_idx_users_items(data_all, users=False)

users = data_all.uid.unique()


H = len(data_all.entityid.unique())

u_train = np.random.binomial(1, p=0.9, size=len(users)).astype(bool)
data_all['user_train'] = u_train[data_all.uid]
data_all['rating_test'] = np.zeros_like(data_all.uid).astype(bool)
for u in users:
    if ~u_train[u]:
        tot = data_all.rating_test[data_all.uid==u].shape[0]
        data_all.rating_test[data_all.uid==u] = np.random.binomial(1, p=0.2, size=tot).astype(bool)

print(' Finished processing data \n')

# Factorization parameters
L = [ 10, 20,30,40,50]
reg = 0.1


# k-means parameters
ks = [20,30,40 ,100]
# Privacy parameters
epsilon = 0.1
delta = 0.1


# hospitals
list_hosp_all = [i for i in range(H)]        #  all hospitals

losses_pr = pickle.load(open('exp4/losses_privatekmeans.p', 'rb'))
centers_pr = pickle.load(open('exp4/centers_privatekmeans.p', 'rb'))

RANGE = 4



it = data_all.iid.unique().shape[0]

for k in ks:
    if k ==100:
        for h in range(1,10):
            data = data_all[data_all.entityid==h].copy()
            data = process_idx_users_items(data, users=True, items=False)
            X = csr_matrix((data['count'][~data.rating_test], (data.uid[~data.rating_test], data.iid[~data.rating_test])), shape=(data.uid.unique().shape[0], it)).toarray()
            RANGE = 4
            SIDE_LENGTH = np.amax(np.abs(X))
            res = clustering( X.T, k=k ,epsilon=epsilon,delta=delta,RANGE= RANGE, side_length=SIDE_LENGTH , JLcoef = 0.5, T=1)
            losses_pr[(k, h)] = res[4]
            centers_pr[(k, h)]=res[0]
            pickle.dump(losses_pr,  open('exp4/losses_privatekmeans.p', 'wb'))
            pickle.dump(centers_pr, open('exp4/centers_privatekmeans.p', 'wb'))
            print('Finished zip {}, k= {}'.format(h, k))
    else:
        for h in list_hosp_all:
            data = data_all[data_all.entityid==h].copy()
            data = process_idx_users_items(data, users=True, items=False)
            X = csr_matrix((data['count'][~data.rating_test], (data.uid[~data.rating_test], data.iid[~data.rating_test])), shape=(data.uid.unique().shape[0], it)).toarray()
            RANGE = 4
            SIDE_LENGTH = np.amax(np.abs(X))
            res = clustering( X.T, k=k ,epsilon=epsilon,delta=delta,RANGE= RANGE, side_length=SIDE_LENGTH , JLcoef = 0.5, T=1)
            losses_pr[(k, h)] = res[4]
            centers_pr[(k, h)]=res[0]
            pickle.dump(losses_pr,  open('exp4/losses_privatekmeans.p', 'wb'))
            pickle.dump(centers_pr, open('exp4/centers_privatekmeans.p', 'wb'))
            print('Finished zip {}, k= {}'.format(h, k))

