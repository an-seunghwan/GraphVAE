#%%
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import time
import re
import matplotlib.pyplot as plt
from PIL import Image
from pprint import pprint
from scipy import sparse
import os
os.chdir('/Users/anseunghwan/Documents/GitHub/textmining')
#%%
'''keyword의 frequency가 1인 데이터는 제거'''
data = []
for i in tqdm(range(10)):
    data.append(pd.read_csv('/Users/anseunghwan/Documents/uos/textmining/한국보건사회연구원_데이터_형태소_{}월.csv'.format(i+1),
                encoding='cp949'))
#%%
'''keyword selection'''
# total frequency
total = np.array(data[0][data[0].columns[9:]].sum(axis=0))
for i in range(1, 10):
    total += np.array(data[i][data[i].columns[9:]].sum(axis=0))
#%%
keyidx = np.argsort(total)[-300:]
keywords = list(data[0].columns[9:][keyidx])

# keywords save
with open("./result/keywords.txt", "w") as f:
    for w in keywords:
        f.write(w + '\n')
#%%
'''frequency 정보를 모두 1로 맞춤'''
for j in range(10): 
    print(j)
    freq = np.array(data[j][data[j].columns[9:][keyidx]])
    freq[np.where(freq > 0)] = 1
    
    # the number of nodes at least: 10
    freq = freq[np.where(np.sum(freq, axis=-1) > 10)[0], :]

    for i in range(int(len(freq) / 1000) + 1):
        print(i)
        ftemp = freq[1000*i : 1000*(i+1), :]
        adj = []
        # adj = np.zeros((1, len(keywords), len(keywords)))
        for k in tqdm(range(len(ftemp))):
            '''adjacency matrix'''
            # A = np.array(sparse.csr_matrix(ftemp[[k], :].T).multiply(sparse.csr_matrix(ftemp[[k], :])).toarray())
            A = ftemp[[k], :].T @ ftemp[[k], :]
            # adj = np.concatenate((adj, A[None, :, :]), axis=0)
            adj.append(A)
        adj = np.array(adj).reshape(len(adj), -1)
        adj = sparse.csr_matrix(adj)
        sparse.save_npz('/Users/anseunghwan/Documents/uos/textmining/data/{}월/A{}.npz'.format(j+1, i), adj)
        # np.save('/Users/anseunghwan/Documents/uos/textmining/data/{}월/A{}'.format(j+1, i), adj)
#%%