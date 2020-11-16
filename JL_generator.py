# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 19:50:23 2019

@author: Adiel
"""
from scipy.sparse import csr_matrix as SM
import scipy.sparse as ssp
import numpy as np
Data = ssp.load_npz('D:/Adiel_data/enwiki-march19-pages-articles.npz')

#Data=Data[0:4096,:]   
Data.data=Data.data.astype(float)
JL=SM.dot(Data,np.random.randn(Data.shape[1],120))
np.save('D:/Adiel_data/Wiki_JL120.npy',JL)