# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 23:49:41 2020

@author: Adiel
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:31:56 2019

@author: Adiel
"""
from scipy import io    
import algs_streaming as algs 
import scipy.sparse as ssp
from scipy.sparse import linalg
from scipy.sparse import coo_matrix as CM
from scipy.sparse import csr_matrix as SM
from scipy.sparse import dia_matrix
from scipy.sparse import hstack,vstack
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
"""
This script compare between merge/reduce tree to a new streaming method which elaborated in the paper
Braverman, V., Feldman, D., Lang, H., Rus, D., & Statman, A. (2020). Sparse Coresets for SVD on Infinite Streams. arXiv preprint arXiv:2002.06296.‏
This script generates the experiments there. 
Although we used random data, any numpy array data can be input here.
"""

d=8
k=d-1
flo=10
Data=np.random.rand(2**flo,30)
Data=Data[:,:d]
Data1=Data[:2**flo,:]  
h=np.arange(int(np.log2(Data1.shape[0])))+1
sizes1=[]
for h1 in h:
    sizes1.append(np.power(2,h1))
sizes=[]
h=np.arange(4*int(np.log2(Data1.shape[0])))+1
for h1 in h:
    sizes.append(int(np.power(np.power(2,0.25),h1)))
ss=np.where(np.array(sizes)>k)[0][0]
sizes=sizes[ss:]
#sizes=np.concatenate((np.ravel(np.array(sizes1[0:6])),np.arange(4,2**flo,100)))
alg_num=3
num_of_exp=1 #number of experiments to average for the new streaming 
num_of_exp1=1 #number of experiments to average for the old streaming

error=np.zeros((num_of_exp,len(sizes)-1))
error0=np.zeros((1,len(sizes)-1))

error1=np.zeros((num_of_exp1*alg_num,len(sizes1)))
error11=np.zeros((alg_num,len(sizes1)))


times1=np.zeros((alg_num,len(sizes1)-2))

sizeof=np.zeros((len(sizes)-1))
#for h in range(numexp):

for j in range(num_of_exp):
    Q,w,times= algs.new_streaming(Data1,k,d,sizes) #k is the "thin" of the SVD in Alg1

    for i in range(1,len(sizes)-1):
            Datai=Data[0:sizes[i],:]
            wi=np.ravel(w[i])
            warr=np.reshape(wi,(len(wi),1))
            X1=np.multiply(warr,Q[i])
            if X1.shape[0]<k+1:
                rind=np.random.choice(sizes[i],1+k-X1.shape[0])
                B=(Datai.shape[0]/(k+1))*Datai[rind,:]
                X=np.concatenate((X1,B),0)
            else:
                X=np.copy(X1)
            error[j,i]=algs.calc_error(Datai,X,k)
            sizeof[i]=Q[i].shape[0]
    if flo==20:
            sizess=np.reshape(sizes[0:-1],(1,len(sizes[0:-1])))
            x=np.concatenate((sizess,error0[:,0:]),0)
            np.save('C:/Users/Adiel/Dropbox/All_experimental/new_streaming/full_errors_d='+str(d)+'_new_'+str(int(time.time()))+'.npy',x)
                
for j in range(num_of_exp1):
    for alg in range(alg_num):
                    
        all_levels,time1=algs.SVD_streaming(Data1,k,int(flo-np.log2(d)-1),alg,0,0)   #calculates coreset    
        times1[alg,:len(time1)]=time1
        offset=0
        for i in range(0,len(all_levels)-offset):
        
                    Datai=Data1[0:sizes1[flo-len(all_levels)+i+offset],:]
                    error1[num_of_exp1*alg+j,i]=algs.calc_error(Datai,all_levels[i],k)

    if flo==20:
        sizess1=np.reshape(sizes1[int(np.log2(d)):int(np.log2(d))+len(all_levels)],(1,len(sizes1[int(np.log2(d)):int(np.log2(d))+len(all_levels)])))
        y=np.concatenate((sizess1,error11[:,:len(all_levels)]),0)
        np.save('C:/Users/Adiel/Dropbox/All_experimental/new_streaming/full_errors_d='+str(d)+'_'+str(int(time.time()))+'.npy',y)

if flo==20:
    sizess=np.reshape(sizes[0:-1],(1,len(sizes[0:-1])))
    x=np.concatenate((sizess,error[:,0:]),0)
    np.save('C:/Users/Adiel/Dropbox/All_experimental/new_streaming/full_errors_d='+str(d)+'_'+str(num_of_exp)+'_exp_new_'+str(int(time.time()))+'.npy',x)
    sizess1=np.reshape(sizes1[int(np.log2(d)):int(np.log2(d))+len(all_levels)],(1,len(sizes1[int(np.log2(d)):int(np.log2(d))+len(all_levels)])))
    y=np.concatenate((sizess1,error1[:,:len(all_levels)]),0)
    np.save('C:/Users/Adiel/Dropbox/All_experimental/new_streaming/full_errors_d='+str(d)+'_'+str(num_of_exp1)+'_exp_'+str(int(time.time()))+'.npy',y)

offset=0
lab=['Uniform sampling','Sensitivity sampling','CNW']
plt.figure(0)
plt.xscale('log')
plt.xlabel("Input size")
#plt.plot(sizes[1:-1],error[0,1:],'b.-',label='new streaming')
plt.plot(sizes[offset*4:-1],np.mean(error[:,offset*4:],0),label='new streaming')
for alg in range(alg_num):
    plt.plot(sizes1[int(np.log2(d))+offset:int(np.log2(d))+len(all_levels)],np.mean(error1[alg*num_of_exp1:(alg+1)*num_of_exp1,offset:len(all_levels)],0),label=lab[alg])
plt.legend(loc=0)
plt.ylabel("Error")
plt.ylabel("Error")
plt.ylim([0,100])
plt.title('k='+str(k)+', d='+str(d))
plt.figure(1)
plt.xscale('log')
plt.xlabel("Input size")
plt.plot(sizes[0:len(times)+2],times,label='new streaming')
for i in range(alg_num):
    plt.plot(sizes1[int(np.log2(d))+1:int(np.log2(d))+len(all_levels)],times1[i,1:len(all_levels)],label=lab[i])
plt.title('k='+str(k)+', d='+str(d))
plt.legend(loc=0)

plt.ylabel("Duration [sec]")
plt.figure(3)
plt.xscale('log')
plt.yscale('log')

plt.xlabel("Input size")
plt.plot(sizes[0:],times[:],label='new streaming')
for i in range(alg_num):
    plt.plot(sizes1[int(np.log2(d))+1:int(np.log2(d))+len(all_levels)],times1[i,1:len(all_levels)],label=lab[i])

plt.ylabel("Duration [sec logscaled]")
plt.title('k='+str(k)+', d='+str(d))
plt.legend(loc=0)
