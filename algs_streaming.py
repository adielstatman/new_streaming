# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 23:50:39 2020

@author: Adiel
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:23:44 2019

@author: Adiel
"""
import scipy 

import numpy as np
import scipy.sparse as ssp
from scipy.sparse import coo_matrix as CM
from scipy.sparse import csr_matrix as SM
import time
def unif_sam(A,coreset_size,is_sparse=0):
    """
    uniform sampling in size of coreset_size
    """


    S=A[np.random.choice(A.shape[0],size=coreset_size),:]
    return S,0,0

def initializing_data(Q,k):
    print('Q',Q.shape)
    #U, D, VT = ssp.linalg.svds(np.transpose(Q),2*k)
    #print('Q',Q)
    Q[np.isnan(Q)]=0
    U, D, VT = np.linalg.svd(np.transpose(Q))

    V=VT.T
    D=np.diag(D)
    Z=V[:,0:2*k]
    V_k=V[:,0:k] 
    print(D.shape)
    DVk=np.dot(D[0:k,0:k],np.transpose(V_k))
    Q_k=np.dot(U[:,0:k],DVk)
    ZZt=np.dot(Z,np.transpose(Z))
    Qt=np.transpose(Q)
    A_2t=(np.sqrt(k)/np.linalg.norm(Q-np.transpose(Q_k),'fro'))*(Qt-np.dot(Qt,ZZt))
    A_2=np.transpose(A_2t)
    An=np.concatenate((Z,A_2),1)

    return An,A_2

def single_CNW_iteration_classic( A,At,delta_u,delta_l,X_u,X_l,Z):
     #AtA=np.dot(At,A)
     M_u = np.linalg.inv(X_u-Z)
     M_l = np.linalg.inv(Z-X_l)
     betha_diff=np.zeros((A.shape[0],2))
     L=np.dot(A,np.dot(M_l,At))
     L2=np.dot(L,L)
     U=np.dot(A,np.dot(M_u,At))
     U2=np.dot(U,U)
     betha_l0=L2/(delta_l*np.trace(L2))-L
     betha_u0=U2/(delta_u*np.trace(U2))+U
     betha_l=np.diag(betha_l0)
     betha_u=np.diag(betha_u0)  

     betha_diff=betha_l-betha_u
     #print(np.max(betha_diff))
     betha_diff2=np.argmax(betha_diff)         
     jj=betha_diff2
     t=(1/betha_l[jj])/2+(1/betha_u[jj])/2 #should be between (1/betha_l[jj]) to (1/betha_u[jj])
     aj=np.zeros((1,A.shape[1]))
     aj[0,:]=A[jj,:]      
     ajtaj=np.dot(np.transpose(aj),aj)
     Z=Z+np.power(t,1)*ajtaj


     return Z,jj,t
 
def SCNW_classic(A2,k,coreset_size,is_jl):
    coreset_size=int(coreset_size)

    """
    This function operates the CNW algorithm, exactly as elaborated in Feldman & Ras

    inputs:
    A: data matrix, n points, each of dimension d.
    k: an algorithm parameter which determines the normalization neededand the error given the coreset size.
    coreset_size: the maximal coreset size (number of lines inequal to zero) demanded for input.
    output:
    error: The error between the original data to the CNW coreset.        
    duration: the duration this CNW operation lasted
    """
    if is_jl==1:
        dex=int(k*np.log(A2.shape[0]))
    
        ran=np.random.randn(A2.shape[1],dex)
        A1=SM.dot(A2,ran)	
    else:
        A1=np.copy(A2)
    print('A1.shape',A1.shape)
    epsi=np.sqrt(k/coreset_size)    #
    A,A3=initializing_data(A1,k)
    print('A.shape',A.shape)
    At=np.transpose(A)
    AtA=np.dot(At,A)
    num_of_channels = A.shape[1]
    ww = np.zeros((int(coreset_size)))
    Z = np.zeros((num_of_channels,num_of_channels))
    X_u = k*np.diag(np.ones(num_of_channels))
    X_l =-k*np.diag(np.ones(num_of_channels))
    delta_u = epsi+2*np.power(epsi, 2)
    delta_l = epsi-2*np.power(epsi, 2)
    ind=np.zeros(int(coreset_size), dtype=np.int)             

    for j in range(coreset_size):
         if j%50==1:
             print('j=',j)
         X_u=X_u+delta_u*AtA
         X_l=X_l+delta_l*AtA                  
         Z,jj,t=single_CNW_iteration_classic(A,At,delta_u,delta_l,X_u,X_l,Z)
         ww[j]=t
         ind[j]=jj             
    sqrt_ww=np.sqrt(epsi*ww/k)
    sqrt_ww=np.reshape(sqrt_ww,(len(sqrt_ww),1))
    if is_jl==1:
        SA0=SM(A2)[ind,:].multiply(sqrt_ww)
    else:
        SA0=np.multiply(A2[ind,:],sqrt_ww)
    return SA0,ind

def SVD_streaming(Data,j,h,alg,trial=None,datum=None):
    """
    alg=0 unif sampling
    alg=1 Sohler
    alg=2 CNW
    alg=3 Alaa
    """
    alfa=0.0000000001
    alg_inmine=3
    real_cs1=0
    prob=0
    coreset_size=Data.shape[0]//(2**(h+1))
    print(coreset_size)
    gamma=j/(coreset_size-j+1)
    #k=218410*2*coreset_size
    k=0
    T_h= [0] * (h+1)
    never= [0] * (h+1)

    DeltaT_h= np.zeros(h+1)
    leaf_ind=np.zeros(h+1)
    Tlist=[]
    last_time=time.time()
    for jj in range(np.power(2,h)): #over all of the leaves
        Q=Data[k:k+2*coreset_size,:]
        print('Q',Q.shape)
    
        k=k+2*coreset_size
        #T=Alg1(Q,j)
        if alg==2:
                T,DeltaT,real_cs1=unif_sam(Q,coreset_size)
                #T=np.random.rand(Q.shape[0],Q.shape[1])
                #DeltaT=0
                #real_cs1=0
                #prob=np.ones(T.shape[0])/T.shape[0]
#           if alg==1:
#                T,DeltaT,real_cs1=sohler1(totT,j,gamma)
        if alg==1:
                #T=Alg1(Q,j)
                #T=np.zeros((coreset_size,Data.shape[1]))
                _,T,DeltaT,u,real_cs1=alaa_coreset(Q,j,gamma,coreset_size,np.ones(Q.shape[0])/Q.shape[0],0,0)       
        if alg==0:
                # spar==1:
                 #   T,_=SCNW_classic(vstack((SM(T),T_h[i])),j,coreset_size,is_jl)
                #else:
                    T,_=SCNW_classic(Q,j,coreset_size,0)

        if alg==3:


               prob,T,DeltaT,real_cs1=Nonuniform_Alaa(Q,j,is_pca,gamma,spar)
       
        print('leaf num',jj)
        print('j=',h)
        i=0
                        

        while (i<h+2)*(type(T_h[i])!=int): #every time the leaf has a neighbor leaf it should merged and reduced
           totT=np.concatenate((T,np.asarray(T_h[i])),0)

                          #T=T*len(totT)/2*real_cs1
           if alg==2:
                T,DeltaT,real_cs1=unif_sam(totT,coreset_size)
                #T=np.random.rand(totT.shape[0],totT.shape[1])
                #DeltaT=0
                #real_cs1=0
                #prob=np.ones(T.shape[0])/T.shape[0]
#           if alg==1:
#                T,DeltaT,real_cs1=sohler1(totT,j,gamma)
           if alg==1:
                #T=Alg1(tot,j)
                ##T=np.zeros((coreset_size,Data.shape[1]))

                #print(totT)

                _,T,DeltaT,u,real_cs1=alaa_coreset(totT,j,gamma,coreset_size,np.ones(totT.shape[0])/totT.shape[0],0,0)       
                #print(T)

           if alg==0:
                # spar==1:
                 #   T,_=SCNW_classic(vstack((SM(T),T_h[i])),j,coreset_size,is_jl)
                #else:
                T,_=SCNW_classic(totT,j,coreset_size,0)

           if alg==3:


               prob,T,DeltaT,real_cs1=Nonuniform_Alaa(totT,j,is_pca,gamma,spar)
       
           #T=Alg1(totT,j)
    
           T_h[i]=0
           #if leaf_ind[i]==0:
            #print('saved T',T)
           leaf_ind[i]=leaf_ind[i]+1


           i=i+1
           if DeltaT_h[i]==0:
               DeltaT_h[i]=time.time()-last_time
        print('len',h)

        print('iiiiiiiiiiiiiiii',i)
        T_h[i]=T
        if never[i]==0:
            Tlist.append(T)
        never[i]=1     

        Q=[]

    return Tlist,DeltaT_h
def new_streaming(A,k,cs,sizes,is_sparse=0): #k is the "thin" of the SVD in Alg1
    beg=time.time()
    d=A.shape[1]
    Y=np.arange(cs)
    if is_sparse==1:
        PSI=SM((d,d))
    else:
        PSI=np.zeros((d,d))
    M= [[] for _ in range(int(cs))]   #M is list of 8*m lists of indeces of A
    ord1= [[] for _ in range(int(cs))]   #M is list of 8*m lists of indeces of A
    B= [[] for _ in range(int(cs))]   #M is list of 8*m lists of indeces of A
    u=np.random.rand(10000,cs) # a probability for each point
    Q=[0]*len(sizes)
    w=[0]*len(sizes)
    j=0 
    Times=[]
    from tqdm import tqdm
    #B=A[0:1,:]
    for i in tqdm(range(1,A.shape[0])):
       # if np.mod(i,10000)==1:
       #    np.save(str(i)+'.npy',i)
        a=A[i:i+1,:]    
        for y in Y:
           M[y].append(i)
        for y in Y:
            B[y]=A[M[y],:]
        PSI,Q1,w1,s,M,time1=stream(B,a,k,PSI,Y,M,ord1,i,u,is_sparse,beg)
        if i in sizes:
            if Q1==[]:
                Q[j]=A[np.random.choice(i,k+1),:]
                w[j]=(i/(k+1))*np.ones((k+1,1))
            else:
                Q[j]=A[np.ravel(Q1),:]
                w[j]=w1
            j=j+1
            Times.append(time1)
    return Q,w,Times

def stream(B,a,k,PSI,Y,M,ord1,i,u,is_sparse,beg):
    Q=[] 
    Q1=[]    
    w=[]            
    if is_sparse==1:
        PSI=PSI+SM.dot(a.T,a)
    else:
        PSI=PSI+np.dot(a.T,a)
    Z=Alg1(PSI,k) #Z and PSI are matrices 
    for y in Y:
        BB=B[y]
        C=np.dot(np.array(BB),Z)
        s=np.sum(np.power(C,2),1) #line 14
        ord1[y]=s.tolist()
        My=M[y].copy()
        ord2=ord1[y].copy()
        for mm in range(len(M[y])):
            if u[np.mod(mm,10000),y]>(s[mm]/(s[mm]+k)):
               My.remove(M[y][mm])
               ord2.remove(ord1[y][mm])
        M[y]=My
        ord1[y]=ord2
    for y in Y:
        if len(M[y])==1:
            Q.append(M[y])
            Q1.append(ord1[y])
    for q in Q1:
        w.append(k/(len(Q1)*q[0]))
    return PSI,Q,w,s,M,time.time()-beg

def Alg1(PSI,k):
    _,D,V=np.linalg.svd(PSI)
    D=np.reshape(D,(len(D),1))
    Z=np.linalg.pinv(np.multiply(np.sqrt(D),V))
    return Z

def Alg2(PSI,k):
    _,D,V=np.linalg.svd(PSI)
    D=np.reshape(D,(len(D),1))
    Z=np.multiply(np.sqrt(D),V)
    return Z
def calc_error(A,SA,k):  
        AtA=np.dot(np.transpose(A),A)
        SAtSA=np.dot(np.transpose(SA),SA)    
        ro=100
        d=AtA.shape[0]
        X=np.random.rand(d,d-k)            
        V,R=np.linalg.qr(X)
        Vt=np.transpose(V)
        VtAtAV=np.dot(Vt,np.dot(AtA,V))
        VtSAtSAV=np.dot(Vt,np.dot(SAtSA,V))
        newro=np.abs(np.trace(VtAtAV)/np.trace(VtSAtSAV))
        j=0
        while np.logical_and(max(ro/newro,newro/ro)>1.01,j<10):
            j=j+1
            ro=newro
            G=AtA-ro*SAtSA
            w,v=np.linalg.eig(G)
            V=v[:,0:d-k]
            Vt=np.transpose(V)
            VtAtAV=np.dot(Vt,np.dot(AtA,V))
            VtSAtSAV=np.dot(Vt,np.dot(SAtSA,V))
            newro=np.abs(np.trace(VtAtAV)/np.trace(VtSAtSAV))
        roS=100
        d=AtA.shape[0]
        X=np.random.rand(d,d-k)            
        V,R=np.linalg.qr(X)
        Vt=np.transpose(V)
        VtAtAV=np.dot(Vt,np.dot(AtA,V))
        VtSAtSAV=np.dot(Vt,np.dot(SAtSA,V))
        newroS=np.abs(np.trace(VtSAtSAV)/np.trace(VtAtAV))
        j=0
        while np.logical_and(max(roS/newroS,newroS/roS)>1.01,j<10):
            j=j+1
            roS=newroS
            G=SAtSA-roS*AtA
            w,v=np.linalg.eig(G)
            V=v[:,0:d-k]
            Vt=np.transpose(V)
            VtAtAV=np.dot(Vt,np.dot(AtA,V))
            VtSAtSAV=np.dot(Vt,np.dot(SAtSA,V))
            newroS=np.abs(np.trace(VtSAtSAV)/np.trace(VtAtAV))
        return max(np.abs(newroS-1),np.abs(newro-1),np.abs(1-newro),np.abs(1-newroS))

def Nonuniform_Alaa(AA0,k,is_pca,eps,spar): 
        d=AA0.shape[1]
        if is_pca==1:
                k=k+1
                AA0=PCA_to_SVD(AA0,eps,spar)
        if is_jl==1:
            dex=int(k*np.log(AA0.shape[0]))
            ran=np.random.randn(AA0.shape[1],dex)
            if spar==1:
                AA=SM.dot(AA0,ran)
            else:
                AA=np.dot(AA0,ran)
        else:
            AA=AA0

        size_of_coreset=int(k+k/eps-1) 
        U,D,VT=ssp.linalg.svds(AA,k)       
        V = np.transpose(VT)
        print('spar',spar)
        print('is_jl',is_jl)
        AAV = np.dot(AA, V)
        del V
        del VT    
        x = np.sum(np.power(AA, 2), 1)
        y = np.sum(np.power(AAV, 2), 1)
        P = np.abs(x - y)
        AAV=np.concatenate((AAV,np.zeros((AAV.shape[0],1))),1)
        Ua, _, _ = ssp.linalg.svds(AAV,k)
        U = np.sum(np.power(Ua, 2), 1)
        pro = 2 * P / np.sum(P) + 8 * U
        if is_pca==1:
            pro=pro+81*eps
        pro0 = pro / sum(pro)
        w=np.ones(AA.shape[0])
        u=np.divide(w,pro0)/size_of_coreset
        DMM_ind=np.random.choice(AA.shape[0],size_of_coreset, p=pro0)
        u1=np.reshape(u[DMM_ind],(len(DMM_ind),1))
        if spar==1:
            SA0=SM(AA0)[DMM_ind,:d].multiply(np.sqrt(u1))

        else:
            SA0=np.multiply(np.sqrt(u1),AA0[DMM_ind,:d])
        return pro0,SA0,0,size_of_coreset#,#AA.shape[0]*SA0/size_of_coreset   
def sorted_eig(A):
	eig_vals, eig_vecs =scipy.linalg.eigh(A)  # np.linalg.eig(A)	
	eig_vals_sorted = np.sort(eig_vals)[::-1]
	eig_vecs = eig_vecs.T
	eig_vecs_sorted = eig_vecs[eig_vals.argsort()][::-1]
	return eig_vals_sorted,eig_vecs_sorted
def get_unitary_matrix(n, m):
	a = np.random.random(size=(n, m))
	q, _ = np.linalg.qr(a)
	return q
	
def get_gamma(A_tag,l,d):
	vals , _ = sorted_eig(A_tag)
	sum_up = 0;sum_down = 0
	for i in range (l) : 
		sum_up += vals[d-i -1]
		sum_down += vals[i]
	return (sum_up/sum_down)
def calc_sens(A,p,j,eps):
	
    d=A.shape[1]; l = d-j;
    A_tag = np.dot(A.T , A) ; 
    p = np.reshape(p, (p.shape[0], 1)).T ; 
    p_tag = np.dot(p.T,p) ;
    s_old = -float("inf")
    x = get_unitary_matrix(d, l)
    step = 0  ; stop = False
    gama = get_gamma(A_tag,l,d);
    stop_rule = (gama*eps)/(1-gama)

    	
    s_l = []
    s_old = 0 
    while  step <100:	
        s_new =  np.trace( np.dot (np.dot(x.T,p_tag) ,x))  / np.trace( np.dot(np.dot(x.T,A_tag) , x  ))
        	
        s_l.append(s_new)
        G = p_tag - s_new*A_tag
        _ , ev = sorted_eig(G)
        x = ev[:l].T
        if s_new - stop_rule < s_old :                
            return max(s_l)
        s_old = s_new 
        step+=1
    #print('step',step)
    return max(s_l)	
def PCA_to_SVD(P,epsi,is_spar):
    if is_spar==0:
        r=1+2*np.max(np.sum(np.power(P,2),1))/epsi**4
        P=np.concatenate((P,r*np.ones((P.shape[0],1))),1)
    else:
        P1=SM.copy(P)
        P1.data=P1.data**2
        r=1+2*np.max(np.sum(P1,1))/epsi**4
        P=hstack((P,r*np.ones((P.shape[0],1))))
    return P
def alaa_coreset(wiki0,j,eps,coreset_size,w,is_pca,spar):    
    #print('1')
    dex=int(j*np.log(wiki0.shape[0]))
    d=wiki0.shape[1]
    if is_pca==1:
        j=j+1
        wiki0=PCA_to_SVD(wiki0,eps,spar)
    wiki=wiki0
    w=w/wiki.shape[0]
    sensetivities=[]
    jd=j
    w1=np.reshape(w,(len(w),1))
    wiki1=np.multiply(np.sqrt(w1),wiki)
    k=0
    for i,p in enumerate(wiki1) :
        k=k+1
        sensetivities.append(calc_sens(wiki1,p,jd,eps))
    p0=np.asarray(sensetivities)
    if is_pca==1:
        p0=p0+81*eps
    indec=np.random.choice(np.arange(wiki.shape[0]),int(coreset_size),p=p0/np.sum(p0)) #sampling according to the sensitivity
    p=p0/np.sum(p0) #normalizing sensitivies
    w=np.ones(wiki.shape[0])

    u=np.divide(np.sqrt(w),p)/coreset_size #caculating new weights
    u1=u[indec]
    #u1=u1/np.mean(u1)
    u1=np.reshape(u1,(len(u1),1))
    squ=np.sqrt(u1)   
    if spar==1:        
        C=SM(wiki0)[indec,:d].multiply(squ)
    else:

        C=np.multiply(squ,wiki0[indec,:d])
    return p,C,0,u[indec],coreset_size#,wiki.shape[0]*wiki[indec,:]/coreset_size#


    
