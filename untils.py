import pandas as pd
import random as rn
import numpy as np



def sample_binary(m, n, p):
    #print(m, n)
    A = np.random.uniform(0.0, 1.0, size = [m, n])
    B = A > p
    C = 1.*B
    return C


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size = size, scale = xavier_stddev)


def build_mask(no,p_miss):
    nan_mat=np.array([])
    for i in p_miss:
        if nan_mat.size==0:
            nan_mat=np.random.random([no,1])>i
        else:
            nan_mat1=np.random.random([no,1])>i
            nan_mat=np.concatenate((nan_mat,nan_mat1), axis=1)
    return nan_mat

def impute(X,M):
    Z= Variable(FloatTensor(np.random.uniform(0, 1, (X.shape[0], dim))))
    New_X = M * X + (1-M) * Z  # Missing Data Introduce
    G_sample = netG(New_X,M)
    Hat_New_X=M*X +(1-M)*G_sample
    
    return Hat_New_X



def missing_method(raw_data,t=0.2) :
    
    data = raw_data.copy()
    rows, cols = data.shape
    
    # missingness threshold
    v = np.random.uniform(size=(rows, cols))

            # missing values where v<=t
    mask = (v<=t)
    data[mask] = 0
    
    return data, 1-mask.astype(float)
            
        

