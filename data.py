import pandas as pd
import random as rn
import numpy as np
from sklearn import model_selection

# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size = [m, n])

# unifrom sampler for categorical data
def sample_lambda(m,min_,max_):
    return np.random.uniform(min_,max_, size = m)

def normalization (data, parameters=None):
    """Normalize data in [0, 1] range."""
      # Parameters
    _, dim = data.shape
    norm_data = data.copy()
    
    if parameters is None:

    # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

    # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:,i])
            max_val[i] = np.nanmax(norm_data[:,i])
            norm_data[:,i] = (norm_data[:,i] - min_val[i]+ 1e-3)
            norm_data[:,i] = norm_data[:,i] / ( (max_val[i]-min_val[i] )+ 1e-3)

        # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                            'max_val': max_val}

    return norm_data, norm_parameters


def renormalization (norm_data, norm_parameters):
    """Renormalize numrical data from [0, 1] range to the original range."""
  
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']
        
    _, dim = norm_data.shape
    renorm_data = norm_data.copy()
    
    for i in range(dim):
        renorm_data[:,i] = renorm_data[:,i] * (max_val[i]-min_val[i] + 1e-3)
        renorm_data[:,i] = renorm_data[:,i] + min_val[i]- 1e-3
    
    return renorm_data


def data_load(file_x,file_y,file_m, mask=False):
    if mask==True:
        trainX_M=np.genfromtxt('trainM_'+file_m+'.csv', delimiter=',',skip_header=1)
        testX_M=np.genfromtxt('testM_'+file_m+'.csv', delimiter=',',skip_header=1)
        devX_M=np.genfromtxt('devM_'+file_m+'.csv', delimiter=',',skip_header=1)

        trainy=np.genfromtxt('trainy_'+file_y+'.csv', delimiter=',',skip_header=1).reshape(-1,1)
        testy=np.genfromtxt('testy_'+file_y+'.csv', delimiter=',',skip_header=1).reshape(-1,1)
        devy=np.genfromtxt('devy_'+file_y+'.csv', delimiter=',',skip_header=1).reshape(-1,1)
        trainX=np.genfromtxt('trainX_'+file_x+'.csv', delimiter=',',skip_header=1)
        testX=np.genfromtxt('testX_'+file_x+'.csv', delimiter=',',skip_header=1)
        devX=np.genfromtxt('devX_'+file_x+'.csv', delimiter=',',skip_header=1)
    
        return trainX, trainy, devX, devy, testX, testy, trainX_M,testX_M,devX_M
    else:
        trainy=np.genfromtxt('trainy_'+file_y+'.csv', delimiter=',',skip_header=1).reshape(-1,1)
        testy=np.genfromtxt('testy_'+file_y+'.csv', delimiter=',',skip_header=1).reshape(-1,1)
        devy=np.genfromtxt('devy_'+file_y+'.csv', delimiter=',',skip_header=1).reshape(-1,1)
        trainX=np.genfromtxt('trainX_'+file_x+'.csv', delimiter=',',skip_header=1)
        testX=np.genfromtxt('testX_'+file_x+'.csv', delimiter=',',skip_header=1)
        devX=np.genfromtxt('devX_'+file_x+'.csv', delimiter=',',skip_header=1)
        
        return trainX, trainy, devX, devy, testX, testy
        

def cv_data_load(file_x,file_y,file_m, testfold):
    if  testfold==-1:
        X=np.array(pd.read_csv("X_"+file_x+'.csv'))
        M=np.array(pd.read_csv('M_'+file_m+'.csv'))
        y=np.array(pd.read_csv('y_'+file_y+'.csv'))
        return X,M ,y
    else:
        test_index=pd.read_csv('test_5flod_idx.csv')
        X=pd.read_csv("X_"+file_x+'.csv')
        M=pd.read_csv('M_'+file_m+'.csv')
        y=pd.read_csv('y_'+file_y+'.csv')
    
        testX=np.array(X.iloc[test_index.iloc[testfold]])
        trainX=np.array(X.drop(test_index.iloc[testfold]))
    
        testX_M=np.array(M.iloc[test_index.iloc[testfold]])
        trainX_M=np.array(M.drop(test_index.iloc[testfold]))
    
        testy=np.array(y.iloc[test_index.iloc[testfold]])
        trainy=np.array(y.drop(test_index.iloc[testfold]))
    
        return trainX, trainy ,testX, testy, trainX_M,testX_M 
    

    
def cv_data_load2(testfold,method="trgain"):
    if (method=="trgain") or (method=="fill0"):
    #test_index=pd.read_csv('test_5flod_idx.csv')
        trainX=np.array(pd.read_csv('data/X'+str(testfold)+'.csv'))
        trainX_M=np.array(pd.read_csv('data/M'+str(testfold)+'.csv'))
        trainy=np.array(pd.read_csv('data/y'+str(testfold)+'.csv'))

        testX=np.array(pd.read_csv('data/testX'+str(testfold)+'.csv'))
        testX_M=np.array(pd.read_csv('data/testM'+str(testfold)+'.csv'))
        testy=np.array(pd.read_csv('data/testy'+str(testfold)+'.csv'))
    else:
        trainX=np.array(pd.read_csv('data/X'+str(testfold)+method+'.csv'))
        trainX_M=np.array(pd.read_csv('data/M'+str(testfold)+'.csv'))
        trainy=np.array(pd.read_csv('data/y'+str(testfold)+'.csv'))

        testX=np.array(pd.read_csv('data/testX'+str(testfold)+method+'.csv'))
        testX_M=np.array(pd.read_csv('data/testM'+str(testfold)+'.csv'))
        testy=np.array(pd.read_csv('data/testy'+str(testfold)+'.csv'))
        
    
    return trainX, trainy ,testX, testy, trainX_M,testX_M 


def read_csv(csv_filename, targername, cat_name=None, header=True):#, discrete=None
    """
    INPUT:
        csv_filename: data path string
        target name : list
    
    OUTPUT:
        minmax normalized: np x
        target: np y
        
        """

    DF = pd.read_csv(csv_filename, header='infer' if header else None)
    result=targername
    
    df_train=DF.drop(columns=result)
    if cat_name!=None:
        num_name=list(set(DF.columns.tolist()) - set(cat_name+result))
        df_cat=df_train.loc[:,cat_name]
        df_cat=df_cat.to_numpy()
        df_cat_norm,norm_cat_parameters=normalization(df_cat)
        #df_cat_norm,norm_cat_parameters=catgorical_norm(df_cat)
    else:
        num_name=sorted(list(set(DF.columns.tolist()) - set(result)))
        
    df_num=df_train.loc[:,num_name]
    df_num=df_num.to_numpy()
    
    # target
    df_y=DF.loc[:,result]
    df_y=df_y.to_numpy()
    
    df_num_norm, norm_num_parameters=normalization(df_num)
    if cat_name!=None:
        df_X=np.concatenate([df_num_norm,df_cat_norm],axis=1)
    else:
        df_X=df_num_norm
    
   #df_X,

    return df_X, df_y,norm_num_parameters, num_name


def train_test_df(df_X, df_y,test_size=0.2):
    trainX,testX,trainy,testy =model_selection.train_test_split(df_X,df_y, test_size=test_size)
    return trainX,testX,trainy,testy
    

def mask(X,M=None):
    if M is not None:
        mask_data=X*M
        return mask_data, M
        
    else:
        no,dim=X.shape
        mask=(np.isnan(X)==False).astype('int')
        masked_data=np.nan_to_num(X, 0)
    
        return masked_data,mask


 
