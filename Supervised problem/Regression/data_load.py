# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:36:17 2017

@author: Futami
"""
import numpy as np
import os

def dataload(name):
    os.chdir('../dataset')
    if name=='concrete':
        X = np.loadtxt('concrete_ARD_Xtrain__FOLD_1', delimiter=' ')
        Y = np.loadtxt('concrete_ARD_ytrain__FOLD_1', delimiter=' ')
        #Y=Y[:,None]
        X_1 = np.loadtxt('concrete_ARD_Xtrain__FOLD_2', delimiter=' ')
        Y_1 = np.loadtxt('concrete_ARD_ytrain__FOLD_2', delimiter=' ')
        
        X_2 = np.loadtxt('concrete_ARD_Xtrain__FOLD_3', delimiter=' ')
        Y_2 = np.loadtxt('concrete_ARD_ytrain__FOLD_3', delimiter=' ')

        X_te = np.loadtxt('concrete_ARD_Xtest__FOLD_1', delimiter=' ')
        X_te2 = np.loadtxt('concrete_ARD_Xtest__FOLD_2', delimiter=' ')
        X_te3 = np.loadtxt('concrete_ARD_Xtest__FOLD_3', delimiter=' ')
        Y_te = np.loadtxt('concrete_ARD_ytest__FOLD_1', delimiter=' ')
        Y_te2 = np.loadtxt('concrete_ARD_ytest__FOLD_2', delimiter=' ')
        Y_te3 = np.loadtxt('concrete_ARD_ytest__FOLD_3', delimiter=' ')

        
        return ([X,Y,X_te,Y_te],[X_1,Y_1,X_te2,Y_te2],[X_2,Y_2,X_te3,Y_te3])

        
    elif name=='powerplant':
        X = np.loadtxt('powerplant_ARD_Xtrain__FOLD_1', delimiter=' ')
        Y = np.loadtxt('powerplant_ARD_ytrain__FOLD_1', delimiter=' ')

        X_2 = np.loadtxt('powerplant_ARD_Xtrain__FOLD_2', delimiter=' ')
        Y_2 = np.loadtxt('powerplant_ARD_ytrain__FOLD_2', delimiter=' ')

        X_3 = np.loadtxt('powerplant_ARD_Xtrain__FOLD_3', delimiter=' ')
        Y_3 = np.loadtxt('powerplant_ARD_ytrain__FOLD_3', delimiter=' ')
        
        X_te = np.loadtxt('powerplant_ARD_Xtest__FOLD_1', delimiter=' ')
        X_te2 = np.loadtxt('powerplant_ARD_Xtest__FOLD_2', delimiter=' ')
        X_te3 = np.loadtxt('powerplant_ARD_Xtest__FOLD_3', delimiter=' ')
                
        Y_te = np.loadtxt('powerplant_ARD_ytest__FOLD_1', delimiter=' ')
        Y_te2 = np.loadtxt('powerplant_ARD_ytest__FOLD_2', delimiter=' ')
        Y_te3 = np.loadtxt('powerplant_ARD_ytest__FOLD_3', delimiter=' ')
        
        return ([X,Y,X_te,Y_te],[X_2,Y_2,X_te2,Y_te2],[X_3,Y_3,X_te3,Y_te3])
    
    elif name=='credit':
        X = np.loadtxt('credit_ARD_Xtrain__FOLD_1', delimiter=' ')
        X_2 = np.loadtxt('credit_ARD_Xtrain__FOLD_2', delimiter=' ')
        X_3 = np.loadtxt('credit_ARD_Xtrain__FOLD_3', delimiter=' ')
        X_4 = np.loadtxt('credit_ARD_Xtrain__FOLD_4', delimiter=' ')
        X_5 = np.loadtxt('credit_ARD_Xtrain__FOLD_5', delimiter=' ')

        Y = np.loadtxt('credit_ARD_ytrain__FOLD_1', delimiter=' ')
        Y_2 = np.loadtxt('credit_ARD_ytrain__FOLD_2', delimiter=' ')
        Y_3 = np.loadtxt('credit_ARD_ytrain__FOLD_3', delimiter=' ')
        Y_4 = np.loadtxt('credit_ARD_ytrain__FOLD_4', delimiter=' ')
        Y_5 = np.loadtxt('credit_ARD_ytrain__FOLD_5', delimiter=' ')
        
        X_te = np.loadtxt('credit_ARD_Xtest__FOLD_1', delimiter=' ')
        X_te2 = np.loadtxt('credit_ARD_Xtest__FOLD_2', delimiter=' ')
        X_te3 = np.loadtxt('credit_ARD_Xtest__FOLD_3', delimiter=' ')
        X_te4 = np.loadtxt('credit_ARD_Xtest__FOLD_4', delimiter=' ')
        X_te5 = np.loadtxt('credit_ARD_Xtest__FOLD_5', delimiter=' ')

        Y_te = np.loadtxt('credit_ARD_ytest__FOLD_1', delimiter=' ')
        Y_te2 = np.loadtxt('credit_ARD_ytest__FOLD_2', delimiter=' ')
        Y_te3 = np.loadtxt('credit_ARD_ytest__FOLD_3', delimiter=' ')
        Y_te4 = np.loadtxt('credit_ARD_ytest__FOLD_4', delimiter=' ')
        Y_te5 = np.loadtxt('credit_ARD_ytest__FOLD_5', delimiter=' ')
        
        return ([X,Y,X_te,Y_te],[X_2,Y_2,X_te2,Y_te2],[X_3,Y_3,X_te3,Y_te3],[X_4,Y_4,X_te4,Y_te4],[X_5,Y_5,X_te5,Y_te5])
    
    elif name=='protein':
        X = np.loadtxt('protein_ARD_Xtrain__FOLD_1', delimiter=' ')
        X_2 = np.loadtxt('protein_ARD_Xtrain__FOLD_2', delimiter=' ')
        X_3 = np.loadtxt('protein_ARD_Xtrain__FOLD_3', delimiter=' ')

        Y = np.loadtxt('protein_ARD_ytrain__FOLD_1', delimiter=' ')
        Y_2 = np.loadtxt('protein_ARD_ytrain__FOLD_2', delimiter=' ')
        Y_3 = np.loadtxt('protein_ARD_ytrain__FOLD_3', delimiter=' ')

        
        X_te = np.loadtxt('protein_ARD_Xtest__FOLD_1', delimiter=' ')
        X_te2 = np.loadtxt('protein_ARD_Xtest__FOLD_2', delimiter=' ')
        X_te3 = np.loadtxt('protein_ARD_Xtest__FOLD_3', delimiter=' ')

        Y_te = np.loadtxt('protein_ARD_ytest__FOLD_1', delimiter=' ')
        Y_te2 = np.loadtxt('protein_ARD_ytest__FOLD_2', delimiter=' ')
        Y_te3 = np.loadtxt('protein_ARD_ytest__FOLD_3', delimiter=' ')
        
        return ([X,Y,X_te,Y_te],[X_2,Y_2,X_te2,Y_te2],[X_3,Y_3,X_te3,Y_te3])

    elif name=='spam':
        X = np.loadtxt('spam_ARD_Xtrain__FOLD_1', delimiter=' ')
        X_2 = np.loadtxt('spam_ARD_Xtrain__FOLD_2', delimiter=' ')
        X_3 = np.loadtxt('spam_ARD_Xtrain__FOLD_3', delimiter=' ')
        X_4 = np.loadtxt('spam_ARD_Xtrain__FOLD_4', delimiter=' ')
        X_5 = np.loadtxt('spam_ARD_Xtrain__FOLD_5', delimiter=' ')

        Y = np.loadtxt('spam_ARD_ytrain__FOLD_1', delimiter=' ')
        Y_2 = np.loadtxt('spam_ARD_ytrain__FOLD_2', delimiter=' ')
        Y_3 = np.loadtxt('spam_ARD_ytrain__FOLD_3', delimiter=' ')
        Y_4 = np.loadtxt('spam_ARD_ytrain__FOLD_4', delimiter=' ')
        Y_5 = np.loadtxt('spam_ARD_ytrain__FOLD_5', delimiter=' ')

        
        X_te = np.loadtxt('spam_ARD_Xtest__FOLD_1', delimiter=' ')
        X_te2 = np.loadtxt('spam_ARD_Xtest__FOLD_2', delimiter=' ')
        X_te3 = np.loadtxt('spam_ARD_Xtest__FOLD_3', delimiter=' ')
        X_te4 = np.loadtxt('spam_ARD_Xtest__FOLD_4', delimiter=' ')
        X_te5 = np.loadtxt('spam_ARD_Xtest__FOLD_5', delimiter=' ')

        Y_te = np.loadtxt('spam_ARD_ytest__FOLD_1', delimiter=' ')
        Y_te2 = np.loadtxt('spam_ARD_ytest__FOLD_2', delimiter=' ')
        Y_te3 = np.loadtxt('spam_ARD_ytest__FOLD_3', delimiter=' ')
        Y_te4 = np.loadtxt('spam_ARD_ytest__FOLD_4', delimiter=' ')
        Y_te5 = np.loadtxt('spam_ARD_ytest__FOLD_5', delimiter=' ')
        
        return ([X,Y,X_te,Y_te],[X_2,Y_2,X_te2,Y_te2],[X_3,Y_3,X_te3,Y_te3],[X_4,Y_4,X_te4,Y_te4],[X_5,Y_5,X_te5,Y_te5])
    
    elif name=='eeg':
        X = np.loadtxt('eeg_ARD_Xtrain__FOLD_1', delimiter=' ')
        X_2 = np.loadtxt('eeg_ARD_Xtrain__FOLD_2', delimiter=' ')
        X_3 = np.loadtxt('eeg_ARD_Xtrain__FOLD_3', delimiter=' ')
        X_4 = np.loadtxt('eeg_ARD_Xtrain__FOLD_4', delimiter=' ')
        X_5 = np.loadtxt('eeg_ARD_Xtrain__FOLD_5', delimiter=' ')

        Y = np.loadtxt('eeg_ARD_ytrain__FOLD_1', delimiter=' ')
        Y_2 = np.loadtxt('eeg_ARD_ytrain__FOLD_2', delimiter=' ')
        Y_3 = np.loadtxt('eeg_ARD_ytrain__FOLD_3', delimiter=' ')
        Y_4 = np.loadtxt('eeg_ARD_ytrain__FOLD_4', delimiter=' ')
        Y_5 = np.loadtxt('eeg_ARD_ytrain__FOLD_5', delimiter=' ')
        
        X_te = np.loadtxt('eeg_ARD_Xtest__FOLD_1', delimiter=' ')
        X_te2 = np.loadtxt('eeg_ARD_Xtest__FOLD_2', delimiter=' ')
        X_te3 = np.loadtxt('eeg_ARD_Xtest__FOLD_3', delimiter=' ')
        X_te4 = np.loadtxt('eeg_ARD_Xtest__FOLD_4', delimiter=' ')
        X_te5 = np.loadtxt('eeg_ARD_Xtest__FOLD_5', delimiter=' ')

        Y_te = np.loadtxt('eeg_ARD_ytest__FOLD_1', delimiter=' ')
        Y_te2 = np.loadtxt('eeg_ARD_ytest__FOLD_2', delimiter=' ')
        Y_te3 = np.loadtxt('eeg_ARD_ytest__FOLD_3', delimiter=' ')
        Y_te4 = np.loadtxt('eeg_ARD_ytest__FOLD_4', delimiter=' ')
        Y_te5 = np.loadtxt('eeg_ARD_ytest__FOLD_5', delimiter=' ')

        return ([X,Y,X_te,Y_te],[X_2,Y_2,X_te2,Y_te2],[X_3,Y_3,X_te3,Y_te3],[X_4,Y_4,X_te4,Y_te4],[X_5,Y_5,X_te5,Y_te5])
    
def generator(arrays, batch_size):
  """Generate batches, one with respect to each array's first axis."""
  starts = [0] * len(arrays)  # pointers to where we are in iteration
  while True:
    batches = []
    for i, array in enumerate(arrays):
      start = starts[i]
      stop = start + batch_size
      diff = stop - array.shape[0]
      if diff <= 0:
        batch = array[start:stop]
        starts[i] += batch_size
      else:
        batch = np.concatenate((array[start:], array[:diff]))
        starts[i] = diff
      batches.append(batch)
    yield batches
  
def add_noise(X_train,y_train,noise_x=(0,0,[0]),noise_y=(0,0),percent=0.1):
    n,d=X_train.shape
    m=np.int(np.floor(n*percent))
    index=np.random.choice(n,size=m)
    index2=np.random.choice(n,size=m)
    for i in noise_x[2]:
        X_train[index,i] =np.random.normal(noise_x[0],noise_x[1],size=m)
    
    y_train[index] =np.random.normal(noise_y[0],noise_y[1],size=m)
    
    return X_train,y_train

def add_noise3(X_train,y_train,noise_x=(0,0,[0]),noise_y=(0,0),percent=0.1):
    n,d=X_train.shape
    m=np.int(np.floor(n*percent))
    index=np.random.choice(n,size=m)
    index2=np.random.choice(n,size=m)
    for i in noise_x[2]:
        X_train[index,i] =np.random.normal(noise_x[0],noise_x[1],size=m)
    
    y_train[index2] =-1*y_train[index2]
    
    return X_train,y_train

def add_noise2(X_train,y_train,noise_x=(0,0,[0]),noise_y=(0,0),percent=0.1):
    import tensorflow as tf
    tf.InteractiveSession()
    n,d=X_train.shape
    m=np.int(np.floor(n*percent))
    index=np.random.choice(n,size=m)
    #index2=np.random.choice(n,size=m)
    for i in noise_x[2]:
        X_train[index,i] =np.random.normal(noise_x[0],noise_x[1],size=m)
    #misspecified_label=tf.one_hot(np.random.choice(2,size=m),1).eval()
    y_train[index] =np.abs(y_train[index]-1)#misspecified_label  
    
    return X_train,y_train

def add_noise4(X_train,y_train,noise_x=(0,0,[0]),noise_y=(0,0),percent=0.1):
    import tensorflow as tf
    tf.InteractiveSession()
    n,d=X_train.shape
    m=np.int(np.floor(n*percent))
    index=np.random.choice(n,size=m)
    #index2=np.random.choice(n,size=m)
    for i in noise_x[2]:
        X_train[index,i] =np.random.normal(noise_x[0],noise_x[1],size=m)
    #misspecified_label=tf.one_hot(np.random.choice(2,size=m),1).eval()    
    return X_train,y_train

def add_noise5(X_train,y_train,noise_x=(0,0,0),noise_y=(0,0),percent=0.1):
    n,d=X_train.shape
    m=np.int(np.floor(n*percent))
    index=np.random.choice(n,size=m)
    y_train[index] =np.abs(y_train[index]-1)   
    return X_train,y_train



def label_noise(y_train,percent=0.1):
    import tensorflow as tf
    tf.InteractiveSession()
    n,d=y_train.shape
    m=np.int(np.floor(n*percent))
    index=np.random.choice(n,size=m)
    
    misspecified_label=tf.one_hot(np.random.choice(10,size=m),10).eval()
    y_train[index]=misspecified_label    
    return y_train

def preprocessing(X_train,y_train,X_test,y_test):

    std_X_train = np.std(X_train, 0)
    std_X_train[ std_X_train == 0 ] = 1
    mean_X_train = np.mean(X_train, 0)
    X_train = (X_train - mean_X_train) / std_X_train
    X_test = (X_test - mean_X_train) / std_X_train
    mean_y_train = np.mean(y_train, 0)
    std_y_train = np.std(y_train, 0)

    y_train = (y_train - mean_y_train) / std_y_train
    y_test = (y_test - mean_y_train) / std_y_train
              
    train_set = (X_train, y_train)
    test_set = (X_test, y_test)

    return X_train,y_train,X_test,y_test, mean_y_train, std_y_train

def preprocessing2(X_train,X_test):

    std_X_train = np.std(X_train, 0)
    std_X_train[ std_X_train == 0 ] = 1
    mean_X_train = np.mean(X_train, 0)
    X_train = (X_train - mean_X_train) / std_X_train
    X_test = (X_test - mean_X_train) / std_X_train

    return X_train,X_test

def dataload2(name):
    os.chdir('../dataset')
    if name=='protein':
        data = np.loadtxt('protein_data/data.txt')
        index_features = np.loadtxt('protein_data/index_features.txt')
        index_target = np.loadtxt('protein_data/index_target.txt')
        index_features=index_features.astype(np.int64)
        index_target=index_target.astype(np.int64)
        X = data[ : , index_features ]
        y = data[ : , index_target.tolist() ]
        n_splits = np.loadtxt('protein_data/n_splits.txt')
        n_splits=n_splits.astype(np.int64)
        indexes=[]
        for i in range(n_splits):
            index_train = np.loadtxt("protein_data/index_train_{}.txt".format(i))
            index_train=index_train.astype(np.int64)
            index_test = np.loadtxt("protein_data/index_test_{}.txt".format(i))
            index_test=index_test.astype(np.int64)
            indexes.append([index_train,index_test])
        return X,y,n_splits,indexes
    
    elif name=='concrete':
        data = np.loadtxt('concrete_data/data.txt')
        index_features = np.loadtxt('concrete_data/index_features.txt')
        index_target = np.loadtxt('concrete_data/index_target.txt')
        index_features=index_features.astype(np.int64)
        index_target=index_target.astype(np.int64)
        X = data[ : , index_features ]
        y = data[ : , index_target.tolist() ]
        n_splits = np.loadtxt('concrete_data/n_splits.txt')
        n_splits=n_splits.astype(np.int64)
        indexes=[]
        for i in range(n_splits):
            index_train = np.loadtxt("concrete_data/index_train_{}.txt".format(i))
            index_train=index_train.astype(np.int64)
            index_test = np.loadtxt("concrete_data/index_test_{}.txt".format(i))
            index_test=index_test.astype(np.int64)
            indexes.append([index_train,index_test])
        return X,y,n_splits,indexes
    
    elif name=='powerplant':
        data = np.loadtxt('powerplant_data/data.txt')
        index_features = np.loadtxt('powerplant_data/index_features.txt')
        index_target = np.loadtxt('powerplant_data/index_target.txt')
        index_features=index_features.astype(np.int64)
        index_target=index_target.astype(np.int64)
        X = data[ : , index_features ]
        y = data[ : , index_target.tolist() ]
        n_splits = np.loadtxt('powerplant_data/n_splits.txt')
        n_splits=n_splits.astype(np.int64)
        indexes=[]
        for i in range(n_splits):
            index_train = np.loadtxt("powerplant_data/index_train_{}.txt".format(i))
            index_train=index_train.astype(np.int64)
            index_test = np.loadtxt("powerplant_data/index_test_{}.txt".format(i))
            index_test=index_test.astype(np.int64)
            indexes.append([index_train,index_test])
        return X,y,n_splits,indexes
    
    elif name=='eeg':
        import pandas as pd
        Dataset_originall=pd.read_csv('eeg/eeg.csv')#ファイルの読み込み
        Dataset=Dataset_originall.values
        Dataset=Dataset.astype(np.float64)
        X=Dataset[:,:-1]
        Y=Dataset[:,-1]
        return X,Y
    
    elif name=='spam':
        import pandas as pd
        Dataset_originall=pd.read_csv('spam/spam.csv')#ファイルの読み込み
        Dataset=Dataset_originall.values
        Dataset=Dataset.astype(np.float64)
        np.random.shuffle(Dataset)
        X=Dataset[:,:-1]
        Y=Dataset[:,-1]
        return X,Y
