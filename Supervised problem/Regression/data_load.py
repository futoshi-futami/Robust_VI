# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:36:17 2017

@author: Futami
"""
import numpy as np
import os


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
