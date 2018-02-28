# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 00:25:53 2017

@author: Futami
"""

import edward as ed
import numpy as np
import tensorflow as tf

KL=True
ed.set_seed(42)

from edward.models import Normal

from quasi_beta import KLqp_beta
from data_load import dataload2,generator,add_noise,preprocessing

def Reyni_estimator(VI,dataset_name,per):
    name=dataset_name
    X0,y0,n_splits,indexes=dataload2(name)
    N,D=X0.shape
        
    def neural_network(X):
      h = tf.nn.relu(tf.matmul(X, W_0) + b_0)
      h = tf.nn.relu(tf.matmul(h, W_1) + b_1)
      h = tf.matmul(h, W_2) + b_2
      return tf.reshape(h, [-1])
    
    

    M = 128    # batch size during training
    H=20
    
    n_batch = int(N / M)
    n_epoch = 200
    
    
    if VI=='KL':
        gamma_list=[1]
    # The outlier_tuning hyper-parameter
    # In this code, the cross validation will not be done for simplicity.
    # Just separate the training dataset into training data and test data and run 10 times for each hyper parameter.
    elif VI=='beta':
        gamma_list=[0.1,0.2,0.3,0.4]

    var=0.01
    var2=1.0000
    num=10
    
    Results=np.zeros(num)
    for gamma in gamma_list:
        # MODEL
        with tf.name_scope("model"):
          W_0 = Normal(loc=tf.zeros([D, H]), scale=tf.ones([D, H])*var2, name="W_0")
          W_1 = Normal(loc=tf.zeros([H, H]), scale=tf.ones([H, H])*var2, name="W_1")
          W_2 = Normal(loc=tf.zeros([H, 1]), scale=tf.ones([H, 1])*var2, name="W_2")
          b_0 = Normal(loc=tf.zeros(H), scale=tf.ones(H)*var2, name="b_0")
          b_1 = Normal(loc=tf.zeros(H), scale=tf.ones(H)*var2, name="b_1")
          b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(1)*var2, name="b_2")
        
        
          X = tf.placeholder(tf.float32, [None, D], name="X")
          y_ph = tf.placeholder(tf.float32, [None])
          y = Normal(loc=neural_network(X), scale=1., name="y")
        
        # INFERENCE
        with tf.name_scope("posterior"):
          with tf.name_scope("qW_0"):
            qW_0 = Normal(loc=tf.Variable(tf.random_normal([D, H],stddev=var), name="loc"),
                          scale=tf.nn.softplus(
                              tf.Variable(tf.random_normal([D, H],stddev=var), name="scale")))
          with tf.name_scope("qW_1"):
            qW_1 = Normal(loc=tf.Variable(tf.random_normal([H, H],stddev=var), name="loc"),
                          scale=tf.nn.softplus(
                              tf.Variable(tf.random_normal([H, H],stddev=var), name="scale")))
          with tf.name_scope("qW_2"):
            qW_2 = Normal(loc=tf.Variable(tf.random_normal([H, 1],stddev=var), name="loc"),
                          scale=tf.nn.softplus(
                              tf.Variable(tf.random_normal([H, 1],stddev=var), name="scale")))
          with tf.name_scope("qb_0"):
            qb_0 = Normal(loc=tf.Variable(tf.random_normal([H],stddev=var), name="loc"),
                          scale=tf.nn.softplus(
                              tf.Variable(tf.random_normal([H],stddev=var), name="scale")))
          with tf.name_scope("qb_1"):
            qb_1 = Normal(loc=tf.Variable(tf.random_normal([H],stddev=var), name="loc"),
                          scale=tf.nn.softplus(
                              tf.Variable(tf.random_normal([H],stddev=var), name="scale")))
          with tf.name_scope("qb_2"):
            qb_2 = Normal(loc=tf.Variable(tf.random_normal([1],stddev=var), name="loc"),
                          scale=tf.nn.softplus(
                              tf.Variable(tf.random_normal([1],stddev=var), name="scale")))
    
    
        results=[]
        for data in range(num):
            index=indexes[data]
            X_train, X_test = X0[index[0],:], X0[index[1],:]
            y_train, y_test = y0[index[0]], y0[index[1]]
            X_train,y_train,X_test,y_test, mean_y_train, std_y_train = preprocessing(X_train,y_train,X_test, y_test)
            
            X_train,y_train=add_noise(X_train,y_train,noise_x=(0,6,np.random.choice(D,D,replace=False)),noise_y=(0,6),percent=per)
            data = generator([X_train, y_train], M)
            N=X_train.shape[0]
            
            if VI=='KL':
                print("KL")
                inference = ed.KLqp({W_0: qW_0, b_0: qb_0,W_1: qW_1, b_1: qb_1, W_2: qW_2, b_2: qb_2}, data={y: y_ph})
                inference.initialize(n_iter=10000,n_samples=15, scale={y: N / M})
            elif VI=='beta':
                print("beta")
                inference = KLqp_beta({W_0: qW_0, b_0: qb_0,W_1: qW_1, b_1: qb_1,W_2: qW_2, b_2: qb_2}, data={y: y_ph})
                inference.initialize(n_iter=10000, n_samples=15,alpha=gamma,size=M,tot=N, scale={y: N / M})

            
            tf.global_variables_initializer().run()
        
            for _ in range(inference.n_iter):
                X_batch, y_batch = next(data)
                info_dict = inference.update({X: X_batch, y_ph: y_batch})
                inference.print_progress(info_dict)
        
            y_post = ed.copy(y, {W_0: qW_0, b_0: qb_0,
                                 W_1: qW_1, b_1: qb_1,
                                 W_2: qW_2, b_2: qb_2})
            print("Mean squared error on test data:")
            a=ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test})
            print(std_y_train*(a**0.5),a,std_y_train)
            print(gamma)
            results.append(std_y_train*(a**0.5))
        results=np.array(results)
        Results=np.vstack((Results,results))
    
    mu=np.mean(Results,-1)
    std=np.std(Results,-1)
    
    Saving=[mu,std]
    np.save(str('RMSE_')+str(VI)+str(name)+str(per)+'.npy', Saving)

if __name__ == '__main__': 
    VI=['KL','beta']
    # Percentage of outliers
    per=[0.]
    # Dataset name
    name=['concrete']
    # Which method we use
    V=VI[1]

    
    for n in name:
        for p in per:
            print(n,p)
            Reyni_estimator(V,n,p)