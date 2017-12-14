# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:28:03 2017

@author: CNL
"""

import numpy as np
import sklearn as sk
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
# =============================================================================
# import matplotlib.pyplot as plt
# =============================================================================

# These datasets were created by smote_NCL_practice.py

# temporarily index
k = 0
with tf.device('/device:GPU:0'):
    # for dataset
    for data in [7,8,9,10]:
    
        # load dataset
        dataset = np.load('C:\\Users\\SHYang\\Desktop\\kaggle\\dataset%d.npy'%(data))
        #dataset = np.delete(dataset, 0, 1)
        # extract the mean and standard variation from jkgj
        std_scale = preprocessing.StandardScaler().fit(dataset[:,0:22])
        
        # divide into label 0 and 1 for skewed dataset
        label_0 = dataset[dataset[:,22]==0]
        label_1 = dataset[dataset[:,22]==1]
        
        # devide into input and label of dataset
        X0 = label_0[:,0:label_0.shape[1]-1]
        Y0 = label_0[:,label_0.shape[1]-1]
        Y0 = np.reshape(Y0, [len(Y0), 1])
        
        X1 = label_1[:,0:label_1.shape[1]-1]
        Y1 = label_1[:,label_1.shape[1]-1]
        Y1 = np.reshape(Y1, [len(Y1), 1])
        
        # train-validation split
        X0_train, X0_val, Y0_train, Y0_val = train_test_split(X0, Y0, test_size=2/9, random_state=42)
        X1_train, X1_val, Y1_train, Y1_val = train_test_split(X1, Y1, test_size=2/9, random_state=42)
        
        # Join the train sets and test sets
        X_train = np.concatenate((X0_train, X1_train), axis=0)
        X_val = np.concatenate((X0_val, X1_val), axis=0)
        Y_train = np.concatenate((Y0_train, Y1_train), axis=0)
        Y_val = np.concatenate((Y0_val, Y1_val), axis=0)
        
        # categorical (normal => column 0 , abnormal => column 1)
        ohe = OneHotEncoder()
        ohe.fit(Y_train)
        Y_train = ohe.transform(Y_train).toarray()
        ohe.fit(Y_val)
        Y_val = ohe.transform(Y_val).toarray()
        
        # normalization with the mean and standard variation from jkgj
        X_train = std_scale.transform(X_train)
        X_val = std_scale.transform(X_val)
        
        # for layers and nodes
        for layer in [1,2,3]:
            for node in [32,64,96,128]:
                # store the accuracy, the number of iteration and weights
                # row 1: iter, 2: tpr\\fpr, 3: precision, 4: recall, 5: f1_score
                #     6: fpr, 7: tpr, 8: tresholds
                # column 1: best, 2: current
                acc = np.zeros((8,1))
                
                # tf Graph input
                x = tf.placeholder("float", [None, X_train.shape[1]])
                y = tf.placeholder("float", [None, Y_train.shape[1]])
                
                
                if layer == 1:
                    # compute with GPU
                    #with tf.device('\\device:GPU:0'):
                    # set the weights and biases
                    W_1 = tf.get_variable("w_1_%d"%(k), shape=[X_train.shape[1], node], 
                                          initializer=tf.contrib.layers.xavier_initializer())
                    b_1 = tf.Variable(tf.constant(1.0, shape=[node]))
                    
                    W_2 = tf.get_variable("w_2_%d"%(k), shape=[node, 2],
                                          initializer=tf.contrib.layers.xavier_initializer())
                    b_2 = tf.Variable(tf.constant(1.0, shape=[2]))
                    
                    # set the tensor
                    z_1 = tf.add(tf.matmul(x, W_1), b_1)        
                    h_1 = tf.nn.relu(z_1) 
                    h_1_drop = tf.nn.dropout(h_1, keep_prob=0.5)
                    
                    output = tf.add(tf.matmul(h_1_drop, W_2), b_2)
                    
                    k += 1
                        
                elif layer == 2:
                    # compute with GPU
                    #with tf.device('\\device:GPU:0'):
                    # set the weights and biases
                    W_1 = tf.get_variable("w_1_%d"%(k), shape=[X_train.shape[1], node], 
                                          initializer=tf.contrib.layers.xavier_initializer())
                    b_1 = tf.Variable(tf.constant(1.0, shape=[node]))
                    
                    W_2 = tf.get_variable("w_2_%d"%(k), shape=[node, node],
                                          initializer=tf.contrib.layers.xavier_initializer())
                    b_2 = tf.Variable(tf.constant(1.0, shape=[node]))
                    
                    W_3 = tf.get_variable("w_3_%d"%(k), shape=[node, 2],
                                          initializer=tf.contrib.layers.xavier_initializer())
                    b_3 = tf.Variable(tf.constant(1.0, shape=[2]))
                    
                    # set the tensor
                    z_1 = tf.add(tf.matmul(x, W_1), b_1)        
                    h_1 = tf.nn.relu(z_1) 
                    h_1_drop = tf.nn.dropout(h_1, keep_prob=0.5)
                    
                    z_2 = tf.add(tf.matmul(h_1_drop, W_2), b_2)
                    h_2 = tf.nn.relu(z_2)
                    h_2_drop = tf.nn.dropout(h_2, keep_prob=0.5)
                    
                    output = tf.add(tf.matmul(h_2_drop, W_3), b_3)
                    
                    k += 1
                    
                else:
                    # compute with GPU
                    #with tf.device('\\device:GPU:0'):
                    # set the weights and biases
                    W_1 = tf.get_variable("w_1_%d"%(k), shape=[X_train.shape[1], node], 
                                          initializer=tf.contrib.layers.xavier_initializer())
                    b_1 = tf.Variable(tf.constant(1.0, shape=[node]))
                    
                    W_2 = tf.get_variable("w_2_%d"%(k), shape=[node, node],
                                          initializer=tf.contrib.layers.xavier_initializer())
                    b_2 = tf.Variable(tf.constant(1.0, shape=[node]))
                    
                    W_3 = tf.get_variable("w_3_%d"%(k), shape=[node, node],
                                          initializer=tf.contrib.layers.xavier_initializer())
                    b_3 = tf.Variable(tf.constant(1.0, shape=[node]))
                    
                    W_4 = tf.get_variable("w_4_%d"%(k), shape=[node, 2],
                                          initializer=tf.contrib.layers.xavier_initializer())
                    b_4 = tf.Variable(tf.constant(1.0, shape=[2]))
                    
                    # set the tensor
                    z_1 = tf.add(tf.matmul(x, W_1), b_1)        
                    h_1 = tf.nn.relu(z_1) 
                    h_1_drop = tf.nn.dropout(h_1, keep_prob=0.5)
                    
                    z_2 = tf.add(tf.matmul(h_1_drop, W_2), b_2)
                    h_2 = tf.nn.relu(z_2)
                    h_2_drop = tf.nn.dropout(h_2, keep_prob=0.5)
                    
                    z_3 = tf.add(tf.matmul(h_2_drop, W_3), b_3)
                    h_3 = tf.nn.relu(z_3) 
                    h_3_drop = tf.nn.dropout(h_3, keep_prob=0.5)
                    
                    output = tf.add(tf.matmul(h_3_drop, W_4), b_4)
                    
                    k += 1
        
                #with tf.device('\\device:GPU:0'):
                    # set the cost function
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
                
                # set the hyperparameter
                rate = tf.Variable(0.01)
                optimizer = tf.train.AdamOptimizer(rate)
                train = optimizer.minimize(cost)
                
                init = tf.global_variables_initializer()
                
                correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
                
                # precision, recall, f1-score...
                y_p = tf.argmax(output, 1)
                
                sess = tf.Session()
                sess.run(init)
                 
                #with tf.device('\\device:GPU:1'):      
                for i in range(1000):
                    model = sess.run(train, feed_dict={x:X_train, y:Y_train})
                    
                    val_cost = sess.run(cost, feed_dict={x:X_val, y:Y_val})
                    
                    y_pred = sess.run(y_p, feed_dict={x:X_val, y:Y_val})
                    y_true = np.argmax(Y_val,1)
                    precision = sk.metrics.precision_score(y_true, y_pred)
                    recall = sk.metrics.recall_score(y_true, y_pred)
                    f1_score = sk.metrics.f1_score(y_true, y_pred)
                    confusion = sk.metrics.confusion_matrix(y_true, y_pred)
                    fpr, tpr, tresholds = sk.metrics.roc_curve(y_true, y_pred)
                    auc = sk.metrics.auc(fpr, tpr)
            
                    if i == 0:
                        acc[:,0] = [i+1, tpr[1]-fpr[1], precision, recall, f1_score, fpr[1], 
                           tpr[1], auc]
                        
                        confusion_matrix = confusion
                        
                        acc_fpr = fpr
                        acc_tpr = tpr
                        
                        acc_auc = auc
                        
                        if layer == 1:
                            W1 = sess.run(W_1)
                            b1 = sess.run(b_1)
                            W2 = sess.run(W_2)
                            b2 = sess.run(b_2)
                            
                        elif layer == 2:
                            W1 = sess.run(W_1)
                            b1 = sess.run(b_1)
                            W2 = sess.run(W_2)
                            b2 = sess.run(b_2)
                            W3 = sess.run(W_3)
                            b3 = sess.run(b_3)
                        
                        else: 
                            W1 = sess.run(W_1)
                            b1 = sess.run(b_1)
                            W2 = sess.run(W_2)
                            b2 = sess.run(b_2)
                            W3 = sess.run(W_3)
                            b3 = sess.run(b_3)
                            W4 = sess.run(W_4)
                            b4 = sess.run(b_4)
                        
                    else:
                        if f1_score > acc[4,0]:
                            acc[:,0] = [i+1, tpr[1]-fpr[1], precision, recall, f1_score, fpr[1],
                               tpr[1], auc]
                            
                            confusion_matrix = confusion
                            
                            acc_fpr = fpr
                            acc_tpr = tpr
                            
                            acc_auc = auc
                            
                            if layer == 1:
                                W1 = sess.run(W_1)
                                b1 = sess.run(b_1)
                                W2 = sess.run(W_2)
                                b2 = sess.run(b_2)
                                
                            elif layer == 2:
                                W1 = sess.run(W_1)
                                b1 = sess.run(b_1)
                                W2 = sess.run(W_2)
                                b2 = sess.run(b_2)
                                W3 = sess.run(W_3)
                                b3 = sess.run(b_3)
                            
                            else: 
                                W1 = sess.run(W_1)
                                b1 = sess.run(b_1)
                                W2 = sess.run(W_2)
                                b2 = sess.run(b_2)
                                W3 = sess.run(W_3)
                                b3 = sess.run(b_3)
                                W4 = sess.run(W_4)
                                b4 = sess.run(b_4)
                            
                        else:
                            pass
                    if (i+1)%100 == 0:
                        print("iter: %d"%(i+1), "// tpr-fpr: %g"%(tpr[1]-fpr[1]),
                              " f1_score: %g"%(f1_score))
                    elif i == 0:
                        print("======== implementation//data: %d, layer: %d, node: %d ========"
                              %(data,layer,node))
                        print("iter: %d"%(i+1), "// tpr-fpr: %g"%(tpr[1]-fpr[1]),
                              " f1_score: %g"%(f1_score))  
                    else:
                        pass
                
                # save
                np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_1'%(data,layer,node),acc)
                np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_2'%(data,layer,node),confusion_matrix)
                np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_3'%(data,layer,node),acc_fpr)
                np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_4'%(data,layer,node),acc_tpr)
                if layer == 1:
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W1'%(data,layer,node),W1)
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b1'%(data,layer,node),b1)
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W2'%(data,layer,node),W2)
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b2'%(data,layer,node),b2)
                elif layer == 2:
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W1'%(data,layer,node),W1)
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b1'%(data,layer,node),b1)
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W2'%(data,layer,node),W2)
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b2'%(data,layer,node),b2)
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W3'%(data,layer,node),W3)
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b3'%(data,layer,node),b3)
                else:
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W1'%(data,layer,node),W1)
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b1'%(data,layer,node),b1)
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W2'%(data,layer,node),W2)
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b2'%(data,layer,node),b2)
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W3'%(data,layer,node),W3)
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b3'%(data,layer,node),b3)
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W4'%(data,layer,node),W4)
                    np.save('C:\\Users\\SHYang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b4'%(data,layer,node),b4)
# =============================================================================
#             plt.plot(acc_fpr, acc_tpr, 'b', label='AUC = %0.2f'% acc[7])
# =============================================================================
