# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 19:51:29 2017

@author: Yang
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn import metrics

# implementation for test set

#with tf.device('/device:GPU:0'):
# load the test set
test = np.load('C:\\Users\\\Yang\\Desktop\\kaggle\\mush_test.npy')

X_test = test[:,0:test.shape[1]-1]
Y_test = test[:,test.shape[1]-1]
Y_test = np.reshape(Y_test, [len(Y_test), 1])

ohe = OneHotEncoder()
ohe.fit(Y_test)
Y_test = ohe.transform(Y_test).toarray()

# store the accuracy, the number of iteration and weights
# row 1: data, 2: layer, 3: node, 4: tpr/fpr, 5: precision, 6: recall,
#     7: f1_score, 8: fpr, 9: tpr, 10: tresholds
best = np.zeros((10,1))
i = 0

for data in [3]:
    
    # load dataset
    dataset = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\dataset%d.npy'%(data))
   
    # extract the mean and standard variation from jkgj
    std_scale = preprocessing.StandardScaler().fit(dataset[:,0:22])
    
    X_test = std_scale.transform(X_test)
    
    for layer in [1,2,3]:
        for node in [32,64,96,128]:
            if layer == 1:
                # load the weight and bias
                W1 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W1.npy'
                     %(data,layer,node))
                W2 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W2.npy'
                     %(data,layer,node))
                
                b1 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b1.npy'
                     %(data,layer,node))
                b2 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b2.npy'
                     %(data,layer,node))
                
            elif layer == 2:
                W1 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W1.npy'
                     %(data,layer,node))
                W2 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W2.npy'
                     %(data,layer,node))
                W3 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W3.npy'
                     %(data,layer,node))
                
                b1 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b1.npy'
                     %(data,layer,node))
                b2 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b2.npy'
                     %(data,layer,node))
                b3 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b3.npy'
                     %(data,layer,node))
                
            else:
                W1 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W1.npy'
                     %(data,layer,node))
                W2 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W2.npy'
                     %(data,layer,node))
                W3 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W3.npy'
                     %(data,layer,node))
                W4 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_W4.npy'
                     %(data,layer,node))
                
                b1 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b1.npy'
                     %(data,layer,node))
                b2 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b2.npy'
                     %(data,layer,node))
                b3 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b3.npy'
                     %(data,layer,node))
                b4 = np.load('C:\\Users\\Yang\\Desktop\\kaggle\\model\\model_nn_%d_%d_%d_b4.npy'
                     %(data,layer,node))
            
            x = tf.placeholder("float", [None, X_test.shape[1]])
            y = tf.placeholder("float", [None, Y_test.shape[1]])
            
            W_1 = tf.placeholder("float", [None, None])
            W_2 = tf.placeholder("float", [None, None])
            W_3 = tf.placeholder("float", [None, None])
            W_4 = tf.placeholder("float", [None, None])
            
            b_1 = tf.placeholder("float", [None])
            b_2 = tf.placeholder("float", [None])
            b_3 = tf.placeholder("float", [None])
            b_4 = tf.placeholder("float", [None])
            
            if layer == 1:
                z_1 = tf.add(tf.matmul(x, W_1), b_1)
                h_1 = tf.nn.relu(z_1)
                h_1_drop = tf.nn.dropout(h_1, keep_prob=0.5)
                
                output1 = tf.add(tf.matmul(h_1_drop, W_2), b_2)
            
            elif layer == 2:
                z_1 = tf.add(tf.matmul(x, W_1), b_1)        
                h_1 = tf.nn.relu(z_1) 
                h_1_drop = tf.nn.dropout(h_1, keep_prob=0.5)
                
                z_2 = tf.add(tf.matmul(h_1_drop, W_2), b_2)
                h_2 = tf.nn.relu(z_2)
                h_2_drop = tf.nn.dropout(h_2, keep_prob=0.5)
                
                output2 = tf.add(tf.matmul(h_2_drop, W_3), b_3)
                
            else:
                z_1 = tf.add(tf.matmul(x, W_1), b_1)        
                h_1 = tf.nn.relu(z_1) 
                h_1_drop = tf.nn.dropout(h_1, keep_prob=0.5)
                
                z_2 = tf.add(tf.matmul(h_1_drop, W_2), b_2)
                h_2 = tf.nn.relu(z_2)
                h_2_drop = tf.nn.dropout(h_2, keep_prob=0.5)
                
                z_3 = tf.add(tf.matmul(h_2_drop, W_3), b_3)
                h_3 = tf.nn.relu(z_3) 
                h_3_drop = tf.nn.dropout(h_3, keep_prob=0.5)
                
                output3 = tf.add(tf.matmul(h_3_drop, W_4), b_4)
                
            if layer == 1:
                y_p = tf.argmax(output1, 1)
            elif layer == 2:
                y_p = tf.argmax(output2, 1)
            else:
                y_p = tf.argmax(output3, 1)
            
            sess = tf.Session()
            
            if layer == 1:
                y_pred = sess.run(y_p, feed_dict={x:X_test, y:Y_test,
                                                  W_1:W1, W_2:W2, b_1:b1, b_2:b2})
            elif layer == 2:
                y_pred = sess.run(y_p, feed_dict={x:X_test, y:Y_test,
                                                  W_1:W1, W_2:W2, b_1:b1, b_2:b2,
                                                  W_3:W3, b_3:b3})
            else:
                y_pred = sess.run(y_p, feed_dict={x:X_test, y:Y_test,
                                                  W_1:W1, W_2:W2, b_1:b1, b_2:b2,
                                                  W_3:W3, b_3:b3, W_4:W4, b_4:b4})
                
            y_true = np.argmax(Y_test,1)
            precision = metrics.precision_score(y_true, y_pred)
            recall = metrics.recall_score(y_true, y_pred)
            f1_score = metrics.f1_score(y_true, y_pred)
            confusion = metrics.confusion_matrix(y_true, y_pred)
            fpr, tpr, tresholds = metrics.roc_curve(y_true, y_pred)
            auc = metrics.auc(fpr, tpr)
            print (auc)
            
            if i == 0:
                best[:,0] = [data, layer, node, tpr[1]-fpr[1], precision, recall,
                     f1_score, fpr[1], tpr[1], auc]
                
                confusion_matrix = confusion
                
                acc_fpr = fpr
                acc_tpr = tpr
                
                acc_auc = auc
                print('data: %d, layer: %d, node: %d, AUC: %g'
                      %(data,layer,node,acc_auc))
                    
                i = 1
                    
            else:
                if auc >= best[9,0]:
                    best[:,0] = [data, layer, node, tpr[1]-fpr[1], precision, recall, 
                         f1_score, fpr[1], tpr[1], auc]
                    
                    confusion_matrix = confusion
                    
                    acc_fpr = fpr
                    acc_tpr = tpr
                    
                    acc_auc = auc
                    print('data: %d, layer: %d, node: %d, AUC: %g'
                          %(data,layer,node,acc_auc))
                    
                else:
                    pass
                
np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best.npy', best)
np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_conf.npy', confusion_matrix)
if best[1,0] == 1:
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_W1.npy', W1)
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_b1.npy', b1)
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_W2.npy', W2)
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_b2.npy', b2)
elif best[1,0] == 2:
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_W1.npy', W1)
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_b1.npy', b1)
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_W2.npy', W2)
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_b2.npy', b2)
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_W3.npy', W3)
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_b3.npy', b3)
else:
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_W1.npy', W1)
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_b1.npy', b1)
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_W2.npy', W2)
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_b2.npy', b2)
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_W3.npy', W3)
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_b3.npy', b3)
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_W4.npy', W4)
    np.save('C:\\Users\\Yang\\Desktop\\kaggle\\best\\best_b4.npy', b4)