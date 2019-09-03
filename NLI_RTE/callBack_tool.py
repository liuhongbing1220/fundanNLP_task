#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:47:52 2019

@author: liuhongbing
"""

from sklearn.metrics import accuracy_score
from keras.callbacks import *
from keras.callbacks import TensorBoard
import logging
import numpy as np



def compute_acc(X, Y, Z, vocab, model, batch_size):
    scores = model.predict([X, Y], batch_size=batch_size)
    prediction = np.zeros(scores.shape)
    for i in range(scores.shape[0]):
        l = np.argmax(scores[i])
        prediction[i][l] = 1.0
    assert np.array_equal(np.ones(prediction.shape[0]), np.sum(prediction, axis=1))
    plabels = np.argmax(prediction, axis=1)
    tlabels = np.argmax(Z, axis=1)
    acc = accuracy_score(tlabels, plabels)
    
    return acc


class AccCallBack(Callback):
    def __init__(self, xtrain, ytrain, ztrain, xdev, ydev,zdev, xtest, ytest,ztest, vocab, batch_size):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.ztrain = ztrain
        self.xdev = xdev
        self.ydev = ydev
        self.zdev = zdev
        self.xtest = xtest
        self.ytest = ytest
        self.ztest = ztest
        self.vocab = vocab
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        train_acc = compute_acc(self.xtrain, self.ytrain, self.ztrain, self.vocab, self.model, self.batch_size)
        dev_acc = compute_acc(self.xdev, self.ydev, self.zdev, self.vocab, self.model, self.batch_size)
        test_acc = compute_acc(self.xtest, self.ytest, self.ztest, self.vocab, self.model, self.batch_size)
        logging.info('----------------------------------')
        logging.info('Epoch ' + str(epoch) + ' train loss:' + str(logs.get('loss')) + ' - Validation loss: ' + str(
            logs.get('val_loss')) + ' train acc: ' + str(train_acc) + ' dev acc: ' + str(
            dev_acc) +  'test acc: ' + str(test_acc) )
        logging.info('----------------------------------')
        
        
       

def TensorBoardCallBack(FILE_DIR):
    tb = TensorBoard(log_dir=FILE_DIR,  # 日志文件保存位置
                    histogram_freq=1,  # 按照何等频率（每多少个epoch计算一次）来计算直方图，0为不计算
                    batch_size=32,     # 用多大量的数据计算直方图
                    write_graph=True,     # 是否在tensorboard中可视化计算图
                    write_grads=False,    # 是否在tensorboard中可视化梯度直方图
                    write_images=False,   # 是否在tensorboard中以图像形式可视化模型权重
                    update_freq='batch')   # 更新频率)
    return tb
                    







