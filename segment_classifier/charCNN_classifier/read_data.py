#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:15:46 2019

@author: liuhongbing
"""


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


root = "/Users/liuhongbing/Documents/tensorflow/data/sentiment-analysis-on-movie-reviews/"


#data = pd.read_csv(root+"train.tsv",delimiter='\t')
#
#char_set = set([char.lower() for val in data['Phrase'] for char in val ])
#alphabet = ('').join(char_set)

    
class Config(object):

    alphabet = "a#r;5`6!xt:4mp*guo723dk8,$n'0jy1?ifhlc+svz/-.be 9q=&\\w"
    alphabet_size = 54
    
    l0 = 600  # 字符表示的序列长度
    nums_classes = 5
    ## model train parameter
    batch_size = 128
    train_test_split = 0.8  # 训练集的比例
    epoches = 10
    evaluateEvery = 100
    checkpoint_every = 100
    learningRate = 0.001
    
    convLayers = [[256, 7, 4],
                  [256, 7, 4],
                  [256, 7, 4]]
#                   [256, 3, None],
#                   [256, 3, None],
#                   [256, 3, 3]]
    fcLayers = [512, 512]
    dropoutKeepProb = 0.5
    
    epsilon = 1e-3  # BN层中防止分母为0而加入的极小值
    decay = 0.999  # BN层中用来计算滑动平均的值
    

class Dataset(object):
    def __init__(self):
        self.data_source = root + "Train.csv"
        self.index_in_epoch = 0
        self.alphabet = Config.alphabet
        self.num_classes = Config.nums_classes
        self.l0 = Config.l0
        self.epochs_completed = 0
        self.batch_size = Config.batch_size
        self.train_image = []
        self.dev_image = []
        self.train_label = []
        self.dev_label = []
        self.example_nums = 124848
     


    def next_batch(self):
        # 得到Dataset对象的batch
        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size
        if self.index_in_epoch > self.example_nums:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.example_nums)
            np.random.shuffle(perm)
            self.train_image = self.train_image[perm]
            self.train_label = self.train_label[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = self.batch_size
            assert self.batch_size <= self.example_nums
            
        end = self.index_in_epoch
        batch_x = np.array(self.train_image[start:end], dtype='int64')
        batch_y = np.array(self.train_label[start:end], dtype='float32')

        return batch_x, batch_y

    def dataset_read(self):
        # doc_vec表示一个一篇文章中的所有字母，doc_image代表所有文章
        # label_class代表分类
        # doc_count代表数据总共有多少行
        data = pd.read_csv(root+"train.tsv",delimiter='\t')
        docs = [content.lower() for content in data['Phrase']]
        labels = data['Sentiment']
        doc_count = len(data)
        # 引入embedding矩阵和字典
        print("引入嵌入词典和矩阵")
        embedding_w, embedding_dic = self.onehot_dic_build()
        # 将每个句子中的每个字母，转化为embedding矩阵的索引
        # 如：doc_vec表示一个一篇文章中的所有字母，doc_image代表所有文章
        doc_image = []
        label_image = []
        print("开始进行文档处理")
        for i in range(doc_count):
            doc_vec = self.doc_process(docs[i], embedding_dic)
            doc_image.append(doc_vec)
#            label_class = np.zeros(self.num_classes, dtype='float32')
#            label_class[int(labels[i]) - 1] = 1
            label_image=tf.keras.utils.to_categorical(labels, 5)
#            label_image.append(label_class)#
        #del embedding_w, embedding_dic
        print("求得训练集与测试集的tensor并赋值")
        doc_image = np.asarray(doc_image, dtype='int64')
        label_image = np.array(label_image, dtype='float32')  
        print("doc_imge:", len(doc_image))
        
        self.train_image, self.dev_image,self.train_label,self.dev_label = \
                train_test_split(doc_image,label_image, test_size=0.2, random_state=42)
                
        print("train_image",len(self.train_image))
        
        
        

    def doc_process(self, doc, embedding_dic):
        # 如果在embedding_dic中存在该词，那么就将该词的索引加入到doc的向量表示doc_vec中，不存在则用UNK代替
        # 不到l0的文章，进行填充，填UNK的value值，即0
        min_len = min(self.l0, len(doc))
        doc_vec = np.zeros(self.l0, dtype='int64')
        for j in range(min_len):
            if doc[j] in embedding_dic:
                doc_vec[j] = embedding_dic[doc[j]]
            else:
                doc_vec[j] = embedding_dic['UNK']
        return doc_vec


    def onehot_dic_build(self):
        # onehot编码
        alphabet = self.alphabet
        embedding_dic = {}
        embedding_w = []
        # 对于字母表中不存在的或者空的字符用全0向量代替
        embedding_dic["UNK"] = 0
        embedding_w.append(np.zeros(len(alphabet), dtype='float32'))

        for i, alpha in enumerate(alphabet):
            onehot = np.zeros(len(alphabet), dtype='float32')
            embedding_dic[alpha] = i + 1
            onehot[i] = 1
            embedding_w.append(onehot)

        embedding_w = np.array(embedding_w, dtype='float32')
        return embedding_w, embedding_dic
    
    
    
    