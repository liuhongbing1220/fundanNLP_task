#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 15:46:12 2019

@author: liuhongbing
"""

import tensorflow as tf
import numpy as np
import pickle
import io
import tensorflow_load_csv as lc


datapath = '/Users/liuhongbing/Documents/tensorflow/data/'
'''
读取全部对word2vec
'''
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = [float(tokens[i]) for i in range(1, len(tokens))]
    return data

'''
选取需要的word2vec
'''
def getWordVector(word2VecSum, word2index, embedding_dim):
    
    embedding_weight = np.zeros((len(word2index)+1, embedding_dim))
    for word,index in word2index.items():
        try:
            embedding_weight[index, :] = word2VecSum[word]
        except KeyError:
            pass
        
    return embedding_weight

    
    
def getW2VEmbedding():
    
    with open(datapath + "word_to_vector/tikenizer_word_index.pkl", 'rb') as f:
        word2index  = pickle.load(f)    
        
    word2vecSum = load_vectors(datapath +'word_to_vector/wiki-news-300d-1M.vec')
    word2vec = getWordVector(word2vecSum, word2index, 300)
    return word2vec



