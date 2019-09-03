#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 21:51:11 2019

@author: liuhongbing
"""

import tensorflow as tf
import numpy as np
import pickle
import io
import time
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
 
with open(datapath + "word_to_vector/tikenizer_word_index.pkl", 'rb') as f:
    word2index  = pickle.load(f)       
#
#baseballIndex = word2index['baseball']
#word2vec[baseballIndex]
#
#firstSentence = np.zeros((5), dtype='int32')
#firstSentence[0] = word2index["i"]
#firstSentence[1] = word2index["thought"]
#firstSentence[2] = word2index["the"]
#firstSentence[3] = word2index["movie"]
#firstSentence[4] = word2index["was"]
#with tf.Session() as sess:
#    print(tf.nn.embedding_lookup(word2vec,firstSentence).eval())
#


max_features=15127
max_len=48  #这个是要和前面padding时的长度一致
epochs = 5  #训练次数
emb_dim = 300 #128代表embedding层的向量维度
batch_size=200   #这是指定批量的大小
numClasses = 5
lstmUnits = 64
iterations = 50000

word2vecSum = load_vectors(datapath +'word_to_vector/wiki-news-300d-1M.vec')

word2vec = getWordVector(word2vecSum, word2index, emb_dim)


tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batch_size, numClasses])
input_data = tf.placeholder(tf.int32, [batch_size, max_len])

weight_embedding = tf.Variable(tf.zeros([batch_size, max_len, emb_dim]))

weight_embedding = tf.nn.embedding_lookup(word2vec, input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)


value, _ = tf.nn.dynamic_rnn(lstmCell, weight_embedding, dtype=tf.float64)


weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses], dtype=tf.float64))

last = tf.cast(value[:,-1,:],tf.float64)
#取最终的结果值
weight = tf.cast(weight, tf.float64)

prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float64))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())


nextBatch_reader, nextBatchLabels_reader = lc.csvreader(datapath+"sentiment-analysis-on-movie-reviews/dataframe_result/", 48)

timestart = time.time()
for i in range(iterations):
    #Next Batch of reviews
    print(f"start sess run {i} iter data")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    nextBatch, nextBatchLabels = sess.run([nextBatch_reader, nextBatchLabels_reader])
    
    
    nextBatchLabels=tf.keras.utils.to_categorical(nextBatchLabels,5)
    print("run optimizer:", len(nextBatch))
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels}) 
    print(f"run optimizer {i} end")
    if (i % 1000 == 0 and i != 0):
        loss_ = sess.run(loss, {input_data: nextBatch, labels: nextBatchLabels})
        accuracy_ = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})
        
        print("iteration {}/{}...".format(i+1, iterations),
              "loss {}...".format(loss_),
              "accuracy {}...".format(accuracy_))    
    #Save the network every 10,000 training iterations
    if (i % 10000 == 0 and i != 0):
        save_path = saver.save(sess, datapath+ "sentiment-analysis-on-movie-reviews/model_lstm_lay2/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)
        
    print(f"iter {i}, time:", (time.time()-timestart))
    print("\n")

        
        
        
        