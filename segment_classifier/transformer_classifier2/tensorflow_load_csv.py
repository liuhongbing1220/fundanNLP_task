#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:24:17 2019

@author: liuhongbing
"""


import tensorflow as tf
import os


def csvreader(root, NumFeatures):
## 文件名列表
    files_name = os.listdir(root)

    filename_list = [os.path.join(root, file) for file in files_name]
    
    file_queue = tf.train.string_input_producer(filename_list)
    
    ## 构造csv阅读器
    reader = tf.TextLineReader(skip_header_lines=1)
    _, value = reader.read(file_queue)
    record_defaults = [[0] for _ in range(NumFeatures)]
    record_defaults.append([0])
    
    data = tf.decode_csv(value, record_defaults = record_defaults)
    
    feature = data[:-1]
    label = data[-1]
    
    feature_batch, label_batch = tf.train.shuffle_batch([feature, label], batch_size = 256, 
                                                        capacity = 1000, 
                                                        num_threads=32,
                                                        min_after_dequeue=200)
    
    return feature_batch, label_batch


#
#if __name__ == '__main__':
#    
#    root = '/Users/liuhongbing/Documents/tensorflow/fudanNLP/segment_classifier/data/'
#    feature_batch, label_batch = csvreader(root, 48)
#
#    with tf.Session() as sess:
#        coord = tf.train.Coordinator()
#        threads = tf.train.start_queue_runners(sess, coord)
#        for i in range(2):
#            fea_batch, lab_batch = sess.run([feature_batch, label_batch])
#            print(fea_batch)
#            print("-----------------")
#        
#        coord.request_stop()
#        coord.join(threads)
    
   


