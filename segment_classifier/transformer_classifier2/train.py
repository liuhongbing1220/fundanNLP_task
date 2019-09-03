#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:4:11 2019

@author: liuhongbing
"""

from __future__ import print_function
import tensorflow as tf
import argparse
from modules import *
import numpy as np
from model import TransformerDecoder
import tensorflow_load_csv as lc


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, default='corpora/data')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    parser.add_argument('--logdir', type=str, default='./ckpt_mlp')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_batches', type=int, default=30000)
    parser.add_argument('--save_every', type=int, default=3000)
    parser.add_argument('--maxlen', type=int, default=48)
    parser.add_argument('--hidden_units', type=int, default=300)
    parser.add_argument('--hidden_units_concat', type=int, default=348)

    parser.add_argument('--weighted_loss', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=9)

    
    args = parser.parse_args()
    print("load model")
    # Construct graph
    model = TransformerDecoder(is_training=True, args=args)
    print("Graph loaded")
    
    datapath  = '/Users/liuhongbing/Documents/tensorflow/data/'

    # Start session
    with tf.Session(graph=model.graph) as sess:
        
        nextBatch_reader, nextBatchLabels_reader = lc.csvreader(datapath+"sentiment-analysis-on-movie-reviews/dataframe_result/", 48)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.ckpt_path)
        
        if ckpt:
            print("restoring from {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        gs = sess.run(model.global_step)
        for step in range(args.num_batches):
            
            print(f"start sess run {step} iter data")
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            nextBatch, nextBatchLabels2 = sess.run([nextBatch_reader, nextBatchLabels_reader])
                
                
            nextBatchLabels=tf.keras.utils.to_categorical(nextBatchLabels2,5)
            
#            print("nextBatch", nextBatch.shape)
#            print("nextBatchLabels",nextBatchLabels.shape)
            [_, mean_loss, loss, preds] = sess.run([model.train_op, model.mean_loss, model.merged, model.preds],
                                                   feed_dict={
                                                       model.x: nextBatch,
                                                       model.y: nextBatchLabels
                                                   })
            if step % 10 == 0:
                acc = 1.0 - 1.0*np.count_nonzero(([int(preds[i]-nextBatchLabels2[i]) for i in range(len(nextBatchLabels2))]))/len(nextBatchLabels2)
                model.train_writer.add_summary(loss, gs + step)
                print("acc = {:.4f}".format(acc))
            print("step = {}/{}, loss = {:.4f}".format(step + 1, args.num_batches, mean_loss))
            if (step + 1) % args.save_every == 0 or step + 1 == args.num_batches:
                gs = sess.run(model.global_step)
                saver.save(sess, args.logdir + '/model_gs_%d' % (gs))

    print("Done")