#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:34:27 2019

@author: liuhongbing
"""

from modules import positional_encoding,multihead_attention,feedforward,feedforwardMLP,fixedPositionEmbedding
import numpy as np
import tensorflow_load_csv as lc
import tensorflow as tf
import load_word_vector



class TransformerDecoder:
    def __init__(self, is_training=True, args=None):
        
        self.embeddings = load_word_vector.getW2VEmbedding()
        
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            self.x = tf.placeholder(tf.int32, shape=(None, args.maxlen))

            if is_training:
                self.y = tf.placeholder(tf.int32, shape=(None, 5))

            
            with tf.variable_scope("decoder"):
                
                self.dec = tf.nn.embedding_lookup(self.embeddings, self.x) 
                
#                if args.sinusoid:
#                self.dec += positional_encoding(self.x,
#                                                num_units=args.hidden_units,
#                                                maxlen = args.maxlen,
#                                                zero_pad=False,
#                                                scale=False,
#                                                scope="dec_pe")
                
                posemb = fixedPositionEmbedding(args.batch_size, args.maxlen)
                self.dec = tf.concat([self.dec, posemb], axis=-1)
#                else:
#                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0),
#                                                  [tf.shape(self.x)[0], 1]),
#                                          vocab_size=args.maxlen,
#                                          num_units=args.hidden_units,
#                                          zero_pad=False,
#                                          scale=False,
#                                          scope="dec_pe")[0]

                ## Dropout
                self.dec = tf.layers.dropout(self.dec,
                                             rate=args.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(args.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.dec,
                                                       values=self.dec,
#                                                       num_units = args.hidden_units ,
                                                       num_units = args.hidden_units_concat,
                                                       num_heads=args.num_heads,
                                                       dropout_rate=args.dropout_rate,
                                                       training=is_training,
                                                       causality=False,
                                                       scope="self_attention")

#                        self.dec = feedforward(self.dec, num_units=[4 * args.hidden_units, args.hidden_units])
#                        self.dec = feedforwardMLP(self.dec, num_units=[4 * args.hidden_units, args.hidden_units])
                        self.dec = feedforwardMLP(self.dec, num_units=[4 * args.hidden_units, args.hidden_units_concat])
                
                
#                self.proj = tf.get_variable("proj", [args.num_classes, args.hidden_units * args.maxlen])
#                
#                self.logits = tf.matmul(tf.reshape(self.dec, [-1, args.hidden_units * args.maxlen]),
#                                        self.proj, transpose_b=True)
                        
                self.proj = tf.get_variable("proj", [args.num_classes, args.hidden_units_concat * args.maxlen])
                
                self.logits = tf.matmul(tf.reshape(self.dec, [-1, args.hidden_units_concat * args.maxlen]),
                                        self.proj, transpose_b=True)




            print(self.logits)
            # self.logits = tf.layers.dense(self.dec, len(word2idx))
            self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))


            if is_training:

                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
                
                self.mean_loss = tf.reduce_mean(self.loss)

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

                # Summary
                tf.summary.scalar('loss', self.mean_loss)
                self.merged = tf.summary.merge_all()
                self.train_writer = tf.summary.FileWriter(args.logdir, self.graph)