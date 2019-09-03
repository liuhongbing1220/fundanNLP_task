#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:44:44 2019

@author: liuhongbing
"""

# coding=utf-8
import tensorflow as tf
import time
import os
from charCNN_model import CharCNN

import datetime
from read_data import Config,Dataset

# Load data
print("正在载入数据...")
# 函数dataset_read：输入文件名,返回训练集,测试集标签
# 注：embedding_w大小为vocabulary_size × embedding_size
#
#train_data = Dataset()
#train_data.dataset_read()
#

print("load the data finished....")


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        cnn = CharCNN(
            l0=Config.l0,
            num_classes=Config.nums_classes,
            alphabet_size=Config.alphabet_size, 
            convLayers=Config.convLayers,
            fcLayers=Config.fcLayers,
            l2_reg_lambda=0)
        
        # cnn = CharConvNet()
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(Config.learningRate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: Config.dropoutKeepProb
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)




        print("初始化完毕，开始训练")
        for i in range(Config.epoches):
            
            for j in range(train_data.example_nums// Config.batch_size):
                
                batch_train = train_data.next_batch()
                # 训练模型
                train_step(batch_train[0], batch_train[1])
                current_step = tf.train.global_step(sess, global_step)
               
                # train_step.run(feed_dict={x: batch_train[0], y_actual: batch_train[1], keep_prob: 0.5})
                # 对结果进行记录
                if current_step % Config.evaluateEvery == 0:
                    print("\nEvaluation:", current_step)
                    dev_step(train_data.dev_image, train_data.dev_label, writer=dev_summary_writer)
                    print("")
                if current_step % Config.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    
                    
"""
Evaluation: 100
2019-06-29T22:01:32.604023: step 100, loss 1.21269, acc 0.508651


Evaluation: 200
2019-06-29T22:08:41.874223: step 200, loss 1.18934, acc 0.517814

Evaluation: 300
2019-06-29T22:16:12.802301: step 300, loss 1.18706, acc 0.523965

Evaluation: 400
2019-06-29T22:23:01.126583: step 400, loss 1.18321, acc 0.532712

Evaluation: 500
2019-06-29T22:29:55.398533: step 500, loss 1.15435, acc 0.528835


Evaluation: 600
2019-06-29T22:37:12.734557: step 600, loss 1.1356, acc 0.539023

Evaluation: 700
2019-06-29T22:43:42.253525: step 700, loss 1.11152, acc 0.547898

Evaluation: 800
2019-06-29T22:50:57.971248: step 800, loss 1.10185, acc 0.55453



Evaluation: 1200
2019-06-29T23:18:18.294406: step 1200, loss 1.07737, acc 0.562636

Evaluation: 1500
2019-06-29T23:39:26.627900: step 1500, loss 1.0239, acc 0.582532


valuation: 1800
2019-06-29T23:59:47.539924: step 1800, loss 1.01305, acc 0.587402
"""