#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:24:14 2019

@author: liuhongbing
"""

import keras
from keras.layers import Embedding,Input,SpatialDropout1D,Bidirectional,LSTM,Dot,Lambda,Permute,subtract,multiply,GlobalAveragePooling1D,GlobalMaxPooling1D,Dense,Dropout
from keras.activations import softmax
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from reader import load_data,get_vocab
from callBack_tool import AccCallBack,TensorBoardCallBack
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from datetime import datetime
import logging


def get_ESIM_model(nb_words, embedding_dim, embedding_matrix, recurrent_units, dense_units, dropout_rate, max_sequence_length, out_size):
    embedding_layer = Embedding(nb_words,
                                embedding_dim,
                                # embeddings_initializer='uniform',
                                #weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                #trainable=False
                                )

    input_q1_layer = Input(shape=(max_sequence_length,), dtype='int32', name='q1')
    input_q2_layer = Input(shape=(max_sequence_length,), dtype='int32', name='q2')

    embedding_sequence_q1 = BatchNormalization(axis=2)(embedding_layer(input_q1_layer))
    embedding_sequence_q2 = BatchNormalization(axis=2)(embedding_layer(input_q2_layer))

    final_embedding_sequence_q1 = SpatialDropout1D(0.25)(embedding_sequence_q1)
    final_embedding_sequence_q2 = SpatialDropout1D(0.25)(embedding_sequence_q2)

    rnn_layer_q1 = Bidirectional(LSTM(recurrent_units, return_sequences=True))(final_embedding_sequence_q1)
    rnn_layer_q2 = Bidirectional(LSTM(recurrent_units, return_sequences=True))(final_embedding_sequence_q2)
    
    ## embedding * embedding
    attention = Dot(axes=-1)([rnn_layer_q1, rnn_layer_q2])
    
    print('attention:', attention)
    
    w_attn_1 = Lambda(lambda x: softmax(x, axis=1))(attention) ## 列归一化
    w_attn_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))##行归一化
    
    align_layer_1 = Dot(axes=1)([w_attn_1, rnn_layer_q1])
    align_layer_2 = Dot(axes=1)([w_attn_2, rnn_layer_q2])
    
    print('align_layer_1:', align_layer_1)
    print('align_layer_1:', align_layer_2)
    
    subtract_layer_1 = subtract([rnn_layer_q1, align_layer_1])
    subtract_layer_2 = subtract([rnn_layer_q2, align_layer_2])

    multiply_layer_1 = multiply([rnn_layer_q1, align_layer_1])
    multiply_layer_2 = multiply([rnn_layer_q2, align_layer_2])

    m_q1 = concatenate([rnn_layer_q1, align_layer_1, subtract_layer_1, multiply_layer_1])
    m_q2 = concatenate([rnn_layer_q2, align_layer_2, subtract_layer_2, multiply_layer_2])

    v_q1_i = Bidirectional(LSTM(recurrent_units, return_sequences=True))(m_q1)
    v_q2_i = Bidirectional(LSTM(recurrent_units, return_sequences=True))(m_q2)

    avgpool_q1 = GlobalAveragePooling1D()(v_q1_i)
    avgpool_q2 = GlobalAveragePooling1D()(v_q2_i)
    
    maxpool_q1 = GlobalMaxPooling1D()(v_q1_i)
    maxpool_q2 = GlobalMaxPooling1D()(v_q2_i)

    merged_q1 = concatenate([avgpool_q1, maxpool_q1])
    merged_q2 = concatenate([avgpool_q2, maxpool_q2])

    final_v = BatchNormalization()(concatenate([merged_q1, merged_q2]))
    
    output = Dense(units=dense_units, activation='relu')(final_v)
    output = BatchNormalization()(output)
    output = Dropout(dropout_rate)(output)
    output = Dense(units=out_size, activation='softmax')(output)

    model = Model(inputs=[input_q1_layer, input_q2_layer], output=output)
    adam_optimizer = keras.optimizers.Adam(lr=1e-3, decay=1e-6, clipvalue=5)
#    parallel_model = multi_gpu_model(model, gpus=2)
#    parallel_model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['binary_crossentropy', 'accuracy'])
    
    print(model.summary())
    # plot(model, 'model.png')
    # # model.compile(loss={'output':'binary_crossentropy'}, optimizer=Adam())
    # model.compile(loss={'output':'categorical_crossentropy'}, optimizer=Adam(options.lr))
    model.compile(loss='categorical_crossentropy',optimizer= adam_optimizer)
    
    return model


def setup_logger(config_str):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=datetime.now().strftime('mylogfile_%H_%M_%d_%m_%Y.log'),
                        filemode='w')

def save_model(model, wtpath, archpath, mode='yaml'):
    if mode == 'yaml':
        yaml_string = model.to_yaml()
        open(archpath, 'w').write(yaml_string)
    else:
        with open(archpath, 'w') as f:
            f.write(model.to_json())
    model.save_weights(wtpath)
    

if __name__ == "__main__":
    
#    tf.reset_default_graph()
    root="/Users/liuhongbing/Documents/tensorflow/data/snli_1.0/"
    train=[l.strip().split('\t') for l in open(root+'snli_1.0_train.txt')]
    dev=[l.strip().split('\t') for l in open(root+'snli_1.0_dev.txt')]
    test=[l.strip().split('\t') for l in open(root+'snli_1.0_test.txt')]
    vocab = get_vocab(train)
    print("vocab (incr. maxfeatures accordingly):",len(vocab))
    
    X_train,Y_train,Z_train=load_data(train,vocab)
    X_dev,Y_dev,Z_dev=load_data(dev,vocab)
    X_test,Y_test,Z_test=load_data(test,vocab)
    print('Build model...')


    MODEL_ARCH = root+"/ESIM/arch_att.yaml"
    MODEL_WGHT = root+"/ESIM/weights_att.weights"
    MODEL_TENSORBOARD = root+"/ESIM/logs"

    MAXLEN = 20
    X_train = pad_sequences(X_train, maxlen=MAXLEN, value=vocab["unk"], padding='pre')
    X_dev = pad_sequences(X_dev, maxlen=MAXLEN, value=vocab["unk"], padding='pre')
    X_test = pad_sequences(X_test, maxlen=MAXLEN, value=vocab["unk"], padding='pre')
    
    Y_train = pad_sequences(Y_train, maxlen=MAXLEN, value=vocab["unk"], padding='post')
    Y_dev = pad_sequences(Y_dev, maxlen=MAXLEN, value=vocab["unk"], padding='post')
    Y_test = pad_sequences(Y_test, maxlen=MAXLEN, value=vocab["unk"], padding='post')


    Z_train = to_categorical(Z_train, num_classes=3)
    Z_dev = to_categorical(Z_dev, num_classes=3)
    Z_test = to_categorical(Z_test, num_classes=3)

    print(X_train.shape, Y_train.shape)
   
    setup_logger("start ESIM Logger ...")

    print('Build model...')
       
    model = get_ESIM_model(nb_words=43444, 
                           embedding_dim=150, 
                           embedding_matrix=None, 
                           recurrent_units=150, 
                           dense_units=256, 
                           dropout_rate=0.5, 
                           max_sequence_length=20, 
                           out_size=3)

    logging.info('start train....')
    logging.info(
        "train size: " + str(len(X_train)) + " dev size: " + str(len(X_dev)) + " test size: " + str(len(X_test)))
    
   
    history = model.fit([X_train,Y_train], Z_train,
                        batch_size=32,
                        nb_epoch=20,
                        validation_data=([X_dev,Y_dev], Z_dev),
                        callbacks=[
                            AccCallBack(X_train, Y_train, Z_train, X_dev, Y_dev, Z_dev, X_test,Y_test, Z_test, vocab, batch_size=32),
                            TensorBoardCallBack(MODEL_TENSORBOARD)]
                        )
    save_model(model, MODEL_WGHT, MODEL_ARCH)
    
    """
    ESIM----result
    loss: 0.7718 - val_loss: 0.6444
    loss: 0.6359 - val_loss: 0.5951
    loss: 0.5867 - val_loss: 0.5807
    """





















