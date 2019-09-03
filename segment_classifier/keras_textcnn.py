#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:49:27 2019

@author: liuhongbing
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding

root = "/Users/liuhongbing/Documents/tensorflow/fudanNLP/sentiment-analysis-on-movie-reviews/"

def dataCleaning(totalText):
    total=[]
    for i in totalText:
        temp=i.lower()
        temp=re.sub('[^a-zA-Z]',' ',temp)
        tempArr=[j for j in temp.strip().split('\t') if isinstance(j,str)]          
        tstr=' '.join(tempArr)
        total.append(tstr)
    return total          


def getTest(name):    

    data=pd.read_csv(name,delimiter='\t')
    totalText=data['Phrase']
    totalText=dataCleaning(totalText)
    return totalText


def loadData(name):    
    data=pd.read_csv(name,delimiter='\t')
    totalText=data['Phrase']
    totalText=dataCleaning(totalText)
    totalLabel=data['Sentiment']
    return totalText,totalLabel        

totalText,totalLabel = loadData(root+"train.tsv")
testText = getTest(root+"test.tsv")


train_tokenizer = Tokenizer()
train_tokenizer.fit_on_texts(totalText)

train_sequences = train_tokenizer.texts_to_sequences(totalText)
test_sequences = train_tokenizer.texts_to_sequences(testText)

# 获得所有tokens的长度
num_tokens = [ len(tokens) for tokens in train_sequences ]
num_tokens = np.array(num_tokens)
print(len(num_tokens))
#输出  156060
# 平均tokens的长度
print('mean',np.mean(num_tokens))
# 最长的评价tokens的长度
print('max',np.max(num_tokens))
# 最长的评价tokens的长度
print('min',np.min(num_tokens))

plt.hist((num_tokens), bins = 50)
plt.ylabel('number of tokens')
plt.xlabel('length of tokens')
plt.title('Distribution of tokens length')
plt.show()

'''
3sigma原则
'''
max_tokens = np.mean(num_tokens) + 3 * np.std(num_tokens)
max_tokens = int(max_tokens)
max_tokens
# 取tokens的长度为19时，大约93%的样本被涵盖
#np.sum( num_tokens < max_tokens ) / len(num_tokens)

train_Data=pad_sequences(train_sequences,maxlen=48)
test_Data = pad_sequences(test_sequences,maxlen=48)

train_label=tf.keras.utils.to_categorical(totalLabel,5)

data_train, data_test, label_train, label_test = train_test_split(train_Data,train_label, test_size = 0.2)

max_features=len(train_tokenizer.index_word)+1
max_len=48  #这个是要和前面padding时的长度一致
epochs = 5  #训练次数
emb_dim = 128 #128代表embedding层的向量维度
batch_size=80   #这是指定批量的大小
class_num = 5


# coding=utf-8

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential

input = Input((max_len,))
 Embedding part can try multichannel as same as origin paper
embedding = Embedding(max_features, emb_dim, input_length=max_len)(input)
convs = []
for kernel_size in [3, 4, 5]:
    c = Conv1D(128, kernel_size, activation='relu')(embedding)
    c = GlobalMaxPooling1D()(c)
    convs.append(c)
x = Concatenate()(convs)

output = Dense(class_num, activation='softmax')(x)
model = Model(inputs=input, outputs=output)


model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])



print('Train...')
early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')

model.fit(data_train, label_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping],
          validation_data=(data_test, label_test))




"""
sequences 操作
只能一层：textCNN
"""
#model = Sequential()
#model.add(Embedding(max_features,emb_dim))
#model.add(Conv1D(128, kernel_size = 3, activation='relu'))
#model.add(MaxPooling1D(max_len-3+1))
#
#
#model.add(Dense(5, activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#print(model.summary())
#
#model.fit(data_train,label_train,batch_size=batch_size,epochs=epochs,validation_data=(data_test,label_test))
#
