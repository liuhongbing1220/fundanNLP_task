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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,Embedding,Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing import sequence
import io


root = "/Users/liuhongbing/Documents/tensorflow/data/sentiment-analysis-on-movie-reviews/"


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

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = [float(tokens[i]) for i in range(1, len(tokens))]
    return data


def getWordVector(word2VecSum, word2index, embedding_dim):
    
    embedding_weight = np.zeros((len(word2index)+1, embedding_dim))
    for word,index in word2index.items():
        try:
            embedding_weight[index, :] = word2VecSum[word]
        except KeyError:
            pass
        
    return embedding_weight
        

word2vecSum = load_vectors('/Users/liuhongbing/Documents/tensorflow/fudanNLP/word_to_vector/wiki-news-300d-1M.vec')


#
#def calc_sim_word(word1, word2):
#    if (word1 not in word2vecSum) | (word2 not in word2vecSum):
#        return "word not in wordvec"
#    else:
#        vec1 = word2vecSum[word1]
#        vec2 = word2vecSum[word2]
#        return np.dot(vec1, vec2)
        
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

columns = ['feat_'+str(i) for i in range(48)]

train_Data2 = train_Data.copy()
train_Data2 = pd.DataFrame(train_Data2)
train_Data2.columns = columns
train_Data2['label'] = totalLabel

train_Data2.to_csv(root+"data_result.csv", index=None)

train_label=tf.keras.utils.to_categorical(totalLabel,5)

data_train, data_test, label_train, label_test = train_test_split(train_Data,train_label, test_size = 0.2)




max_features=len(train_tokenizer.index_word)+1
max_len=48  #这个是要和前面padding时的长度一致
epochs = 5  #训练次数
emb_dim = 300 #128代表embedding层的向量维度
batch_size=80   #这是指定批量的大小


model = Sequential()

embedding_weight = getWordVector(word2vecSum,train_tokenizer.word_index, emb_dim)
embedding_weight_copy = embedding_weight.copy()


#import pickle
#with open(root+"tikenizer_word_index.pkl", 'wb') as f:
#    pickle.dump(train_tokenizer.word_index, f)
##  15126*128


##model.add(Embedding(max_features,emb_dim, mask_zero=True))
model.add(Embedding(max_features,emb_dim, weights=[embedding_weight], trainable=False, mask_zero=True))
## 【（128+64）*64+64】*4
model.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
## 【（64+32）*32+32】*4
model.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))

## 32*5 + 5
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(data_train,label_train,batch_size=batch_size,epochs=epochs,validation_data=(data_test,label_test))

model.evaluate(data_test, label_test)
#
#print("start saver test submission")
#y_pred=model.predict_classes(test_Data)
#
#sub_file = pd.read_csv(root + 'sampleSubmission.csv', sep=',')
#sub_file.Sentiment=y_pred
#sub_file.to_csv('Submission_2.csv',index=False)

model.save('/Users/liuhongbing/Documents/tensorflow/fudanNLP/model_2.h5')












