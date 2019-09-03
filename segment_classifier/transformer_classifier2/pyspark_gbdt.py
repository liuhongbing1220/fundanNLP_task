#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:32:32 2019

@author: liuhongbing
"""


data = spark.sql("select * from table")
data = data.na.fill(0)

#feature_list = data.columns[2:-12]
feature_list = ['feature_1','feature_2','feature_3','...']
label = ['label']

"""
每行数据清洗
"""
def clean_data(line) :
    new_result = []
    for item in line:
        if item is None: ## 有些空值，fill不能填充 
            item = 0
        new_value = float(item)
        
        if new_value < 10e-5:
            new_result.append(0)
        else:
            new_result.append(item)
    return new_result


data_train = data.select(label+feature_list).rdd.map(clean_data)
data_train_label_df = spark.createDataFrame(data_train, label+feature_list)
data_train_label_df = data_train_label_df.na.fill(0)


def clean_label(item):
    item = float(item)
    if item>=16.0:
        return 0
    else:
        return 1
    
    
import random

def sample_rate(currate_rate):
    base_line = 1.5
    if currate_rate <=1.0:
        return 0.0
    elif currate_rate>1.0 and currate_rate<5.0:
        return random.uniform(base_line, currate_rate+1.0)
    else:
        return random.uniform(base_line+1.0, currate_rate+2.0)
    
    
def over_sampling(data):
    l0_rdd = data.rdd.filter(lambda line:line[1]==0)
    l1_rdd = data.rdd.filter(lambda line:line[1]==1)
    l0_count = l0_rdd.count()
    l1_count = l1_rdd.count()
    
    max_count = max(l0_count, l1_count)
    min_count = min(l0_count, l1_count)
    
    if min_count ==0:
        raise RuntimeError("one label count is zero", 'l0_count:',l0_count,'l1_count:',l1_count)
        
    l0_sample_rdd = l0_rdd.sample(True, sample_rate(1.0*max_count/l0_count))
    l1_sample_rdd = l1_rdd.sample(True, sample_rate(1.0*max_count/l0_count))
    
    new_train_rdd = l0_sample_rdd.union(l1_rdd).union(l0_rdd).union(l1_sample_rdd)
    new_train_df = spark.createDataFrame(new_train_rdd,['features','label'])
    
    return new_train_df



from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import RandomForestClassifier,GBTClassifier,GBTClassifierModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


jiaolvxing_df = spark.createDataFrame(data_train_label_df.select(label+feature_list).rdd.map(
        lambda line:[Vectors.dense(line[1:]), clean_label(line[0])]),['features','label'])


jiaolvxing_splits = jiaolvxing_df.randomSplit([0.9, 0.1])
jiaolvxing_rf = GBTClassifier(maxDepth = 6, maxBins=128)
jiaolvxing_model = jiaolvxing_rf.fit(over_sampling(jiaolvxing_splits[0]))

test_score_and_labels = jiaolvxing_model.transform(jiaolvxing_splits[1])
evaluation_dataset = spark.createDataFrame(test_score_and_labels,['label',"prediction"])
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction')

accuracy = evaluator.evaluate(evaluation_dataset, {evaluator.metricName:"accuracy"})
f1 = evaluator.evaluate(evaluation_dataset, {evaluator.metricName:"f1"})
precision = evaluator.evaluate(evaluation_dataset, {evaluator.metricName:"precision"})
recall = evaluator.evaluate(evaluation_dataset, {evaluator.metricName:"recall"})



    






























            