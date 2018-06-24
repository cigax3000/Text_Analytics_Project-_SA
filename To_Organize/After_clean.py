#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:31:30 2018

@author: MarioAntao
"""
import itertools


#Flat List

def merge_flat_list (lista):
    my_list = []
    
    for r in lista:
        if r.text_cleaned is None:
            r.text_cleaned=" "
            continue
        text_cleaned = []
        
        for s in r.text_cleaned:
            for w in s:
                text_cleaned.append(w)
        text_cleaned = " ".join(text_cleaned)
        my_list.append(text_cleaned)
    
    return my_list

def merge_flat_list_label (lista):
    my_list = []
    
    for r in lista:
        if r.text_cleaned is None:
            r.text_cleaned=" "
            continue
        label_cleaned = []
        
        for s in r.overall:
            for w in str(s):
                label_cleaned.append(w)
        label_cleaned = "".join(label_cleaned)
        my_list.append(label_cleaned)
    
    return my_list


def merge_flat_list_label_ov(lista):
    ov_only =  [r.overall for r in lista]
    return ov_only



review_train_list=pd.read_pickle('review_train_list_clean.pickle')

#Create Corpus
train_corpus=merge_flat_list(review_train_list)
test_corpus=merge_flat_list(review_test_list)

#Create Labels
train_label=merge_flat_list_label_ov(review_train_list)
test_label=merge_flat_list_label_ov(review_test_list)
#Export label and corpus pickle



"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//train_corpus_v3.pickle', 'wb') as handle:
    pickle.dump(train_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//test_corpus_v3.pickle', 'wb') as handle:
    pickle.dump(test_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//train_label_v3.pickle', 'wb') as handle:
    pickle.dump(train_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//test_label_v3.pickle', 'wb') as handle:
    pickle.dump(test_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#Open Pickle
    train_corpus=pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//train_corpus_V2.pickle')
    test_corpus=pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//test_corpus_V2.pickle')
    train_label=pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//train_label_V2.pickle')
    test_label=pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//test_label_V2.pickle')
    
#Create Sample
    tr_corpus_sample=train_corpus[35000:105000]
    te_corpus_sample=test_corpus[0:30000]
    tr_label_sample=train_label[0:70000]
    te_label_sample=test_label[0:30000]


a=random.sample(train_corpus,10000)

a=review_train_str[0:5000]
b=review_train_label[0:5000]


review_sample_set=pd.read_pickle('review_sample_set.pickle')

review_train_str=pd.read_pickle('review_train_str.pickle')
#Save Picke_list_clean
"""Export Data with pickle"""
with open('review_train_list_clean_shuffle.pickle', 'wb') as handle:
    pickle.dump(review_train_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('review_test_list_clean_shuffle.pickle', 'wb') as handle:
    pickle.dump(review_test_list, handle, protocol=pickle.HIGHEST_PROTOCOL)





# Save Pickle_corpus
    
    
    
"""Export Data with pickle"""
with open('review_train_str.pickle', 'wb') as handle:
    pickle.dump(review_train_str, handle, protocol=pickle.HIGHEST_PROTOCOL)


"""Export Data with pickle"""
with open('review_sample_str.pickle', 'wb') as handle:
    pickle.dump(review_sample, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Count the most frequent words
    
from collections import Counter
import re

counts = Counter()
words = re.compile(r'\w+')

for sentence in train_corpus:
    counts.update(words.findall(sentence.lower()))

top50=counts.most_common(50)

review_train_list=pd.read_pickle('train_corpus.pickle')