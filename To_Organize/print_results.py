#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 19:59:47 2018

@author: MarioAntao
"""

test_label = pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//Try_2_part1//corpus_label//test_label_v3.pickle')
SVM_bow_chi2 = pd.read_pickle('//Users/MarioAntao//Documents//ta_project_sa//data//Try_2_part1//svm//SVM_bow_chi2.pickle')
SVM_bow_f = pd.read_pickle('//Users/MarioAntao//Documents//ta_project_sa//data//Try_2_part1//svm//SVM_bow_f.pickle')
SVM_bow_lsa_5 = pd.read_pickle('//Users/MarioAntao//Documents//ta_project_sa//data//Try_2_part1//svm//SVM_bow_lsa_5')
SVM_bownadj_chi2 = pd.read_pickle('//Users/MarioAntao//Documents//ta_project_sa//data//Try_2_part1//svm//SVM_bownadj_chi2')
SVM_bownadj_f = pd.read_pickle('//Users/MarioAntao//Documents//ta_project_sa//data//Try_2_part1//svm//SVM_bownadj_f')
SVM_bownadj_lsa_5 = pd.read_pickle('//Users/MarioAntao//Documents//ta_project_sa//data//Try_2_part1//svm//SVM_bownadj_lsa_5')
SVM_bownadv_chi2 = pd.read_pickle('//Users/MarioAntao//Documents//ta_project_sa//data//Try_2_part1//svm//SVM_bownadv_chi2')
SVM_bownadv_f = pd.read_pickle('//Users/MarioAntao//Documents//ta_project_sa//data//Try_2_part1//svm//SVM_bownadv_f')
SVM_bownadv_lsa_5 = pd.read_pickle('//Users/MarioAntao//Documents//ta_project_sa//data//Try_2_part1//svm//SVM_bownadv_lsa_5')


print('BAG OF WORDS MODELS')
print('SVM with tf-idf F-statistic:')
print(get_metrics(test_label,SVM_bow_f))
print(' ')
print('SVM with tf-idf Chi2:')
print(get_metrics(test_label,SVM_bow_chi2))
print(' ')
print('SVM with tf-idf Lsa with 5 concepts:')
print(get_metrics(test_label,SVM_bow_lsa_5))
print(' ')
print(' ')
print('BOW Nouns and Adj MODELS')
print('SVM with tf-idf F-statistic:')
print(get_metrics(test_label,SVM_bownadj_f))
print(' ')
print('SVM with tf-idf Chi2:')
print(get_metrics(test_label,SVM_bownadj_chi2))
print(' ')
print('SVM with tf-idf Lsa with 5 concepts:')
print(get_metrics(test_label,SVM_bownadj_lsa_5))
print(' ')
print(' ')
print('BAG OF WORDS Adverbs, Nouns and Adj:')
print('SVM with tf-idf F-statistic:')
print(get_metrics(test_label,SVM_bownadv_f))
print(' ')
print('SVM with tf-idf Chi2:')
print(get_metrics(test_label,SVM_bownadv_chi2))
print(' ')
print('SVM with tf-idf Lsa with 5 concepts:')
print(get_metrics(test_label,SVM_bownadv_lsa_5))

bownadj_tfidf_chi2_tr = pd.read_pickle('//Users/MarioAntao//Documents//ta_project_sa//data//Try_2_part1//bown_adj//bownadj_tfidf_chi2_tr')
