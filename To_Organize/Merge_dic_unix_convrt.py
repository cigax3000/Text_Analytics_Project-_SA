#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 17:44:33 2018

@author: MarioAntao
"""
import pickle
import pandas as pd

Digital_Music_sample_dict=pd.read_pickle('/Users//MarioAntao//Desktop//Text Analytics//Project//TA_Samples//List_Dict_Label//Digital_Music_sample_dict_lab.pickle')

baby_sample_dict=pd.read_pickle('//Users//MarioAntao/Desktop//Text Analytics//Project//TA_Samples//List_Dict_Label//baby_sample_dict_lab.pickle')

cd_and_vinyl_sample_dict=pd.read_pickle('//Users//MarioAntao//Desktop//Text Analytics//Project//TA_Samples//List_Dict_Label//cd_and_vinyl_sample_dict_lab.pickle')

Toys_games_dict=pd.read_pickle('//Users//MarioAntao//Desktop//Text Analytics//Project//TA_Samples//List_Dict_Label//Toys_games_dict_lab.pickle')



def unix(dictionairy):
    from datetime  import datetime
    import datetime
    for item in dictionairy:
            item['unixReviewTime']=datetime.datetime.fromtimestamp(
            item ['unixReviewTime']
            ).strftime('%Y-%m-%d %H:%M:%S')        



def split_overall(dic,te_size):
    """

    :param dic: variable type dict to be split by overall 20000 in 20000 and merged defining test size
    :return: variable with overall 1,2,3,4,5 separated per variable
    """

    from sklearn.cross_validation import train_test_split

    me_va_tr = []
    me_va_te = []
    va1 = dic [0:20000]
    va2 = dic [20000:40000]
    va3 = dic [40000:60000]
    va4 = dic [60000:80000]
    va5 = dic [80000:100000]

    va1_tr, va1_te = train_test_split(va1,test_size=te_size)
    va2_tr, va2_te = train_test_split(va2, test_size=te_size)
    va3_tr, va3_te = train_test_split(va3, test_size=te_size)
    va4_tr, va4_te = train_test_split(va4, test_size=te_size)
    va5_tr, va5_te = train_test_split(va5, test_size=te_size)

    me_va_tr = va1_tr + va2_tr + va3_tr + va4_tr + va5_tr
    me_va_te = va1_te + va2_te + va3_te + va4_te + va5_te


    return me_va_tr, me_va_te




review_dig_train, review_did_test = split_overall(Digital_Music_sample_dict,0.3)
review_baby_train,review_baby_test = split_overall(baby_sample_dict,0.3)
review_cd_train, review_cd_test = split_overall(cd_and_vinyl_sample_dict,0.3)
review_toys_train,review_toys_test = split_overall(Toys_games_dict,0.3)





#Merge Train data
merged_train_dic = review_dig_train + review_baby_train + review_cd_train + review_toys_train

#Merge Test Data
merged_test_dic = review_did_test + review_baby_test + review_cd_test + review_toys_test

#Convert unix to date
unix(merged_test_dic)
unix(merged_train_dic)




"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//merged_train_sa_dic.pickle', 'wb') as handle:
    pickle.dump(merged_train_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//merged_test_sa_dic.pickle', 'wb') as handle:
    pickle.dump(merged_test_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
   


merged_train_dic=pd.read_pickle('merged_train_dic.pickle')
merged_test_dic=pd.read_pickle('//Users//MarioAntao//Documents//Try_1//merged_test_dic.pickle')


import random

random.shuffle(merged_train_dic)
random.shuffle(merged_test_dic)


