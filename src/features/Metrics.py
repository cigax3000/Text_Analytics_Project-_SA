#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 11:45:13 2018

@author: MarioAntao
"""

from sklearn import metrics
import numpy as np
from sklearn.svm import SVC


def get_metrics(true_labels, predicted_labels):
    
    print ('Accuracy:', np.round(
                        metrics.accuracy_score(true_labels, 
                                               predicted_labels),
                        2))
    print ('Precision:', np.round(
                        metrics.precision_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        2))
    print ('Recall:', np.round(
                        metrics.recall_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        2))
    print ('F1 Score:', np.round(
                        metrics.f1_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        2))
                        
                        
                        
                        weee
def train_predict_evaluate_model(train_features, train_labels, 
                                 test_features, test_labels):
    
    # build model    
    model = SVC(kernel='linear',C=1.0).fit(train_features, train_labels)
    # predict using model
    predictions = model.predict(test_features) 
    # evaluate model prediction performance   
    get_metrics(true_labels=test_labels, 
                predicted_labels=predictions)
    return predictions    




