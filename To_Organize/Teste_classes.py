#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 01:40:02 2018

@author: MarioAntao
"""
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

class reviews(object):
    """Class represents tweet from Twitter.

    Attributes:
        reviewer_id: A unique identifier.
        product_id : A unique identifier.
        ReviewTime: A date-time, describing when the review was created.
        reviewText: A string, containing the text of the review.
        overall: A float, stating the rating of the product.
        summary: A string, containing the summary of the review.
        helpful: An integer, stating the number of favorites.
        category: A list, containing the helpfulness rating of the review.
    """

    def __init__(self, reviewer_id, product_id, ReviewTime, reviewText, overall, summary, helpful, category, text_cleaned):
        """Initialzes Review class with defined content."""
        self.reviewer_id = reviewer_id
        self.product_id = product_id
        self.ReviewTime = ReviewTime
        self.reviewText = reviewText
        self.overall = overall
        self.summary = summary
        self.helpful = helpful
        self.category = category
        self.cleaning_log = dict()
        self.text_cleaned = None


def create_review_from_dict(review_dict):
    """
    Creates a review object from dictionary.
    
    Extracts reviewer_id, helpful, reviewText, overall,
    summary,unixReviewTime, product_id and category from dictionary.
    
    Args:
        review_dict: A dictionary, containing review information.
        
    Returns:
        A review object.
    """
    
    # Extract parameters from dictionary
    reviewer_id = review_dict.get('reviewer_id')
    helpful = review_dict.get('helpful')
    reviewText = review_dict.get('reviewText')
    overall = review_dict.get('overall')
    summary = review_dict.get('summary')
    ReviewTime = review_dict.get('unixReviewTime')
    category = review_dict.get('category')
    product_id = review_dict.get('product_id')
    text_cleaned = review_dict.get('text_cleaned')
    
    # Create review object
    review = reviews(reviewer_id, product_id, ReviewTime, reviewText, overall, summary, helpful, category,text_cleaned)
    
    return review

def make_review_set(filename):
    """
    Creates set of tweets from week_3.data/01_raw/trump_tweets.json.
    Returns:
        Set of week_3.tweet objects.
    """
    
    # Create set and review.Review objects.
    review_set = set()
      
    for revieww in review_train:
        review = create_review_from_dict(revieww)
        review_set.add(review)

    return review_set

def make_review_list(filename):
    """
    Creates set of tweets from week_3.data/01_raw/trump_tweets.json.
    Returns:
        Set of week_3.tweet objects.
    """
    
    # Create set and review.Review objects.
    review_list = []
      
    for revieww in merged_test_dic:
        review = create_review_from_dict(revieww)
        review_list.append(review)

    return review_list
"""

test = [review.text_cleaned for review in a]

def create_list_list(list_of_text):
    test20 = []
    for test in list_of_text:

        if test is None:
            test20.append([])
            print(test20)
        else:
            test=" ".join(test)
            test20.append(test.split())

    return test20
asss= create_list_list(test)

train_tuple = tuple(tuple(x)for x in asss)

"""


merged_dic=pd.read_pickle('merged_dic.pickle')


merged_train_dic=pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//raw//merged_train_sa_dic.pickle')
merged_test_dic=pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//raw//merged_test_sa_dic.pickle')


d_dic=pd.read_pickle('review_test_list.pickle')

train_label = pd.read_pickle('train_label.pickle')

a=random.sample(review_train_list,10000)


#List

review_sample_list=make_review_list(sample_review)

review_train_list=make_review_list(merged_train_dic)
review_test_list=make_review_list(merged_test_dic)




review_train_list_clean1= pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//processed//review_train_lis_afterclean1.pickle')
review_test_list_clean1 = pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//processed//review_test_lis_afterclean1.pickle')

review_train_list= pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//processed//review_train_lis.pickle')
review_test_list = pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//processed//review_test_lis.pickle')



review_train_sent_list = review_train_list
review_test_sent_list = review_test_list

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//review_test_lis.pickle', 'wb') as handle:
    pickle.dump(review_test_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//review_train_lis.pickle', 'wb') as handle:
    pickle.dump(review_train_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
#After Clean code not file
"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//review_test_lis_afterclean1.pickle', 'wb') as handle:
    pickle.dump(review_test_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//review_train_lis_afterclean1.pickle', 'wb') as handle:
    pickle.dump(review_train_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


"""Import pickle"""
review_test_set=pd.read_pickle('review_test_set.pickle')
review_train_set=pd.read_pickle('review_train_set.pickle')
review_sample_set=pd.read_pickle('review_sample_set.pickle')


