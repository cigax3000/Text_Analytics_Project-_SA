#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:08:01 2018

@author: MarioAntao
"""

from statistics import mean
from collections import Counter
from re import split

def calculate_average_reviews_per_day(review_set):
    """
    Calculated the average number of tweets per day.
    Args:
        tweet_set: Set of week_3.tweet objects.

    Returns:
        Float value, corresponding to the average number of tweets.
    """
    # Calculate difference in days between maximum date and minimum date.
    min_date = min([review.ReviewTime for review in reviews])
    max_date = max([review.ReviewTime for review in reviews])
    delta_date = (max_date - min_date).days
    # Return average number of tweets per day.
    return len(reviews) / delta_date

def calculate_average_word_count(review_set):
    """"
    Calculates average number of words in set of tweets.

    Args:
        tweet_set: A set of tweets.
    Returns:
        The average number of words per week_3.tweet (float).
    """
    return mean([review.count_number_of_words() for review in review_set])


