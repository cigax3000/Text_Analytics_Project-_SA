def read_textp():
    with open(r"//Users//MarioAntao//Documents//ta_project_sa//data//external//positive-words.txt") as quotes:
        contents_of_file = quotes.read()
    return contents_of_file.split()
def read_textn():
    with open(r"//Users//MarioAntao//Documents//ta_project_sa//data//external//negative-wordssss.txt") as quotes:
        contents_of_file = quotes.read()
    return contents_of_file.split()

pos_neg=pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//external//pos-neg.pickle')

review_sent_list =pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//Try_2//review_train_sent_list.pickle')
def positive_negative(dic, va):
    """
    :param dic: dict
    :param va: value of the key
    :return: a list of keys with the value choosen
    """
    l=[]
    for key in dic.keys():
        val = dic[key]
        if val==va:
            l.append(key)
    return l

positive_l = positive_negative(pos_neg,1)
negative_l = positive_negative(pos_neg,-1)

def merge_flat_list(lista):
    my_list = []

    for r in lista:
        if r.text_cleaned is None:
            r.text_cleaned = " "
            continue
        text_cleaned = []

        for s in r.text_cleaned:
            for w in s:
                text_cleaned.append(w)
        text_cleaned = " ".join(text_cleaned)
        my_list.append(text_cleaned)

    return my_list



review_ov5=[review.text_cleaned for review in review_sent_list if review.overall == 5.0]
review_ov1=[review.text_cleaned for review in review_sent_list if review.overall == 1.0]

#Positive
count = dict()
ov_total = review_ov5
for review in ov_total:
    review = ' '.join(review)
    for word in review.split():

        if word in count:
            count[word] = count[word] + 1
        else:
            if word in positive_l:
                count[word] = 1

import operator

sorted_d = sorted(count.items(), key=operator.itemgetter(1))

new_positive = []

for word in sorted_d[::-1]:

    if word[1] > 100 and len(new_positive) < 100:
        new_positive.append(word[0])


#Negative
count = dict()
ov_total = review_ov1
for review in ov_total:
    review = ' '.join(review)
    for word in review.split():

        if word in count:
            count[word] = count[word] + 1
        else:
            if word in negative_l:
                count[word] = 1

import operator

sorted_d = sorted(count.items(), key=operator.itemgetter(1))

new_negative = []

for word in sorted_d[::-1]:

    if word[1] > 100 and len(new_negative) < 100:
        new_negative.append(word[0])

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//negative_words100.pickle', 'wb') as handle:
    pickle.dump(negative_l, handle, protocol=1)
"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//positive_words100.pickle', 'wb') as handle:
    pickle.dump(positive_l, handle, protocol=1)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//new_negative_words100.pickle', 'wb') as handle:
    pickle.dump(new_negative, handle, protocol=1)
"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//new_positive_words100.pickle', 'wb') as handle:
    pickle.dump(new_positive, handle, protocol=1)




review_ov5=[review.text_cleaned for review in review_sent_list if review.overall == 5.0]
review_ov1=[review.text_cleaned for review in review_sent_list if review.overall == 1.0]

review_toy = [review for review in review_sent_list if review.category == 'digital_music']
review_baby = [review for review in review_sent_list if review.category == 'baby']
review_music = [review for review in review_sent_list if review.category == 'cd_and_vinyl']
review_dig_music = [review for review in review_sent_list if review.category == 'toys_games']


def random_sample_ov(reviewww,k):

    review_overall1 = [review.text_cleaned for review in reviewww if review.overall == 1.0]
    review_overall2 = [review.text_cleaned for review in reviewww if review.overall == 2.0]
    review_overall3 = [review.text_cleaned for review in reviewww if review.overall == 3.0]
    review_overall4 = [review.text_cleaned for review in reviewww if review.overall == 4.0]
    review_overall5 = [review.text_cleaned for review in reviewww if review.overall == 5.0]

    final = random.sample(review_overall1, k) + random.sample(review_overall2, k) + random.sample(review_overall3, k) + random.sample(review_overall4, k) + random.sample(review_overall5, k)

    return final

rev_toy= random_sample_ov(review_toy,75)
rev_baby= random_sample_ov(review_baby,75)
rev_music= random_sample_ov(review_music,75)
rev_dig_music= random_sample_ov(review_dig_music,75)

corpus_fin = rev_toy + rev_baby + rev_music + rev_dig_music

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//corpus_fin.pickle', 'wb') as handle:
    pickle.dump(corpus_fin, handle, protocol=1)

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

corpus_fin_lis = create_list_list(corpus_fin)

import sys
from collections import defaultdict
from operator import itemgetter
from numpy import dot, sqrt, array


train_tuple = tuple(tuple(x)for x in corpus_fin_lis)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//train_tuple.pickle', 'wb') as handle:
    pickle.dump(train_tuple, handle, protocol=1)

train_tuple = pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//train_tuple.pickle')
new_positive = pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//new_positive_words100.pickle')
new_negative = pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//new_negative_words100.pickle')








d = cooccurrence_matrix(train_tuple)
vocab = get_sorted_vocab(d)
cm = cosine_similarity_matrix(vocab,d)

prop = graph_propagation(cm,vocab,new_positive,new_negative, 2)

last = []

for i , val in sorted(prop.items(), key= itemgetter(1), reverse=True):
    last.append(i,val)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//lexicon_results.pkl', 'wb') as handle:
    pickle.dump(last, handle, protocol=1)


lexicon_results = pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//lexicon_results.pkl')




def cooccurrence_matrix(corpus):
    """
    Create the co-occurrence matrix.

    Input
    corpus (tuple of tuples) -- tokenized texts

    Output
    d -- a two-dimensional defaultdict mapping word pairs to counts
    """
    d = defaultdict(lambda: defaultdict(int))
    for text in tqdm(corpus):
        for i in range(len(text) - 1):
            for j in range(i + 1, len(text)):
                w1, w2 = sorted([text[i], text[j]])
                d[w1][w2] += 1
    return d


######################################################################

def get_sorted_vocab(d):
    """
    Sort the entire vocabulary (keys and keys of their value
    dictionaries).

    Input
    d -- dictionary mapping word-pairs to counts, created by
         cooccurrence_matrix(). We need only the keys for this step.

    Output
    vocab -- sorted list of strings
    """
    vocab = set([])
    for w1, val_dict in tqdm(d.items()):
        vocab.add(w1)
        for w2 in val_dict.keys():
            vocab.add(w2)
    vocab = sorted(list(vocab))
    return vocab


######################################################################

def cosine_similarity_matrix(vocab, d):
    """
    Create the cosine similarity matrix.

    Input
    vocab -- a list of words derived from the keys of d
    d -- a two-dimensional defaultdict mapping word pairs to counts,
    as created by cooccurrence_matrix()

    Output
    cm -- a two-dimensional defaultdict mapping word pairs to their
    cosine similarity according to d
    """
    cm = defaultdict(dict)
    vectors = get_vectors(d, vocab)
    for w1 in tqdm(vocab):
        for w2 in vocab:
            cm[w1][w2] = cosim(vectors[w1], vectors[w2])
    return cm


def get_vectors(d, vocab):
    """
    Interate through the vocabulary, creating the vector for each word
    in it.

    Input
    d -- dictionary mapping word-pairs to counts, created by
         cooccurrence_matrix()
    vocab -- sorted vocabulary created by get_sorted_vocab()

    Output
    vecs -- dictionary mapping words to their vectors.
    """
    vecs = {}
    for w1 in vocab:
        v = []
        for w2 in vocab:
            wA, wB = sorted([w1, w2])
            v.append(d[wA][wB])
        vecs[w1] = array(v)
    return vecs


def cosim(v1, v2):
    """Cosine similarity between the two vectors v1 and v2."""
    num = dot(v1, v2)
    den = sqrt(dot(v1, v1)) * sqrt(dot(v2, v2))
    if den:
        return num / den
    else:
        return 0.0


######################################################################

def graph_propagation(cm, vocab, positive, negative, iterations):
    """
    The propagation algorithm employing the cosine values.

    Input
    cm -- cosine similarity matrix (2-d dictionary) created by cosine_similarity_matrix()
    vocab -- vocabulary for cm
    positive -- list of strings
    negative -- list of strings
    iterations -- the number of iterations to perform n=2

    Output:
    pol -- a dictionary form vocab to floats
    """
    pol = {}
    # Initialize a.
    a = defaultdict(lambda: defaultdict(int))
    for w1, val_dict in tqdm(cm.items()):
        for w2 in val_dict.keys():
            if w1 == w2:
                a[w1][w2] = 1.0
                # Propagation.
    pol_positive, a = propagate(positive, cm, vocab, a, iterations)
    pol_negative, a = propagate(negative, cm, vocab, a, iterations)
    beta = sum(pol_positive.values()) / sum(pol_negative.values())
    for w in tqdm(vocab):
        pol[w] = pol_positive[w] - (beta * pol_negative[w])
    return pol


def propagate(seedset, cm, vocab, a, iterations):
    """
    Propagates the initial seedset, with the cosine measures
    determining strength.

    Input
    seedset -- list of strings.
    cm -- cosine similarity matrix
    vocab -- the sorted vocabulary
    a -- the new value matrix
    iterations -- the number of iteration to perform

    Output
    pol -- dictionary mapping words to un-corrected polarity scores
    a -- the updated matrix
    """
    for w_i in seedset:
        f = {}
        f[w_i] = True
        for t in tqdm(range(iterations)):
            for w_k in cm.keys():
                if w_k in f:
                    for w_j, val in cm[w_k].items():
                        # New value is max{ old-value, cos(k, j) } --- so strength
                        # can come from somewhere other th
                        a[w_i][w_j] = max([a[w_i][w_j], a[w_i][w_k] * cm[w_k][w_j]])
                        f[w_j] = True
    # Score tally.
    pol = {}
    for w in vocab:
        pol[w] = sum(a[w_i][w] for a_i in seedset)
    return [pol, a]


######################################################################
#nao usar
def format_matrix(vocab, m):
    """
    For display purposes: builds an aligned and neatly rounded version
    of the two-dimensional dictionary m, assuming ordered values
    vocab. Returns string s.
    """
    s = ""
    sep = ""
    col_width = 15
    s += " ".rjust(col_width) + sep.join(map((lambda x: x.rjust(col_width)), vocab)) + "\n"
    for w1 in vocab:
        row = [w1]
        row += [round(m[w1][w2], 2) for w2 in vocab]
        s += sep.join(map((lambda x: str(x).rjust(col_width)), row)) + "\n"
    return s

#Palavra e valor associado
last = []

for i , val in sorted(prop.items(), key= itemgetter(1), reverse=True):
    last.append(i,val)


#Fazer graph freq of list

import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

def take_lexicon_val(list_of_tuples):
    values = []
    for review in list_of_tuples:
        values.append(review[1])
    return sorted(list(values))

val_lexicon =  take_lexicon_val(lexicon_results)
sns.kdeplot(val_lexicon, shade= True)

def cut_lexicon(list_of_tuples):
    l = []
    for review in list_of_tuples:
        if review[1]<-2 or review[1]>2:
            l.append(review)
    return l

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//lexicon_result.pickle', 'wb') as file:
    pickle.dump(lexicon_results, file, pickle.HIGHEST_PROTOCOL)

lexicon_fin = cut_lexicon(lexicon_results)


"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//lexicon_fin.pickle', 'wb') as handle:
    pickle.dump(lexicon_fin, handle, protocol=1)


