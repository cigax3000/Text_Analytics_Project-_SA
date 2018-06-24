#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 17:38:42 2018

@author: MarioAntao
"""
"""
from sklearn.feature_selection import SelectPercentile, f_classif

selector = SelectPercentile(f_classif,percentile = 10)
selector.fit (bow_tr_idfs, tr_label_sample)
features_train_transformed = selector.transform(bow_tr_idfs).toarray()
features_test_transformed = selector.transform(bow_te_idfs).toarray()
"""
import sys
print (sys.version)
import matplotlib.pyplot as plt

def lsa_try (idfs,x,featurenames):
    """
    Input:(idfs,ncomponents,featurenames)
        
        
    Output: print of concepts
       
    """
    from sklearn.decomposition import TruncatedSVD
    lsa = TruncatedSVD(n_components=x, n_iter=100)
    lsa.fit(idfs)
    terms = bow_tr_vectorizer.get_feature_names()    
    for i, comp in enumerate(lsa.components_): 
        termsInComp = zip (terms,comp)
        sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
        print("Concept %d:" % i )
        for term in sortedTerms:
            print(term[0])
        print (" ")
  
    lsa_sv=lsa.singular_values_
    
    return lsa_sv


def elbow_lsa(lsa_singularvalue,ncomponents_lsa):
    """
    Input:lsa_singular_falues, n_components_lsa
    Output:Elbow Graph
        
    """
    import matplotlib.pyplot as plt
    aa=list(range(0, ncomponents_lsa))
    
    # Plot the elbow
    plt.plot(aa,lsa_singularvalue)
    plt.xlabel('Nº of Components')
    plt.ylabel('Singular Values')
    plt.title('The Elbow Method showing the optimal nº of components')
    plt.show()

def fit_lsa(idfs):
    """
    Input:idfs
    Output: LSA fit transform of idfs
    """
    
    lsa.fit_transform(idfs)

 
lsa_try(bow_tr_idfs,30,bow_tr_features_name)
 

lsa_sv=lsa.singular_values_
elbow_lsa(lsa_sv,30)
    
    
    
    
    

 """  
from sklearn.decomposition import TruncatedSVD
bow_tr_idfs.shape

lsa = TruncatedSVD(n_components=30, n_iter=100)

lsa.fit(bow_tr_idfs)

lsa.components_[0]

terms = bow_tr_vectorizer.get_feature_names()
for i, comp in enumerate(lsa.components_): 
    termsInComp = zip (terms,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
    print("Concept %d:" % i )
    for term in sortedTerms:
        print(term[0])
    print (" ")
    
lsa.singular_values_
lsa.fit_transform(bow_tr_idfs)

aa=list(range(0, 27))
bb=list(range(4, 250))
# Plot the elbow
plt.plot(aa,ss)
plt.xlabel('Nº of Components')
plt.ylabel('Singular Values')
plt.title('The Elbow Method showing the optimal nº of components')
plt.show()

"""
