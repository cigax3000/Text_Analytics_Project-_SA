#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:35:34 2018

@author: MarioAntao
"""


import matplotlib.pyplot as plt

def lsa_try (idfs,x,featurenames):
    """
    Input:(idfs,ncomponents,featurenames)
        
        
    Output: print of concepts
       
    """
    from sklearn.decomposition import TruncatedSVD
    lsa = TruncatedSVD(n_components=x, n_iter=100)
    lsa_features = lsa.fit_transform(idfs)
    terms = featurenames   
    for i, comp in enumerate(lsa.components_): 
        termsInComp = zip (terms,comp)
        sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
        print("Concept %d:" % i )
        for term in sortedTerms:
            print(term[0])
        print (" ")
  
    lsa_sv=lsa.singular_values_
    
    return lsa, lsa_features


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
