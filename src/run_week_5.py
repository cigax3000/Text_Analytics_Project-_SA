#from  evaluation import metrics
from  src.features.BOW_Bigrams.BOW_Tokenization import word_tokenize, ngrams_tokenize, vocabulary_size
from  src.features.process_text.feature_weighting import compute_tfidf, compute_tfidf_stopwords, compute_tfidfle_stopwords
from  src.features.LSA_1 import lsa_try,elbow_lsa,fit_lsa
from  src.features.Metrics import get_metrics,train_predict_evaluate_model
from  src.models.supervised_classification import SVC_model,look_at_predictions, prediction
#from  src.utils.look_tfidfs import look_at_features


['problem', 'help', 'function', 'one']
def main():
    """
    The objective of this class is to investigate different techniques to:
        1. Use sklearn to explore data
        2. Lemmatization with sklearn
        3. TF-IDF
        4. Supervised learning
        5. Evaluation metrics
    """


    """
    SECTION  2:
        Look at an example were we do BOW and n-grams(=2)
            2.1 Use the function from features.process_text.lemmatization to compute lemmas from the dataset
                a) Look at data (similar to point 2)
            2.2. Compare the vocabulary size with and without using lemmas.
                a) Is it the same as before we applied the lemmas? Yes/No what happen?

    """
"""
def display_features(features, feature_names):
   df = pd.DataFrame(data=features,
                      columns=feature_names)
    return df
"""    
    
    """
    BOW
    """
       # Tokenization_train (BOW)
        bow_tr_features, bow_tr_vectorizer = word_tokenize(train_corpus)  # data.data = data_lst.tolist()
     
        
        #bow_transformer, bow_features = compute_tfidf(bow_index)
      
      #Tokenization_test (BOW)
        #bow_te_features, bow_te_vectorizer = word_tokenize(te_corpus_sample)  # data.data = data_lst.tolist()
        bow_te_features = bow_tr_vectorizer.transform(test_corpus) 

              
        #Vocabulary size
     # print('Vocabulary size (bow): '+str(vocabulary_size(bow_vectorizer)))
     
        #Look at some of the vocabulary
      #print(bow_vectorizer.get_feature_names()[90:1000])
     
 #Get features name
   
   bow_tr_features_name = bow_tr_vectorizer.get_feature_names()
   
   #bow_te_features_name = bow_te_vectorizer.get_feature_names()
   #display_features(bow_features,bow_features_names)

import nltk
from tqdm import tqdm

"""
BOW NOUNS
"""

# BOW only Nouns NOT USE

def extract_nouns(list_of_strings):
    l = []
    nouns = []
    for word in tqdm(list_of_strings):
        word_tagged = nltk.pos_tag(word.split())
        l.append([w for (w, t) in word_tagged if t.startswith("N")])

    nouns = [' '.join(x) for x in l]
    return nouns


"""
BOW NOUNS ADJ and ADV
"""


def extract_n_adv_adj(list_of_strings):
    l = []
    nouns = []
    for word in tqdm(list_of_strings):
        word_tagged = nltk.pos_tag(word.split())
        l.append([w for (w, t) in word_tagged if t.startswith("N") or t.startswith("J") or t.startswith("R")])

    nouns = [' '.join(x) for x in l]
    return nouns


bow_n_adv_tr = extract_n_adv_adj(train_corpus)
bow_n_adv_te = extract_n_adv_adj(test_corpus)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//bow_n_tr.pickle', 'wb') as handle:
    pickle.dump(bow_n_adv_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//bow_n_te.pickle', 'wb') as handle:
    pickle.dump(bow_n_adv_te, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Train
    bownadv_tr_features, bownadv_tr_vectorizer = word_tokenize(bow_n_adv_tr)  # data.data = data_lst.tolist()
    bownadv_tr_features_name = bownadv_tr_vectorizer.get_feature_names()

    # Test
    bownadv_te_features = bownadv_tr_vectorizer.transform(bow_n_adv_te)

    #bown_te_features_name = bown_te_vectorizer.get_feature_names()

"""
BOW ADJ_NOUNS
"""
a = random.sample(train_corpus, 1000)


def adj_noun(list_of_strings):
    fl = []
    for sent in tqdm(list_of_strings):
        tok = sent.split()
        l = []
        size = len(tok)
        for w in range(0, size):
            wrd = tok[w]
            pos_tag = nltk.pos_tag(wrd)
            if pos_tag.startswith('JJ'):
                l.append(wrd)
                if w + 1 < size and nltk.pos_tag(tok[w + 1]).startswith('NN'):
                    l.append(wrd + '_' + tok[w + 1])
                if w + 2 < size and nltk.pos_tag(tok[w + 2]).startswith('NN'):
                    l.append(wrd + '_' + tok[w + 2])
        fl.append(''.join(l))

    return fl


def adj_noun(list_of_strings):
    fl = []
    for sent in tqdm(list_of_strings):
        tok = sent.split()
        word_tagged = nltk.pos_tag(tok)
        l = []
        size = len(word_tagged)
        for w in range(0, size):
            wrd = word_tagged[w]

            if wrd[1][1] == 'J':

                l.append(wrd[0])
                if w + 1 < size and word_tagged[w + 1][1][1].startswith('N'):
                    l.append(wrd[0] + '_' + word_tagged[w + 1][0])
                if w + 2 < size and word_tagged[w + 2][1][1].startswith('N'):
                    l.append(wrd[0] + '_' + word_tagged[w + 2][0])
        fl.append(''.join(l))

    return fl


def adj_noun_fin(my_list):
    ab = []
    for word in tqdm(my_list):
        previous_adjective = False
        my_important_terms = []
        my_important_term = []
        for (w, pos_tag) in word:
            if not previous_adjective:
                if my_important_term:
                    my_important_terms.append(my_important_term)
                my_important_term = []
            if pos_tag.startswith('NN'):
                my_important_term.append(w)
                previous_adjective = False
            if pos_tag.startswith('JJ'):
                my_important_term.append(w)
                previous_adjective = True
            else:
                previous_adjective = False

        if my_important_term:
            my_important_terms.append(my_important_term)
        my_important_terms = ["_".join([w for w in important_term]) \
                              for important_term in my_important_terms]

        my_important_terms = " ".join([w for w in my_important_terms])

        ab.append(my_important_terms)
    return ab


def tag_wrd(list_of_strings):
    l = []
    for word in tqdm(list_of_strings):
        word_tagged = nltk.pos_tag(word.split())
        l.append([w for w in word_tagged])

    return l


# aaa= adj_noun_fin(sas)
train_tag = tag_wrd(train_corpus)
test_tag = tag_wrd(test_corpus)

bow_n_adj_tr = adj_noun_fin(train_tag)
bow_n_adj_te = adj_noun_fin(test_tag)




"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//bow_n_adj_tr.pickle', 'wb') as handle:
    pickle.dump(bow_n_adj_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//bow_n_adj_te.pickle', 'wb') as handle:
    pickle.dump(bow_n_adj_te, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Train
    bown_adj_tr_features, bown_adj_tr_vectorizer = word_tokenize(bow_n_adj_tr)  # data.data = data_lst.tolist()
    bown_adj_tr_features_name = bown_adj_tr_vectorizer.get_feature_names()

    # Test
    bown_adj_te_features = bown_adj_tr_vectorizer.transform(bow_n_adj_te)
    #bown_adj_te_features_name = bown_adj_te_vectorizer.get_feature_names()




    """
    SECTION  3:
        TF-IDF
            3.1 Look at the functions: tfidf, tfidf_lemma_stopwords ( src.features.process_text.feature_weighting)
            3.2 Run the bellow code. Notice that if we use stopwords or lemmatization TFIDF weights change. Why is that?
    WARNING: THIS SECTION CAN TAKE A LONG TIME TO COMPUTE!
    """
    
    
    
#Tdif BOW
    
#Train
    #bow_transformer, bow_features = compute_tfidf(bow_index)
    
      bow_tr_tfidf, bow_tr_idfs = compute_tfidf(bow_tr_features)
     
     # bow_tfidfle_sw, bow_idfsle_sw = compute_tfidfle_stopwords(bow_tr_features, stopwords_lang='english')
      
      #bow_tfidf_sw, bow_idfs_sw = compute_tfidf_stopwords(bow_tr_features,stopwords_lang='english)

#Test
      #bow_transformer, bow_features = compute_tfidf(bow_te_features)
    
      #bow_te_tfidf, bow_te_idfs = compute_tfidf(bow_te_features)
      
      bow_te_idfs = bow_tr_tfidf.transform(bow_te_features)

 
#Tdif BOW NOUNS ADJ and ADV

#Train

      bownadv_tr_tfidf, bownadv_tr_idfs = compute_tfidf(bownadv_tr_features)
     
      #big_tr_tfidfle_sw, big_tr_idfsle_sw = compute_tfidfle_stopwords(bigram_tr_features, stopwords_lang='english')
#Test
      
      bownadv_te_idfs = bownadv_tr_tfidf.transform(bownadv_te_features)
      
      #big_te_tfidf, big_te_idfs = compute_tfidf(bigram_te_features)
     
      #big_te_idfsle_sw =  big_tr_tfidfle_sw.transform(bigram_te_features)
      


#Tdif BOW ADJ_NOUNS

#Train
      bownadj_tr_tfidf, bownadj_tr_idfs = compute_tfidf(bown_adj_tr_features)
      
     #print(big_a_idfs)
     
      #bown_tr_tfidfle_sw, bown_tr_idfsle_sw = compute_tfidfle_stopwords(bown_tr_features, stopwords_lang='english')
#Test

      bownadj_te_idfs = bownadj_tr_tfidf.transform(bown_adj_te_features)
      
      
      #bown_te_idfsle_sw = bown_tr_tfidfle_sw.transform(bown_te_features)
     
    #bown_te_tfidf, bown_te_idfs = compute_tfidf(bown_te_features)
    # bown_te_tfidfle_sw, bown_te_idfsle_sw = compute_tfidfle_stopwords(bown_te_features, stopwords_lang='english')





#import numpy as np
#features=np.round(bown_tfidf,2)
#display_features(features, bown_features_names)     


"""

Perform LSA

"""


#BOW Td_Idf
#Train
lsa_model_bow_tr, lsa_features_bow_tr = lsa_try(bow_tr_idfs,5,bow_tr_features_name)

#Graphic
lsa_sv_bow=lsa_model_bow_tr.singular_values_
elbow_lsa(lsa_sv_bow,30)


"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//lsa_model_bow_tr', 'wb') as handle:
    pickle.dump(lsa_model_bow_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data/lsa_features_bow_tr', 'wb') as handle:
    pickle.dump(lsa_features_bow_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#Cut-off   
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
 v
#F_classif
selector_bow_f = SelectKBest(f_classif, k=5)

bow_tfidf_f_tr = selector_bow_f.fit_transform(bow_tr_idfs,train_label)
bow_tfidf_f_te = selector_bow_f.transform(bow_te_idfs)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//bow_tfidf_f_tr', 'wb') as handle:
    pickle.dump(bow_tfidf_f_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//bow_tfidf_f_te', 'wb') as handle:
    pickle.dump(bow_tfidf_f_te, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
    
#Chi2
selector_bow_chi2 = SelectKBest(chi2, k=5)

bow_tfidf_chi2_tr = selector_bow_chi2.fit_transform(bow_tr_idfs,train_label)
bow_tfidf_chi2_te = selector_bow_chi2.transform(bow_te_idfs)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//bow_tfidf_chi2_tr', 'wb') as handle:
    pickle.dump(bow_tfidf_chi2_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//bow_tfidf_chi2_te', 'wb') as handle:
    pickle.dump(bow_tfidf_chi2_te, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Caso queiramos usar LSA
#Teste

lsa_features_bow_te = lsa_model_bow_tr.transform(bow_te_idfs)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//lsa_features_bow_te', 'wb') as handle:
    pickle.dump(lsa_features_bow_te, handle, protocol=pickle.HIGHEST_PROTOCOL)



#BOW Noun ADJ ADV

lsa_model_bownadv_tr, lsa_features_bownadv_tr = lsa_try(bownadv_tr_idfs,5,bownadv_tr_features_name)

#Gráfico
lsa_sv_big=lsa_model_bownadv_tr.singular_values_
elbow_lsa(lsa_sv_big,30)


"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//lsa_model_bownadv_tr', 'wb') as handle:
    pickle.dump(lsa_model_bownadv_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//lsa_features_bownadv_tr', 'wb') as handle:
    pickle.dump(lsa_features_bownadv_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Cut-off
    
#F_classif
selector_bownadv_f = SelectKBest(f_classif, k=5)

bownadv_tfidf_f_tr = selector_bownadv_f.fit_transform(bownadv_tr_idfs,train_label)
bownadv_tfidf_f_te = selector_bownadv_f.transform(bownadv_te_idfs)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//bownadv_tfidf_f_tr', 'wb') as handle:
    pickle.dump(bownadv_tfidf_f_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//bownadv_tfidf_f_te', 'wb') as handle:
    pickle.dump(bownadv_tfidf_f_te, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
    
#Chi2
selector_bownadv_chi2 = SelectKBest(chi2, k=5)

bownadv_tfidf_chi2_tr = selector_bownadv_chi2.fit_transform(bownadv_tr_idfs,train_label)
bownadv_tfidf_chi2_te = selector_bownadv_chi2.transform(bownadv_te_idfs)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//bownadv_tfidf_chi2_tr', 'wb') as handle:
    pickle.dump(bownadv_tfidf_chi2_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//bownadv_tfidf_chi2_te', 'wb') as handle:
    pickle.dump(bownadv_tfidf_chi2_te, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Caso queiramos usar LSA
#Teste

lsa_features_bownadv_te = lsa_model_bownadv_tr.transform(bownadv_te_idfs)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//lsa_features_bownadv_te', 'wb') as handle:
    pickle.dump(lsa_features_bownadv_te, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Bown_adj_Td_Idf


lsa_model_bownadj_tr, lsa_features_bownadj_tr = lsa_try(bownadj_tr_idfs,5,bown_adj_tr_features_name)

#Gráfico
lsa_sv_bownadj=lsa_model_bownadj_tr.singular_values_
elbow_lsa(lsa_sv_bownadj,30)


"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//lsa_model_bownadj_tr', 'wb') as handle:
    pickle.dump(lsa_model_bownadj_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//lsa_features_bownadj_tr', 'wb') as handle:
    pickle.dump(lsa_features_bownadj_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Cut-off
    
#F_classif
selector_bownadj_f = SelectKBest(f_classif, k=5)

bownadj_tfidf_f_tr = selector_bownadj_f.fit_transform(bownadj_tr_idfs,train_label)
bownadj_tfidf_f_te = selector_bownadj_f.transform(bownadj_te_idfs)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//bownadj_tfidf_f_tr', 'wb') as handle:
    pickle.dump(bownadj_tfidf_f_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//bownadj_tfidf_f_te', 'wb') as handle:
    pickle.dump(bownadj_tfidf_f_te, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
    
#Chi2
selector_bownadj_chi2 = SelectKBest(chi2, k=5)

bownadj_tfidf_chi2_tr = selector_bownadj_chi2.fit_transform(bownadj_tr_idfs,train_label)
bownadj_tfidf_chi2_te = selector_bownadj_chi2.transform(bownadj_te_idfs)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//bownadj_tfidf_chi2_tr', 'wb') as handle:
    pickle.dump(bownadj_tfidf_chi2_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//bownadj_tfidf_chi2_te', 'wb') as handle:
    pickle.dump(bownadj_tfidf_chi2_te, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Caso queiramos usar LSA
#Teste
lsa_features_bownadj_te = lsa_model_bownadj_tr.transform(bownadj_te_idfs)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data/lsa_features_bownadj_te', 'wb') as handle:
    pickle.dump(lsa_features_bownadj_te, handle, protocol=pickle.HIGHEST_PROTOCOL)




    """
    SECTION  4:
        Supervised learning
            4.1 Compute the Multinominal Naive Bayes for BOW, Bigrams and TFIDF (with lemmas and stopwords)
            4.2 Implement SVM (TIP: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
    """


      
      svc_model=SVC_model(X_train_clean, tr_label_sample)
     
      predicted = prediction(svc_model, a, te_label_sample)
     
      look_at_predictions(predicted, tr_corpus_sample, te_corpus_sample)
"""
SVM
"""

    """
    BOW
    """

#SVM Bow tf_idf
     
    #F-classif
SVM_bow_f= train_predict_evaluate_model(bow_tfidf_f_tr,train_label,
                             bow_tfidf_f_te,test_label)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//SVM_bow_f', 'wb') as handle:
    pickle.dump(SVM_bow_f, handle, protocol=pickle.HIGHEST_PROTOCOL)


SVM_bow_f=pd.read_pickle('SVM_bow_f')
    
    
 # Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, SVM_bow_f)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Bow F-class Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM Bow F-class Normalized confusion matrix')

plt.show()   
    
    
    
    
    
     #Chi2   
    
SVM_bow_chi2= train_predict_evaluate_model(bow_tfidf_chi2_tr,train_label,
                             bow_tfidf_chi2_te,test_label)
        
"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//SVM_bow_chi2', 'wb') as handle:
    pickle.dump(SVM_bow_chi2, handle, protocol=pickle.HIGHEST_PROTOCOL)

SVM_bow_chi2=pd.read_pickle('SVM_bow_chi2')

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, SVM_bow_chi2)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Bow Chi 2Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM  Chi2 Bow Normalized confusion matrix')

plt.show()






#Lsa 5
        SVM_bow_lsa_5= train_predict_evaluate_model(lsa_features_bow_tr,train_label,
                             lsa_features_bow_te,test_label)
"""Export Data with pickle"""
with open('SVM_bow_lsa_5', 'wb') as handle:
    pickle.dump(SVM_bow_lsa_5, handle, protocol=pickle.HIGHEST_PROTOCOL)    


SVM_bow_lsa_5=pd.read_pickle('SVM_bow_lsa_5')

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, SVM_bow_lsa_5)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Lsa Bow Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM Lsa Bow Normalized confusion matrix')

plt.show()




    """
    BOW NOUN ADJ ADV
    """
    
    
    #Tf_idf
    
    #F-classif
    
    SVM_bownadv_f= train_predict_evaluate_model(bownadv_tfidf_f_tr,train_label,
                                                bownadv_tfidf_f_te,test_label)
"""Export Data with pickle"""
with open('SVM_bownadv_f', 'wb') as handle:
    pickle.dump(SVM_bownadv_f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
 # Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, SVM_bownadv_f)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Bigrams F-class Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM Bigrams F-class Normalized confusion matrix')

plt.show()  

    
    #Chi2

    SVM_bownadv_chi2=train_predict_evaluate_model(bownadv_tfidf_chi2_tr,train_label,
                                                  bownadv_tfidf_chi2_te,test_label)

"""Export Data with pickle"""
with open('SVM_bownadv_chi2', 'wb') as handle:
    pickle.dump(SVM_bownadv_chi2, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, SVM_bownadv_chi2)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Bigrams Chi 2Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM  Chi2 Bigrams Normalized confusion matrix')

plt.show()




    #Lsa 5
        SVM_bownadv_lsa_5= train_predict_evaluate_model(lsa_features_bownadv_tr,train_label,
                             lsa_features_bownadv_te,test_label)

"""Export Data with pickle"""
with open('SVM_bownadv_lsa_5', 'wb') as handle:
    pickle.dump(SVM_bownadv_lsa_5, handle, protocol=pickle.HIGHEST_PROTOCOL)

cnf_matrix = confusion_matrix(test_label, SVM_bownadv_lsa_5)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Lsa Bow Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM Lsa Bow Normalized confusion matrix')

plt.show()


"""
BOW_ADJ_NOUN
"""
    #F-classif
    
   SVM_bownadj_f= train_predict_evaluate_model(bownadj_tfidf_f_tr,train_label,
                                               bownadj_tfidf_f_te,test_label)
"""Export Data with pickle"""
with open('SVM_bownadj_f', 'wb') as handle:
    pickle.dump(SVM_bownadj_f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
     # Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, SVM_bownadj_f)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Bow with nouns F-class Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM Bow with nouns F-class Normalized confusion matrix')

plt.show()  

    
    #Chi2

    SVM_bownadj_chi2=train_predict_evaluate_model(bownadj_tfidf_chi2_tr,train_label,
                                                  bownadj_tfidf_chi2_te,test_label)

"""Export Data with pickle"""
with open('SVM_bownadj_chi2', 'wb') as handle:
    pickle.dump(SVM_bownadj_chi2, handle, protocol=pickle.HIGHEST_PROTOCOL)

 # Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, SVM_bownadj_chi2)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Bow with nouns Chi2 Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM Bow with nouns Chi2 Normalized confusion matrix')

plt.show()  


    #Lsa 5

        SVM_bownadj_lsa_5= train_predict_evaluate_model(lsa_features_bownadj_tr,train_label,
                             lsa_features_bownadj_te,test_label)
"""Export Data with pickle"""
with open('SVM_bownadj_lsa_5', 'wb') as handle:
    pickle.dump(SVM_bownadj_lsa_5, handle, protocol=pickle.HIGHEST_PROTOCOL)

cnf_matrix = confusion_matrix(test_label, SVM_bownadj_lsa_5)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=' SVM Lsa Bow  with nouns Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=' SVM Lsa Bow with nouns Normalized confusion matrix')

plt.show()





if __name__ == '__main__':
    main()