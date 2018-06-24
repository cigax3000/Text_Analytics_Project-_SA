from sklearn import linear_model
from src.features.Metrics import get_metrics

#SVM My_Lexicon
def SVM_evaluate_model(train_features, train_labels,
                                 test_features, test_labels):
    # build model
    model = linear_model.SGDClassifier().fit(train_features, train_labels)
    # predict using model
    predictions = model.predict(test_features)
    # evaluate model prediction performance
    get_metrics(true_labels=test_labels,
                predicted_labels=predictions)
    return predictions

SVM_sent_my= SVM_evaluate_model(tr_sparse_matrix,train_label_sent,
                             te_sparse_matrix,test_label_sent)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//svm_sent_my.pickle', 'wb') as handle:
    pickle.dump(SVM_sent_my, handle, protocol=pickle.HIGHEST_PROTOCOL)

#SVM Hu and Liu

SVM_sent_hu= SVM_evaluate_model(tr_sparse_matrix_hu,train_label_sent,
                             te_sparse_matrix_hu,test_label_sent)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//svm_sent_hu.pickle', 'wb') as handle:
    pickle.dump(SVM_sent_hu, handle, protocol=pickle.HIGHEST_PROTOCOL)





