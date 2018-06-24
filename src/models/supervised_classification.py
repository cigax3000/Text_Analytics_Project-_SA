from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def SVC_model(c_index, dataset_target):
   svc_model = SVC(kernel='linear',C=1.0).fit(c_index, dataset_target)

   return svc_model

""" subistituir SVC"""
def prediction(model, tranf_dtest, dataset_test):
    predicted = model.predict(tranf_dtest)

    return predicted


def look_at_predictions(predicted, dataset_train, dataset_test):
    for sample, class_pos in zip(dataset_test, predicted):
        print('%r => %s' % (sample, dataset_train.target_names[class_pos]))

