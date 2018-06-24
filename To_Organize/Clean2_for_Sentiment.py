"""
In this clean we will use the dict of emoticons
And the output text_cleaned  will be a list (review) of str
"""

from src.features.process_tweet.clean_tweet import clean_tweet
from src.features.Clean_4_Sentiment import merge_flat_list_label_ov
from tqdm import tqdm


review_train_list= pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//review_train_lis_V2.pickle')
review_test_list = pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//review_test_lis_V2.pickle')

review_train_afclean1= pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//processed//review_train_lis_afterclean1.pickle')
review_test_aftclean1 = pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//processed//review_train_lis_afterclean1.pickle')


review_train_sent_list = review_train_list
review_test_sent_list = review_test_list


def main():
    # Clean tweets.
    for review in tqdm(review_test_sent_list): clean_tweet(review)


if __name__ == '__main__':
    main()

#test = [review.text_cleaned for review in a]
#Normal
#Create Corpus
train_corpus=merge_flat_list(review_train_list)
test_corpus=merge_flat_list(review_test_list)

#Create Labels
train_label=merge_flat_list_label_ov(review_train_list)
test_label=merge_flat_list_label_ov(review_test_list)



#Sentiment
train_corpus_sent = [review.text_cleaned for review in review_train_sent_list]
train_label_sent = merge_flat_list_label_ov(review_train_sent_list)
test_corpus_sent = [review.text_cleaned for review in review_test_sent_list]
test_label_sent = merge_flat_list_label_ov(review_test_sent_list)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//review_train_sent_list.pickle', 'wb') as handle:
    pickle.dump(review_train_sent_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//review_test_sent_list.pickle', 'wb') as handle:
    pickle.dump(review_test_sent_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//train_corpus_sent.pickle', 'wb') as handle:
    pickle.dump(train_corpus_sent, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//train_label_sent.pickle', 'wb') as handle:
    pickle.dump(train_label_sent, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//test_corpus_sent.pickle', 'wb') as handle:
    pickle.dump(test_corpus_sent, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//test_label_sent.pickle', 'wb') as handle:
    pickle.dump(test_label_sent, handle, protocol=pickle.HIGHEST_PROTOCOL)




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
asss= create_list_list(train_corpus_sent)


