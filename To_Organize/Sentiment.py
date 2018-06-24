#Lista em sentence
#Lista review
#Usar Test
#list_of_string list list of string
#tuple tuples a palavra e o sentimento

from To_Organize.words import get_polarity_map

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



neutral = ['despite','although','though', 'however','but']

negative = ['no','not','none','nobody','nothing','neither','nowhere'',never','hardly','scarcely','barely']

#Neut_neg

def sent(list_of_strings, sentiment_lexicon, negative, neutral):
    """

    :param list_of_strings: corpus of the review
    :param sentiment_lexicon: sentiment lexicon created
    :param negative: list of negative words
    :param neutral: list of neutral words
    :return:
    """
    l=[]
    for review in tqdm(list_of_strings):
        l=[]
        for sentence in review:
            sentence = sentence.split()
            for i in range(0, len(sentence)):
                if sentence[i] in sentiment_lexicon:
                    val = sentiment_lexicon[sentence[i]]
                    l2.append((sentence[i], val))
                    if i + 1 < len(sentence) and sentence[i+1] in negative:
                        l2.append((sentence[i] + "_" + sentence[i+1], -1*val))
                    if i + 2 < len(sentence) and sentence[i + 2] in negative:
                        l2.append((sentence[i] + "_" + sentence[i + 2], -1*val))
                    if i + 1 < len(sentence) and sentence[i+1] in neutral:
                        l2.append((sentence[i] + "_" + sentence[i+1], 0*val))
                    if i + 2 < len(sentence) and sentence[i + 2] in neutral:
                        l2.append((sentence[i] + "_" + sentence[i + 2], 0*val))
        l.append(l2)
    return l

def transform_dict (lexico):
    lexico_dict = dict()
    for pair in lexico:
        lexico_dict[pair[0]] = pair[1]
    return lexico_dict


lexicon_dic= transform_dict(lexicon_fin)
sentiment = sent(train_corpus_sent,lexicon_dic, negative, neutral)

#Train and Test sentiment
senti_train = sent(train_corpus_sent,lexicon_dic, negative, neutral)
senti_test = sent(test_corpus_sent,lexicon_dic, negative, neutral)
senti_tot = senti_train + senti_test

#Train and Test Hu and Liu
senti_hu = get_polarity_map()

senti_train_hu = sent(train_corpus_sent,senti_hu, negative, neutral)
senti_test_hu = sent(test_corpus_sent,senti_hu, negative, neutral)
senti_tot_hu= senti_train_hu + senti_test_hu

