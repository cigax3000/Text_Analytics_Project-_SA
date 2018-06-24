
from scipy.sparse import coo_matrix

#For my Sentiment

def extract_unique_words(list_of_tup):
    """
    :param list_of_tup: list of tuples
    :return: The unique words from all the reviews
    """
    unique_wrs = set()
    for review in list_of_tup:
        for pair in review:
            unique_wrs.add(pair[0])
    return sorted(list(unique_wrs))

unique_wrd = extract_unique_words(senti_tot)

#Remove duplicated words

def sum_rep (dataset):
    """

    :param dataset: list of list with tuples
    :return: list of list with tuples with unique words of the review
    """
    final =  dict()
    list_without_rep = []
    for review in dataset:
        final.clear()
        for pair in review:
            if pair[0] in final:
                final[pair[0]] = final[pair[0]] + pair[1]
            else:
                final[pair[0]] = pair[1]
        list_without_rep.append([(key,value) for key , value in final.items()])
    return list_without_rep

sentii_train = sum_rep(senti_train)
sentii_test = sum_rep(senti_test)

def create_sparse_matrix(unique_words, senti):
    """

    :param unique_words: the unique words list
    :param senti: the words (not duplicated) and the sentiment weight for each word in the review
    :return: matrix where the row are the review number, the column the unique word (in number) and the data is the sentiment weight
    """
    columns = []
    rows = []
    data = []
    dataset_size = len(senti)
    unique_words_size = len(unique_words)
    print(senti[0])
    for i in tqdm(range(0, dataset_size)):
        for pair in senti[i]:
            index = unique_words.index(pair[0])
            columns.append(index)
            rows.append(i)
            data.append(pair[1])
    data = np.asarray(data)
    rows = np.asarray(rows)
    columns = np.asarray(columns)
    matrix = coo_matrix((data, (rows, columns)),shape=(dataset_size, unique_words_size))

    return matrix

tr_sparse_matrix = create_sparse_matrix(unique_wrd,sentii_train)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//tr_sparse_matrix.pickle', 'wb') as handle:
    pickle.dump(tr_sparse_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

tr_sparse_matrix = pd.read_pickle('//Users//MarioAntao//Documents//ta_project_sa//data//processed//sparce_matrix//tr_sparse_matrix.pickle')

te_sparse_matrix = create_sparse_matrix(unique_wrd,sentii_test)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//te_sparse_matrix.pickle', 'wb') as handle:
    pickle.dump(te_sparse_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)




#For  Hu and Liu

unique_wrd_hu = extract_unique_words(senti_tot_hu)

#Remove duplicated words
sentii_train_hu = sum_rep(senti_train_hu)
sentii_test_hu = sum_rep(senti_test_hu)


#Create Matrix
tr_sparse_matrix_hu = create_sparse_matrix(unique_wrd_hu,sentii_train_hu)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//tr_sparse_matrix_hu.pickle', 'wb') as handle:
    pickle.dump(tr_sparse_matrix_hu, handle, protocol=pickle.HIGHEST_PROTOCOL)

te_sparse_matrix_hu = create_sparse_matrix(unique_wrd_hu,sentii_test_hu)

"""Export Data with pickle"""
with open('//Users//MarioAntao//Documents//ta_project_sa//data//te_sparse_matrix_hu.pickle', 'wb') as handle:
    pickle.dump(te_sparse_matrix_hu, handle, protocol=pickle.HIGHEST_PROTOCOL)
