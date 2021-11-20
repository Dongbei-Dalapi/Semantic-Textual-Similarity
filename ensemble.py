from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import numpy as np
import os
from traditional_nlp_model import extract_features
import re
from lstm_model import CNN_Model, MaLSTM
from word2vec import get_glove_embeddings
from keras.preprocessing.sequence import pad_sequences
from preprocessing import load_data, load_eval_data


def ensemble(x_train, y_train):
    rfr = RandomForestRegressor(
        n_jobs=-1, random_state=10, oob_score=True, n_estimators=300)
    rfr.fit(x_train, y_train)
    gbr = GradientBoostingRegressor(
        random_state=10, learning_rate=0.01, loss="ls", n_estimators=300)
    gbr.fit(x_train, y_train)
    xg = XGBRegressor(nthread=4)
    xg.fit(x_train, y_train)
    return [rfr, gbr, xg]  # xg


def ensemble_predict(models, x_test, lstm_model, s1, s2, cnn_model):
    predictions = []
    for model in models:
        predictions.append(model.predict(x_test))
    predictions.append(lstm_model.predict(s1, s2).flatten())
    predictions.append(cnn_model.predict(s1, s2))
    preds = sum(predictions) / (len(models) + 2)
    classified_preds = []
    for i in range(len(preds)):
        p = round(preds[i])
        if p > 5:
            classified_preds.append(5)
        elif p < 0:
            classified_preds.append(0)
        else:
            classified_preds.append(p)
    return np.array(classified_preds)


if __name__ == '__main__':
    # get the directory
    root_dir = os.getcwd()
    data_dir = root_dir + "/data"
    train_dir = data_dir + "/training"
    test_dir = data_dir + "/testing"
    # load data
    # 2012 - train - test
    train_text1, train_text2, train_labels, train_one_hot_labels = load_data(
        train_dir + "/2012/train")
    test_2012 = load_data(train_dir + "/2012/test",)
    train_text1.extend(test_2012[0])
    train_text2.extend(test_2012[1])
    train_labels.extend(test_2012[2])
    train_one_hot_labels.extend(test_2012[3])
    # 2013 - 2015
    for i in range(2013, 2016):
        text1, text2, label, one_hot = load_data(train_dir + "/" + str(i))
        train_text1.extend(text1)
        train_text2.extend(text2)
        train_labels.extend(label)
        train_one_hot_labels.extend(one_hot)
    # 2016 - testing data
    categories = ['headlines', 'plagiarism', 'postediting',
                  'answer-answer', 'question-question']
    test_text1, test_text2, test_labels, test_one_hot_labels, indexs = load_eval_data(
        test_dir + '/2016', categories)

    train_features, test_features = extract_features(train_text1, train_text2,
                                                     test_text1, test_text2)
    max_len = 20
    tokenzier, embedding_matrix = get_glove_embeddings(
        train_text1, train_text2)
    train_seq1 = tokenzier.texts_to_sequences(train_text1)
    train_seq2 = tokenzier.texts_to_sequences(train_text2)
    train_seq1 = pad_sequences(train_seq1, maxlen=max_len)
    train_seq2 = pad_sequences(train_seq2, maxlen=max_len)
    test_seq1 = tokenzier.texts_to_sequences(test_text1)
    test_seq2 = tokenzier.texts_to_sequences(test_text2)
    test_seq1 = pad_sequences(test_seq1, maxlen=max_len)
    test_seq2 = pad_sequences(test_seq2, maxlen=max_len)

    train_ngram_overlap = train_features[:, 0:4]
    train_bow = train_features[:, [4]]

    test_ngram_overlap = test_features[:, 0:4]
    test_bow = test_features[:, [4]]

    test_ngram_overlap_c = []
    test_bow_c = []
    test_all_c = []
    test_labels_c = []
    test_seq1_c = []
    test_seq2_c = []
    for i in range(len(indexs)):
        if i != len(indexs)-1:
            test_ngram_overlap_c.append(
                test_features[indexs[i]:indexs[i+1], 0:4])
            test_bow_c.append(test_features[indexs[i]:indexs[i+1], [4]])
            test_all_c.append(test_features[indexs[i]:indexs[i+1], :])
            test_labels_c.append(test_labels[indexs[i]:indexs[i+1]])
            test_seq1_c.append(test_seq1[indexs[i]:indexs[i+1], :])
            test_seq2_c.append(test_seq2[indexs[i]:indexs[i+1], :])

    # all features
    models = ensemble(train_features, np.array(train_labels))
    lstm_model = MaLSTM(embedding_matrix, seq_dim=max_len,
                        epoch=50, batch_size=256)
    lstm_model.train(train_seq1, train_seq2, train_labels)
    cnn_model = CNN_Model(embedding_matrix, seq_dim=max_len,
                          epoch=50, batch_size=256)
    cnn_model.train(train_seq1, train_seq2, train_one_hot_labels)
    predictions = []
    for c, test_c, g, seq1, seq2 in zip(categories, test_all_c, test_labels_c, test_seq1_c, test_seq2_c):
        p = ensemble_predict(models, test_c, lstm_model, seq1, seq2, cnn_model)
        print('Pearsons %s = %2.4f' % (c, pearsonr(np.squeeze(p), g)[0]))
    p = ensemble_predict(models, test_features,
                         lstm_model, test_seq1, test_seq2, cnn_model)
    print('Pearsons all test = %2.4f' %
          pearsonr(np.squeeze(p), test_labels)[0])
