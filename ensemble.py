from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import numpy as np
import os
from traditional_nlp_model import extract_features
import re


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


def ensemble_predict(models, x_test):
    predictions = []
    for model in models:
        predictions.append(model.predict(x_test))
    preds = sum(predictions) / len(models)
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


def load_data(folder_path):
    """
    Load training data
    :param folder_path: the folder path
    :return: sentences and labels
    """
    text1 = []
    text2 = []
    labels = []
    for filename in os.listdir(folder_path):
        input = re.search(r'.*input.*txt', filename)
        if input:
            with open(folder_path + '/' + filename) as f:
                text_lines = f.readlines()
            gs_file = filename.replace("input", "gs")
            with open(folder_path + '/' + gs_file) as f:
                labels_lines = f.readlines()
            for text, label in zip(text_lines, labels_lines):
                if label != '' and label != '\n':
                    t1 = text.strip().split('\t')[0]
                    t2 = text.strip().split('\t')[1]
                    text1.append(t1)
                    text2.append(t2)
                    labels.append(float(label))
    return text1, text2, labels


def load_eval_data(folder_path, category):
    """
    load testing data and remove data without label
    :param folder_path: the folder path
    :param category: source category
    :return: sentences and labels
    """
    text1 = []
    text2 = []
    labels = []
    try:
        with open(folder_path + '/' + "STS2016.input." + category + ".txt") as f:
            text_lines = f.readlines()
        with open(folder_path + '/' + "STS2016.gs." + category + ".txt") as f:
            labels_lines = f.readlines()
        for text, label in zip(text_lines, labels_lines):
            if label != '' and label != '\n':
                t1 = text.strip().split('\t')[0]
                t2 = text.strip().split('\t')[1]
                text1.append(t1)
                text2.append(t2)
                labels.append(float(label))
    except:
        print('not a valid category name')
    return text1, text2, labels


if __name__ == '__main__':
    # get the directory
    root_dir = os.getcwd()
    data_dir = root_dir + "/data"
    train_dir = data_dir + "/training"
    test_dir = data_dir + "/testing"
    # load data
    # 2012 - train - test
    train_text1, train_text2, train_labels = load_data(
        train_dir + "/2012/train")
    test_2012 = load_data(train_dir + "/2012/train")
    train_text1.extend(test_2012[0])
    train_text2.extend(test_2012[1])
    train_labels.extend(test_2012[2])
    # 2013 - 2015
    for i in range(2013, 2016):
        text1, text2, label = load_data(train_dir + "/" + str(i))
        train_text1.extend(text1)
        train_text2.extend(text2)
        train_labels.extend(label)
    # 2016 - testing data
    test_text1_headline, test_text2_headline, test_labels_headline = load_eval_data(
        test_dir + "/2016", 'headlines')
    test_text1_plagiarism, test_text2_plagiarism, test_labels_plagiarism = load_eval_data(
        test_dir + "/2016", 'plagiarism')
    test_text1_postediting, test_text2_postediting, test_labels_postediting = load_eval_data(
        test_dir + "/2016", 'postediting')
    test_text1_aa, test_text2_aa, test_labels_aa = load_eval_data(
        test_dir + "/2016", 'answer-answer')
    test_text1_qq, test_text2_qq, test_labels_qq = load_eval_data(
        test_dir + "/2016", 'question-question')
    test_text1, test_text2, test_labels = load_data(
        test_dir + '/2016')

    train_features, test_features = extract_features(train_text1, train_text2,
                                                     test_text1, test_text2)

    train_ngram_overlap = train_features[:, 0:4]
    train_bow = train_features[:, [4]]
    # train_pos_overlap = train_features[:, [5, 6]]

    test_ngram_overlap = test_features[:, 0:4]
    test_bow = test_features[:, [4]]
    # test_pos_overlap = test_features[:, [5, 6]]

    test_index_start = 0
    test_index_end = len(test_text1_headline)
    test_ngram_overlap_headline = test_features[test_index_start:test_index_end, 0:4]
    test_bow_headline = test_features[test_index_start:test_index_end, [4]]
    test_headline = test_features[test_index_start:test_index_end, :]
    test_labels_headline = test_labels[test_index_start:test_index_end]

    test_index_start = test_index_end
    test_index_end += len(test_text1_plagiarism)
    test_ngram_overlap_plagiarism = test_features[test_index_start:test_index_end, 0:4]
    test_bow_plagiarism = test_features[test_index_start:test_index_end, [4]]
    test_plagiarism = test_features[test_index_start:test_index_end, :]
    test_labels_plagiarism = test_labels[test_index_start:test_index_end]

    test_index_start = test_index_end
    test_index_end += len(test_text1_postediting)
    test_ngram_overlap_postediting = test_features[test_index_start:test_index_end, 0:4]
    test_bow_postediting = test_features[test_index_start:test_index_end, [4]]
    test_postediting = test_features[test_index_start:test_index_end, :]
    test_labels_postediting = test_labels[test_index_start:test_index_end]

    test_index_start = test_index_end
    test_index_end += len(test_text1_aa)
    test_ngram_overlap_aa = test_features[test_index_start:test_index_end, 0:4]
    test_bow_aa = test_features[test_index_start:test_index_end, [4]]
    test_aa = test_features[test_index_start:test_index_end, :]
    test_labels_aa = test_labels[test_index_start:test_index_end]

    test_index_start = test_index_end
    test_index_end += len(test_text1_qq)
    test_ngram_overlap_qq = test_features[test_index_start:test_index_end, 0:4]
    test_bow_qq = test_features[test_index_start:test_index_end, [4]]
    test_qq = test_features[test_index_start:test_index_end, :]
    test_labels_qq = test_labels[test_index_start:test_index_end]

    # ngram overlap
    models = ensemble(train_ngram_overlap, np.array(train_labels))
    preds = ensemble_predict(models, test_ngram_overlap)
    print('Pearsons ngram overlap  = %2.4f' % pearsonr(preds, test_labels)[0])
    preds = ensemble_predict(models, test_ngram_overlap_headline)
    print('Pearsons ngram overlap headline = %2.4f' %
          pearsonr(preds, test_labels_headline)[0])
    preds = ensemble_predict(models, test_ngram_overlap_plagiarism)
    print('Pearsons ngram overlap plagiarism = %2.4f' %
          pearsonr(preds, test_labels_plagiarism)[0])
    preds = ensemble_predict(models, test_ngram_overlap_postediting)
    print('Pearsons ngram overlap postediting = %2.4f' %
          pearsonr(preds, test_labels_postediting)[0])
    preds = ensemble_predict(models, test_ngram_overlap_aa)
    print('Pearsons ngram overlap A-A = %2.4f' %
          pearsonr(preds, test_labels_aa)[0])
    preds = ensemble_predict(models, test_ngram_overlap_qq)
    print('Pearsons ngram overlap Q-Q = %2.4f' %
          pearsonr(preds, test_labels_qq)[0])

    # BOW - tf-idf cosine similarity
    models = ensemble(train_bow, np.array(train_labels))
    preds = ensemble_predict(models, test_bow)
    print('Pearsons BOW = %2.4f' % pearsonr(preds, test_labels)[0])
    preds = ensemble_predict(models, test_bow_headline)
    print('Pearsons BOW headline = %2.4f' %
          pearsonr(preds, test_labels_headline)[0])
    preds = ensemble_predict(models, test_bow_plagiarism)
    print('Pearsons BOW plagiarism = %2.4f' %
          pearsonr(preds, test_labels_plagiarism)[0])
    preds = ensemble_predict(models, test_bow_postediting)
    print('Pearsons BOW postediting = %2.4f' %
          pearsonr(preds, test_labels_postediting)[0])
    preds = ensemble_predict(models, test_bow_aa)
    print('Pearsons BOW A-A = %2.4f' % pearsonr(preds, test_labels_aa)[0])
    preds = ensemble_predict(models, test_bow_qq)
    print('Pearsons BOW Q-Q = %2.4f' % pearsonr(preds, test_labels_qq)[0])

    # all features
    models = ensemble(train_features, np.array(train_labels))
    preds = ensemble_predict(models, test_features)
    print('Pearsons all = %2.4f' % pearsonr(preds, test_labels)[0])
    preds = ensemble_predict(models, test_headline)
    print('Pearsons headline = %2.4f' %
          pearsonr(preds, test_labels_headline)[0])
    preds = ensemble_predict(models, test_plagiarism)
    print('Pearsons plagiarism = %2.4f' %
          pearsonr(preds, test_labels_plagiarism)[0])
    preds = ensemble_predict(models, test_postediting)
    print('Pearsons postediting = %2.4f' %
          pearsonr(preds, test_labels_postediting)[0])
    preds = ensemble_predict(models, test_aa)
    print('Pearsons A-A = %2.4f' % pearsonr(preds, test_labels_aa)[0])
    preds = ensemble_predict(models, test_qq)
    print('Pearsons Q-Q = %2.4f' % pearsonr(preds, test_labels_qq)[0])
