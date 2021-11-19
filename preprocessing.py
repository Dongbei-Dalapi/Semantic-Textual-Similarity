import re
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from nltk.corpus import stopwords


def text_clean(text):
    text = str(text).lower()
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"\'m", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"\s{2,}", " ", text)
    #stop_words = stopwords.words('english')
    #filter_words = [w for w in text.split() if not w in stop_words]
    return text  # " ".join(filter_words).strip()


def one_hot_encode(label):
    label_one_hot = [0] * 6
    label = round(label)
    label_one_hot[label] = 1
    return label_one_hot


def load_data(folder_path, one_hot=False):
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
                    text1.append(text_clean(t1))
                    text2.append(text_clean(t2))
                    if one_hot:
                        labels.append(one_hot_encode(float(label)))
                    else:
                        labels.append(float(label))
    return text1, text2, labels


def load_eval_data(folder_path, categories, one_hot=False):
    """
    load testing data and remove data without label
    :param folder_path: the folder path
    :param category: a list of source category
    :return: sentences and labels
    """
    text1 = []
    text2 = []
    labels = []
    indexs = [0]
    current_len = 0
    for category in categories:
        labels_count = 0
        try:
            with open(folder_path + '/' + "STS2016.input." + category + ".txt") as f:
                text_lines = f.readlines()
            with open(folder_path + '/' + "STS2016.gs." + category + ".txt") as f:
                labels_lines = f.readlines()
            for text, label in zip(text_lines, labels_lines):
                if label != '' and label != '\n':
                    labels_count += 1
                    t1 = text.strip().split('\t')[0]
                    t2 = text.strip().split('\t')[1]
                    text1.append(text_clean(t1))
                    text2.append(text_clean(t2))
                    if one_hot:
                        labels.append(one_hot_encode(float(label)))
                    else:
                        labels.append(float(label))
            current_len += labels_count
            indexs.append(current_len)
        except:
            print('not a valid category name')
    return text1, text2, labels, indexs
