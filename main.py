
from lstm_model import BERT_Model, CNN_Model, LSTM_Model, MaLSTM, BertSemanticDataGenerator
from preprocessing import load_data, load_eval_data, text_clean
from word2vec import get_w2v_embeddings, get_glove_embeddings
import argparse
import os
import numpy as np
from scipy.stats import pearsonr
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # set up the argument
    argp = argparse.ArgumentParser()
    argp.add_argument('--epoch', default=50)
    argp.add_argument('--batch_size', default=256)
    argp.add_argument('--BERT', default=False)
    argp.add_argument('--word2vec', default='glove')
    args = argp.parse_args(args=[])
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
    if args.BERT:
        train_text, val_text, train_labels, val_labels = train_test_split(
            list(zip(train_text1, train_text2)), train_labels, test_size=0.3, random_state=1)
        train_text = [[i for i, j in train_text], [j for i, j in train_text]]
        val_text = [[i for i, j in val_text], [j for i, j in val_text]]
        train_text1 = train_text[0]
        train_text2 = train_text[1]
        val_text1 = val_text[0]
        val_text2 = val_text[1]
    # 2016 - testing data
    categories = ['headlines', 'plagiarism', 'postediting',
                  'answer-answer', 'question-question']
    test_text1, test_text2, test_labels, test_one_hot_labels, indexs = load_eval_data(
        test_dir + '/2016', categories)
    max_len = 30
    # if not use BERT
    if not args.BERT:
        if args.word2vec == "glove":
            # Download glove.840B.300d.zip at https://nlp.stanford.edu/projects/glove/
            # place it in the data folder
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
        else:
            train_seq1, train_seq2, embedding_matrix = get_w2v_embeddings(
                train_text1, train_text2, max_len=max_len)
            val_seq1, val_seq2, embedding_matrix = get_w2v_embeddings(
                val_text1, val_text2, max_len=max_len)
            test_seq1, test_seq2, _ = get_w2v_embeddings(
                test_text1, test_text2, max_len=max_len)
        # train model

        model = CNN_Model(embedding_matrix, seq_dim=max_len, cnnfilters=300,
                          epoch=args.epoch, batch_size=64, lr=0.0001)
        # model = MaLSTM(embedding_matrix, seq_dim=max_len, epoch = args.epoch, batch_size = 256)
        model.train(train_seq1, train_seq2, train_one_hot_labels)
        p = model.predict(train_seq1, train_seq2)
        print(p)
        predictions = []
        gold_labels = []
        for i in range(len(indexs)):
            if i != len(indexs)-1:
                predictions.append(model.predict(
                    test_seq1[indexs[i]:indexs[i+1], :], test_seq2[indexs[i]:indexs[i+1], :]))  # .round())
                gold_labels.append(test_labels[indexs[i]:indexs[i+1]])
        all_p = model.predict(test_seq1, test_seq2)
        print('Pearsons Training data = %2.4f' %
              pearsonr(p, train_labels)[0])
        for c, p, g in zip(categories, predictions, gold_labels):
            print('Pearsons %s = %2.4f' %
                  (c, pearsonr(p, g)[0]))
        print('Pearsons All testing data = %2.4f' %
              pearsonr(all_p, test_labels)[0])
    else:  # use BERT
        train_data = BertSemanticDataGenerator(np.append(np.array(train_text1).reshape(-1, 1), np.array(
            train_text2).reshape(-1, 1), 1), np.array(train_labels), batch_size=args.batch_size, shuffle=True, max_len=max_len)
        val_data = BertSemanticDataGenerator(np.append(np.array(val_text1).reshape(-1, 1), np.array(
            val_text2).reshape(-1, 1), 1), np.array(val_labels), batch_size=args.batch_size, shuffle=True, max_len=max_len)
        test_data = BertSemanticDataGenerator(np.append(np.array(test_text1).reshape(-1, 1), np.array(
            test_text2).reshape(-1, 1), 1), np.array(test_labels), batch_size=1, shuffle=False, max_len=max_len)
        model = BERT_Model(seq_dim=max_len, epoch=args.epoch,
                           batch_size=args.batch_size)
        model.train(train_data, val_data)

        p = model.predict(test_data)
        print(p)
        print(pearsonr(np.squeeze(p), test_labels)[0])

        predictions = []
        gold_labels = []
        for i in range(len(indexs)):
            if i != len(indexs)-1:
                text1 = test_text1[indexs[i]:indexs[i+1], :]
                text2 = test_text2[indexs[i]:indexs[i+1], :]
                data = BertSemanticDataGenerator(np.append(np.array(text1).reshape(-1, 1), np.array(
                    text2).reshape(-1, 1), 1), np.array(test_labels), batch_size=1, shuffle=False, max_len=max_len)
                predictions.append(model.predict(data))
                gold_labels.append(test_labels[indexs[i]:indexs[i+1]])
        all_p = model.predict(test_data)
        print('Pearsons Training data = %2.4f' %
              pearsonr(np.squeeze(p), train_labels)[0])
        for c, p, g in zip(categories, predictions, gold_labels):
            print('Pearsons %s = %2.4f' % (c, pearsonr(np.squeeze(p), g)[0]))
        print('Pearsons All testing data = %2.4f' %
              pearsonr(np.squeeze(all_p), test_labels)[0])
