from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
from nltk import ngrams
import string


# matching features
# ngram overlap
def char_ngram_overlap(sent1, sent2, n=1):
    sent1_ngrams = []
    sent2_ngrams = []
    words1 = [i.strip('.,? ') for i in sent1.split(' ')]
    words2 = [i.strip('.,? ') for i in sent2.split(' ')]
    for word in words1:
        try:
            sent1_ngrams.extend([word[i:i+n] for i in range(len(word)-n+1)])
        except:
            pass
    for word in words2:
        try:
            sent2_ngrams.extend([word[i:i+n] for i in range(len(word)-n+1)])
        except:
            pass
    return calculate_overlap(sent1_ngrams, sent2_ngrams)


def calculate_overlap(sent1_ngrams, sent2_ngrams):
    s1_len = len(sent1_ngrams)
    s2_len = len(sent2_ngrams)
    if s1_len == 0 and s2_len == 0:
        return 0
    s1_s2 = []
    for n in sent1_ngrams:
        if n in sent2_ngrams:
            s1_s2.append(n)
    s1_s2_len = max(1, len(s1_s2))
    return 2 / (s1_len / s1_s2_len + s2_len / s1_s2_len)


def max_sim_text(x, y):
    dp = [([0] * len(y)) for i in range(len(x))]
    maxlen = maxindex = 0
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            if x[i] == y[j]:
                if i != 0 and j != 0:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                if i == 0 or j == 0:
                    dp[i][j] = 1
                if dp[i][j] > maxlen:
                    maxlen = dp[i][j]
                    maxindex = i + 1 - maxlen
    return maxlen, x[maxindex:maxindex + maxlen]


def tag_and_parser(sent1, sent2):
    """
    maximum same text between two postag
    :param sent1: sentence 1
    :param sent2: sentence 2
    :return: percentage of same string in text
    """
    def not_empty(s):
        return s and s.strip()
    sent1 = list(filter(not_empty, sent1.split(" ")))
    sent2 = list(filter(not_empty, sent2.split(" ")))
    pos_sent1 = nltk.pos_tag(sent1)
    pos_sent2 = nltk.pos_tag(sent2)
    pos1, pos2 = "", ""
    for word, pos in pos_sent1:
        pos1 += pos + " "
    for word, pos in pos_sent2:
        pos2 += pos + " "
    maxlen, subseq = max_sim_text(pos1, pos2)
    subseq_s = subseq.split()
    return len(subseq_s) / len(sent1), len(subseq_s) / len(sent2)


def extract_features(train_sents1, train_sents2, test_sents1, test_sents2):
    tf_idf_vec = TfidfVectorizer(
        use_idf=True, smooth_idf=False, ngram_range=(1, 1), stop_words='english')
    sents = np.append(train_sents1, train_sents2, 0)
    sents = tf_idf_vec.fit_transform(sents)
    train_features = []
    test_features = []
    for sent1, sent2 in zip(train_sents1, train_sents2):
        sent_features = []
        for i in range(1, 5):
            sent_features.append(
                char_ngram_overlap(sent1, sent2, n=i))
        sent_features.append(cosine_similarity(tf_idf_vec.transform([sent1]),
                                               tf_idf_vec.transform([sent2]))[0][0])
        # sent_features.append(tag_and_parser(sent1, sent2)[0])
        # sent_features.append(tag_and_parser(sent1, sent2)[1])
        train_features.append(sent_features)

    for sent1, sent2 in zip(test_sents1, test_sents2):
        sent_features = []
        for i in range(1, 5):
            sent_features.append(
                char_ngram_overlap(sent1, sent2, n=i))
        sent_features.append(cosine_similarity(tf_idf_vec.transform([sent1]),
                                               tf_idf_vec.transform([sent2]))[0][0])
        # sent_features.append(tag_and_parser(sent1, sent2)[0])
        # sent_features.append(tag_and_parser(sent1, sent2)[1])
        test_features.append(sent_features)
    train_features = np.array(train_features)
    test_features = np.array(test_features)

    return train_features, test_features
