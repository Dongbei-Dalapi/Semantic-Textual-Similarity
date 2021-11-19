from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import numpy as np
from preprocessing import text_clean
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


def get_w2v_embeddings(sents1, sents2, embedding_dim=300, max_len=20):
    vocabs = {}
    vocabs_count = 0
    sent1_seq = []
    sent2_seq = []
    # get stopwords
    stops = set(stopwords.words('english'))
    # get word2vec
    print("loading embedding model ...")
    word2vec = KeyedVectors.load_word2vec_format(
        "./data/GoogleNews-vectors-negative300.bin.gz", binary=True)
    for sent in sents1:
        s2n = []
        for word in text_clean(sent).split():
            if word in stops:
                continue
            if word not in vocabs:
                vocabs_count += 1
                vocabs[word] = vocabs_count
                s2n.append(vocabs_count)
            else:
                s2n.append(vocabs[word])
        sent1_seq.append(s2n)
    for sent in sents2:
        s2n = []
        for word in text_clean(sent).split():
            if word in stops:
                continue
            if word not in vocabs:
                vocabs_count += 1
                vocabs[word] = vocabs_count
                s2n.append(vocabs_count)
            else:
                s2n.append(vocabs[word])
        sent2_seq.append(s2n)
    # This will be the embedding matrix
    embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)
    embeddings[0] = 0  # So that the padding will be ignored
    for word, index in vocabs.items():
        if word in list(word2vec.index_to_key):
            embeddings[index] = word2vec.get_vector(word)
    del word2vec
    print("embedding matrix built")
    return pad_sequences(sent1_seq, maxlen=max_len), pad_sequences(sent2_seq, maxlen=max_len), embeddings


def get_glove_embeddings(text1, text2):
    word_vec = {}
    tokenizer = tokenizer = Tokenizer()
    tokenizer.fit_on_texts(np.append(np.array(text1), np.array(text2), 0))
    word_index = tokenizer.word_index
    try:
        embedding_matrix = np.load('./data/embedding_matrix.npy')
    except:
        print("Load GloVe model ...")
        with open('./data/glove.840B.300d.txt', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.split(' ')
                word_vec[line[0]] = np.asarray(line[1:], dtype='float32')
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            embedding_vector = word_vec.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector[:300]
        np.save('./data/embedding_matrix.npy', embedding_matrix)
        print("Embedding matrix saved")
    return tokenizer, embedding_matrix
