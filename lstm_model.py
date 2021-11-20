import warnings
from typing import Sequence
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, LSTM, concatenate, Layer, Flatten, Dense, Dropout, Bidirectional, GlobalMaxPooling1D, GRU, GlobalAveragePooling1D, Lambda, multiply
from word2vec import get_w2v_embeddings
import os
from keras import backend as K
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.models import load_model
import numpy as np
import tensorflow as tf
import transformers
from matplotlib import pyplot
from keras.regularizers import l2
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")


class MaLSTM:
    def __init__(self, embedding_matrix, embedding_dim=300, seq_dim=20,
                 dropout=0.1, lstm_unit=64, epoch=30, batch_size=256, lr=0.005, dense_unit=64):
        self.embedding_dim = embedding_dim
        self.seq_dim = seq_dim
        self.dropout = dropout
        self.lstm_unit = lstm_unit
        self.epoch = epoch
        self.batch_size = batch_size
        self.embedding_matrix = embedding_matrix
        self.lr = lr
        self.dense_unit = dense_unit
        self.model = self.build_model()

    def build_model(self):
        s1 = Input(shape=(self.seq_dim,), dtype='int32')
        s2 = Input(shape=(self.seq_dim,), dtype='int32')
        m = Sequential()
        m.add(Embedding(self.embedding_matrix.shape[0], self.embedding_dim, input_length=self.seq_dim, weights=[
            self.embedding_matrix], trainable=False))
        initializer = tf.keras.initializers.Orthogonal()
        m.add(LSTM(self.lstm_unit, kernel_initializer=initializer,
              dropout=0.1, recurrent_dropout=0.1, recurrent_regularizer=l2(0.01)))  # recurrent_regularizer=l2(0.01), kernel_regularizer=l2(0.01))
        d = ManDist()([m(s1), m(s2)])
        d = Dense(self.dense_unit, activation='relu')(d)
        d = Dropout(self.dropout)(d)
        d = Dense(self.dense_unit/2, activation='relu')(d)
        d = Dropout(self.dropout)(d)
        d = Dense(1, activation='sigmoid')(d)

        model = Model(inputs=[s1, s2], outputs=d)
        opt = optimizers.Adam(learning_rate=self.lr)
        model.compile(optimizer=opt,
                      loss="mse", metrics=['acc'])
        return model

    def train(self, train_data1, train_data2, train_label):
        train_label = np.array(train_label) / 5
        es_callback = EarlyStopping(
            monitor='val_loss', patience=3, verbose=1)
        history = self.model.fit([train_data1, train_data2], train_label, batch_size=self.batch_size,
                                 epochs=self.epoch, verbose=1, shuffle=True, validation_split=0.3, callbacks=[es_callback])
        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.title('model train vs validation loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        pyplot.savefig('./train_val_loss.png')
        pyplot.close()

    def predict(self, sent1, sent2):
        p = self.model.predict([sent1, sent2]) * 5
        return p


class ManDist(Layer):
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]),
                                   axis=1, keepdims=True))
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


class LSTM_Model:
    def __init__(self, embedding_matrix, embedding_dim=300, seq_dim=20,
                 dropout=0.2, lstm_unit=10, epoch=30, batch_size=64):
        self.embedding_dim = embedding_dim
        self.seq_dim = seq_dim
        self.dropout = dropout
        self.lstm_unit = lstm_unit
        self.epoch = epoch
        self.batch_size = batch_size
        self.embedding_matrix = embedding_matrix
        self.model = self.build_model()

    def build_model(self):
        s1 = Input(shape=(self.seq_dim,), dtype='int32')
        s2 = Input(shape=(self.seq_dim,), dtype='int32')
        m = Sequential()
        m.add(Embedding(self.embedding_matrix.shape[0], self.embedding_dim, input_length=self.seq_dim, weights=[
            self.embedding_matrix], trainable=False))

        initializer = tf.keras.initializers.Orthogonal()
        m.add(LSTM(self.lstm_unit, kernel_initializer=initializer))
        merge = concatenate([m(s1), m(s2)])
        merge = Flatten()(merge)
        merge = Dense(128, activation='relu')(merge)
        merge = Dropout(self.dropout)(merge)
        merge = Dense(64, activation='relu')(merge)
        merge = Dropout(self.dropout)(merge)
        d = Dense(1, activation='sigmoid')(merge)
        model = Model(inputs=[s1, s2], outputs=d)
        opt = optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt,
                      loss="binary_crossentropy", metrics=['acc'])
        return model

    def train(self, train_data1, train_data2, train_label, val_data, val_label):
        train_label = np.array(train_label) / 5
        es_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        history = self.model.fit([train_data1, train_data2], train_label, batch_size=self.batch_size,
                                 epochs=self.epoch, verbose=1, shuffle=True, validation_split=0.3, callbacks=[es_callback])
        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.title('model train vs validation loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        pyplot.savefig('./train_val_loss.png')
        pyplot.close()

    def predict(self, sent1, sent2):
        p = self.model.predict([sent1, sent2]) * 5
        return np.round(p)


class CNN_Model:
    def __init__(self, embedding_matrix, embedding_dim=300, seq_dim=20,
                 dropout=0.2, epoch=30, batch_size=64, lr=0.01, cnnfilters=300):
        self.embedding_dim = embedding_dim
        self.seq_dim = seq_dim
        self.dropout = dropout
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.cnnfilters = cnnfilters
        self.embedding_matrix = embedding_matrix
        self.model = self.build_model()

    def build_model(self):
        s1 = Input(shape=(self.seq_dim,), dtype='int32')
        s2 = Input(shape=(self.seq_dim,), dtype='int32')
        m = Sequential()
        m.add(Embedding(self.embedding_matrix.shape[0], self.embedding_dim, input_length=self.seq_dim, weights=[
            self.embedding_matrix], trainable=False))

        m.add(Conv1D(self.cnnfilters, 1, padding='valid',
              activation='relu', kernel_initializer='he_uniform'))
        m.add(MaxPool1D(pool_size=self.seq_dim))
        m.add(Flatten())
        absDifference = Lambda(lambda X: K.abs(X[0] - X[1]))([m(s1), m(s2)])
        mulDifference = multiply([m(s1), m(s2)])
        allDifference = concatenate([absDifference, mulDifference])
        allDifference = Dense(units=int(self.cnnfilters), activation='tanh',
                              kernel_initializer='he_uniform')(allDifference)
        dropout = Dropout(self.dropout)(allDifference)
        output = Dense(6, activation='softmax',
                       kernel_initializer='he_uniform')(dropout)
        model = Model(inputs=[s1, s2], outputs=output)
        opt = optimizers.Adam(learning_rate=self.lr)
        model.compile(optimizer=opt,
                      loss='categorical_crossentropy', metrics=['acc'])
        return model

    def _lossfunction(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        ny_true = y_true[:, 1] + 2*y_true[:, 2] + 3 * \
            y_true[:, 3] + 4*y_true[:, 4] + 5*y_true[:, 5]
        ny_pred = y_pred[:, 1] + 2*y_pred[:, 2] + 3 * \
            y_pred[:, 3] + 4*y_pred[:, 4] + 5*y_pred[:, 5]
        my_true = K.mean(ny_true)
        my_pred = K.mean(ny_pred)
        var_true = (ny_true - my_true)**2
        var_pred = (ny_pred - my_pred)**2
        return -K.sum((ny_true-my_true)*(ny_pred-my_pred), axis=-1) / (K.sqrt(K.sum(var_true, axis=-1)*K.sum(var_pred, axis=-1)))

    def train(self, train_data1, train_data2, train_label):
        train_label = np.array(train_label)
        es_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        history = self.model.fit([train_data1, train_data2], train_label, batch_size=self.batch_size,
                                 epochs=self.epoch, verbose=1, shuffle=True, validation_split=0.3, callbacks=[es_callback])
        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.title('model train vs validation loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        pyplot.savefig('./train_val_loss.png')
        pyplot.close()

    def predict(self, sent1, sent2):
        p = self.model.predict([sent1, sent2])
        return np.argmax(p, axis=1)


class BERT_Model:
    def __init__(self, seq_dim=20,
                 dropout=0.3, lstm_unit=64, epoch=30, batch_size=16, lr=0.01, model=None):
        self.seq_dim = seq_dim
        self.dropout = dropout
        self.lstm_unit = lstm_unit
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        if model:
            self.model = model
        else:
            self.model = self.build_model()

    def build_model(self):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            input_ids = tf.keras.layers.Input(shape=(self.seq_dim,),
                                              dtype="int32", name="input_ids")
            attention_masks = tf.keras.layers.Input(shape=(self.seq_dim,),
                                                    dtype="int32", name="attention_masks")
            token_type_ids = tf.keras.layers.Input(
                shape=(self.seq_dim,), dtype="int32", name="token_type_ids")
            # Loading pretrained BERT model.
            bert_model = transformers.TFBertModel.from_pretrained(
                "bert-base-uncased")
            bert_output = bert_model(
                input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
            sequence_output = bert_output.last_hidden_state
            pooled_output = bert_output.pooler_output
            bi_lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(self.lstm_unit, return_sequences=True))(sequence_output)
            avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
            max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
            concat = tf.keras.layers.concatenate([avg_pool, max_pool])
            dropout = tf.keras.layers.Dropout(self.dropout)(concat)
            output = tf.keras.layers.Dense(1, activation="sigmoid")(dropout)
            model = tf.keras.models.Model(inputs=[input_ids, attention_masks,
                                                  token_type_ids], outputs=[output])
            opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
            model.compile(optimizer=opt,
                          loss="mse", metrics=['acc'])
            return model

    def train(self, train_data, val_data):
        es_callback = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
        self.model.fit(train_data, batch_size=self.batch_size, epochs=self.epoch,
                       use_multiprocessing=True, workers=-1, validation_data=val_data, callbacks=[es_callback])

    def predict(self, test_data):
        return self.model.predict(test_data) * 5

    def save(self, filename='./BERT_Model'):
        self.model.save(filename)

    def load(self, filename='./BERT_Model'):
        self.model = load_model(filename)


class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.
    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.
    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(self, sentence_pairs, labels, batch_size=32,
                 shuffle=True, include_targets=True, max_len=20):
        self.sentence_pairs = sentence_pairs
        self.labels = labels / 5
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        self.max_len = max_len
        # Load our BERT Tokenizer to encode the text.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx *
                               self.batch_size: (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]
        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            truncation=True,
            return_tensors="tf")
        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")
        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)
