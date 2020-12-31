# -*- coding:utf-8 -*-
# @Time: 2020/12/28 15:18
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: LSTM.py

import os
import re
import pandas as pd
from keras.models import Sequential
from keras import preprocessing, optimizers
from keras.preprocessing.text import Tokenizer
from keras.layers import Flatten, Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping

def get_reviews(path):
    data = []
    files = [f for f in os.listdir(path)]
    for file in files:
        with open(path+file, 'r', encoding='utf-8') as f:
            data.append(f.read())

    return data


train_pos = pd.DataFrame({
    'review': get_reviews('data/train/pos/'),
    'label': 1
})

train_neg = pd.DataFrame({
    'review': get_reviews('data/train/neg/'),
    'label': 0
})

test_pos = pd.DataFrame({
    'review': get_reviews('data/test/pos/'),
    'label': 1
})

test_neg = pd.DataFrame({
    'review': get_reviews('data/test/neg/'),
    'label': 0
})

train = pd.concat([train_pos, train_neg], ignore_index=True).sample(frac=1)
test = pd.concat([test_pos, test_neg], ignore_index=True).sample(frac=1)


def clean_word(text):
    """ data clean

    :param text:
    :return:
    """
    # reversion word
    replacement_patterns = [
        (r'won\'t', 'will not'),
        (r'can\'t', 'cannot'),
        (r'i\'m', 'i am'),
        (r'ain\'t', 'is not'),
        (r'(\w+)\'ll', '\g<1> will'),
        (r'(\w+)n\'t', '\g<1> not'),
        (r'(\w+)\'ve', '\g<1> have'),
        (r'(\w+)\'s', '\g<1> is'),
        (r'(\w+)\'re', '\g<1> are'),
        (r'(\w+)\'d', '\g<1> would')]

    for abb_word, original in replacement_patterns:
        text = re.sub(re.compile(abb_word), original, text)

    text = re.sub(re.compile('<.*?>'), '', text)  # clean html
    text = re.sub('[^a-zA-Z]', ' ', text)  # save word only

    text = text.lower()

    return text


train_text = [clean_word(i) for i in train['review']]
test_text = [clean_word(i) for i in test['review']]

vocabulary = 10000  # vocabulary number
embedding_dim = 100  # embedding dimension = word vector dimension
word_num = 200  # max length of sequence
epochs = 4

# word tokenizer
tokenizer = Tokenizer(num_words=vocabulary)
tokenizer.fit_on_texts(train_text)

# get word_index as: ['mero': 46311, 'vachon': 46312, 'meditteranean': 46313, 'thuggees': 46314, ......]
word_index = tokenizer.word_index

# get word_sequence as: [[[39, 282, 10, 21, 252, 267, 203, 271, 1, 1200, .......],......]
sequences_train = tokenizer.texts_to_sequences(train_text)

# reshape word_sequence as: [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    9,
#          13,   52,    9,   84,  209,   10,    8,   39, 2101, 4638,    2,
#         797,  307,   93,  139,    1,   19,   18,    1,    5,   25, 3102,
#           3, 4639,    2,   12, 1518,    9,   72,   12,   52,  401,    8,
#        1182, 2330,    8,    7,    2,   82, 2216,    4,   61,   42, 1827,
#           3, 5038,   52,    9,   13,    8, 1182, 2330,    3,  535,    1,
#         290,   29,    4,  804,    8, 1907,   59, 1114, 3354,   21,   91,
#         606,   37,    2,    4,  168,    6,  820,   35,   10,   61,   15,
#          28, 4233]
x_train = preprocessing.sequence.pad_sequences(sequences_train, maxlen=word_num)
y_train = [i for i in train['label']]

model = Sequential()
model.add(Embedding(vocabulary, embedding_dim, input_length=word_num))
model.add(LSTM(64, return_state=False, dropout=0.5 ))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

history = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=64, validation_split=0.3, callbacks=[earlystopping],)
model.save('model/lstm.h5')

tokenizer.fit_on_texts(test_text)

word_index = tokenizer.word_index
sequences_train = tokenizer.texts_to_sequences(test_text)
x_test = preprocessing.sequence.pad_sequences(sequences_train, maxlen=word_num)
y_test = [i for i in train['label']]

result = model.evaluate(x=x_test, y=y_test, batch_size=64)

# model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 200, 100)          1000000
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 64)                42240
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 65
# =================================================================
# Total params: 1,042,305
# Trainable params: 1,042,305
# Non-trainable params: 0
# _________________________________________________________________
# Embedding layer: Param = word vector * vocabulary = 100 * 10000 , 200  = input_length
# LSTM layer: Param = 4 * (64 * (100 + 64) + 64)
#             Param = 4 * (shape h * (shape h + shape x) + bias shape)
#             shape h = LSTM state dimension
#             shape x = word vector shape
#             bias shape = shape h


