# -*- coding:utf-8 -*-
# @Time: 2020/12/30 12:44
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: utils.py


import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical


def get_data(file_path, samples):
    """data process

    :param file_path:
    :param samples:
    :return:
    """
    text = pd.read_table(file_path, header=None)

    text.columns = ['en', 'zh', 'other']

    en_text = text['en'][:samples]
    zh_text = text['zh'][:samples]

    encoder_text = [i for i in en_text]
    # use \t as start sign, \n as end sign
    decoder_text = ['\t' + i + '\n' for i in zh_text]

    def text2sequences(max_len, lines):
        """split sequence by char

        :param max_len:
        :param lines:
        :return:
        """
        tokenizer = Tokenizer(
            char_level=True,
            filters='',
            lower=False)  # lower en word
        tokenizer.fit_on_texts(lines)
        seqs = tokenizer.texts_to_sequences(lines)
        seqs_pad = pad_sequences(seqs, maxlen=max_len, padding='post')
        # padding='pre' or 'post': padding with 0 at the front or back end of the sequence.
        # value='float': use to padding
        return seqs_pad, tokenizer.word_index

    encoder_text_max_length = max([len(i) for i in encoder_text])
    decoder_text_max_length = max([len(i) for i in decoder_text])

    encoder_text, encoder_text_index = text2sequences(
        encoder_text_max_length, encoder_text)
    decoder_text, decoder_text_index = text2sequences(
        decoder_text_max_length, decoder_text)

    encoder_classes = len(encoder_text_index)
    decoder_classes = len(decoder_text_index)

    encoder_data = np.zeros(
        (encoder_text.shape[0],
         encoder_text_max_length,
         encoder_classes))
    decoder_data = np.zeros(
        (decoder_text.shape[0],
         decoder_text_max_length,
         decoder_classes))
    target_data = np.zeros(
        (decoder_text.shape[0],
         decoder_text_max_length,
         decoder_classes))

    # # one hot encode target sequence
    # this way will cause memory error, this func 'to_cacategorical()' is the main cuase
    # def onehot_encode(sequences, max_len, vocab_size):
    #     n = len(sequences)
    #     data = np.zeros((n, max_len, vocab_size))
    #     for i in range(n):
    #         data[i, :, :] = to_categorical(sequences[i], num_classes=vocab_size)
    #     return data

    for seq_index, seq in enumerate(encoder_text):
        for char_index, char in enumerate(seq):
            if char != 0:
                encoder_data[seq_index, char_index, char - 1] = 1.0  # char - 1

    for seq_index, seq in enumerate(decoder_text):
        for char_index, char in enumerate(seq):
            if char != 0:
                decoder_data[seq_index, char_index, char - 1] = 1.0  # char - 1
            if char_index > 0:  # remove '\t'
                target_data[seq_index, char_index - 1, char - 1] = 1.0

    return encoder_data, decoder_data, target_data, encoder_classes, decoder_classes, encoder_text_index, decoder_text_index, encoder_text_max_length, decoder_text_max_length