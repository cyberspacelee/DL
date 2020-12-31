# -*- coding:utf-8 -*-
# @Time: 2020/12/29 18:37
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: Seq2Seq.py

import re
import os
import numpy as np
import pandas as pd
import string
from utils import get_data
from keras import preprocessing, optimizers
from keras.layers import Dense, LSTM, Input
from keras.models import Model, load_model
from keras.utils import plot_model


class Seq2Seq(object):

    def __init__(self, encoder_input_shape, decoder_output_shape, n_units, pretrained_path=None):
        """init model

        :param pretrained_path:
        :param encoder_input_shape: input shape
        :param decoder_output_shape: decoder_shape
        :param n_units: LSTM params
        """
        self.encoder_input_shape = encoder_input_shape
        self.decoder_output_shape = decoder_output_shape
        self.n_units = n_units
        self.pretrained_path = pretrained_path

    def build(self, model_predict=False):
        """

        :param predict:
        :return:
        """

        # Encoder
        encoder_input = Input(shape=(None, self.encoder_input_shape))
        # instantiate a Keras tensor shape
        encoder = LSTM(self.n_units, return_state=True)
        """
        input: 3D tensor with dimensions (batch_size, timesteps, input_dim).
        return_state=True: return(last hidden state, last hidden state, last cell state)
        """
        _, encoder_h, encoder_c = encoder(encoder_input)
        encoder_state = [encoder_h, encoder_c]

        # Decoder
        decoder_input = Input(shape=(None, self.decoder_output_shape))
        decoder = LSTM(self.n_units, return_sequences=True, return_state=True)
        # return_state=True, return_sequences=True: return(all hidden state,
        # last hidden state, last cell state)
        decoder_output, _, _ = decoder(
            decoder_input, initial_state=encoder_state)
        decoder_dense = Dense(self.decoder_output_shape, activation='softmax')
        decoder_output = decoder_dense(decoder_output)

        self.train_model = Model([encoder_input, decoder_input], decoder_output)
        if self.pretrained_path:
            self.train_model.load_weights(self.pretrained_path)
            print('Model Loaded from {}'.format(self.pretrained_path))

        self.train_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.train_model.summary()

        if model_predict:
            # predict decoder
            # (input, output) 预测值进行编码，返回 last state_h,
            self.encoder_infer = Model(encoder_input, encoder_state)
            decoder_state_input_h = Input(shape=(self.n_units,))
            decoder_state_input_c = Input(shape=(self.n_units,))
            decoder_state_input = [
                decoder_state_input_h,
                decoder_state_input_c]

            decoder_infer_output, decoder_infer_state_h, decoder_infer_state_c = decoder(decoder_input,
                                                                                         initial_state=decoder_state_input)
            decoder_infer_state = [
                decoder_infer_state_h,
                decoder_infer_state_c]  # infer state
            decoder_infer_output = decoder_dense(decoder_infer_output)  # output

            self.decoder_infer = Model(
                [decoder_input] +
                decoder_state_input,
                [decoder_infer_output] +
                decoder_infer_state)

            return self.train_model, self.encoder_infer, self.decoder_infer

        else:
            return self.train_model





if __name__ == '__main__':
    n_units = 64
    batch_size = 64
    epochs = 150
    file_path = 'data/cmn.txt'
    samples = 1000

    encoder_data, decoder_data, target_data, encoder_input_shape, decoder_output_shape, encoder_text_index, \
    decoder_text_index, encoder_text_max_length, decoder_text_max_length = get_data(file_path, samples)

    if not os.path.exists('model/model_train.h5'):
        seq2seq = Seq2Seq(encoder_input_shape, decoder_output_shape, n_units)




        # print(model_train.summary())
        # print(encoder_infer.summary())
        # print(decoder_infer.summary())
    else:
        seq2seq = Seq2Seq(encoder_input_shape, decoder_output_shape, n_units,pretrained_path='model/model_train.h5')


    train_model = seq2seq.build()
    plot_model(to_file='img/seq2seq.png', model=train_model, show_shapes=True)

    # model_train.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # train_model.fit([encoder_data, decoder_data], target_data, batch_size=batch_size, epochs=epochs,
    #                     validation_split=0.2)
    #
    # train_model.save("model/model_train.h5")


