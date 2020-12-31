# -*- coding:utf-8 -*-
# @Time: 2020/12/31 16:10
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: Seq2seq_Attenton.py

import os
import pandas as pd
from Seq2Seq import Seq2Seq
from utils import get_data
from predict import predict
from keras.layers import Dense, LSTM, Input
from keras.models import Model, load_model
from keras.layers import Activation, dot, concatenate
from keras.utils import plot_model

class Seq2seqAttention(Seq2Seq):

    def __init__(self):
        super().__init__()

    def build(self, model_predict=False):
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

        # Luong's Attention
        attention = dot([encoder_state, decoder_output], axes=[2, 2])
        attention = Activation('softmax', name='attention')(attention)
        context = dot([attention, encoder_state], name='context', axes=[2, 1])
        decoder_combined_context = concatenate([context, decoder_output], name='decoder_combined_context')

        decoder_dense = Dense(self.decoder_output_shape, activation='softmax')
        decoder_output = decoder_dense(decoder_combined_context)

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

            # Luong's Attention
            attention = dot([decoder_input, decoder_infer_output ], axes=(2, 2))
            attention = Activation('softmax', name='attention')(attention)
            context = dot([attention, decoder_input], axes=[2, 1], name='context_vector')
            decoder_combined_context = concatenate([context, decoder_infer_output],
                                                   name='decoder_combined_context_vector')

            decoder_infer_output = decoder_dense(decoder_combined_context)  # output

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

    if not os.path.exists('model/model_train_attention.h5'):
        seq2seq = Seq2Seq(encoder_input_shape, decoder_output_shape, n_units)

    else:
        seq2seq = Seq2Seq(encoder_input_shape, decoder_output_shape, n_units, pretrained_path='model/model_train_attention.h5')

    train_model = seq2seq.build()

    plot_model(to_file='img/seq2seq-attention-model.png', model=train_model, show_shapes=True)

    # train_model.fit([encoder_data, decoder_data], target_data, batch_size=batch_size, epochs=epochs,
    #                 validation_split=0.2)
    #
    # train_model.save("model/model_train_attention.h5")
    # seq2seq = Seq2Seq(encoder_input_shape, decoder_output_shape, n_units, pretrained_path='model/model_train_attention.h5')
    # train_model, encoder_infer, decoder_infer = seq2seq.build(model_predict=True)
    # data = pd.read_table(file_path, header=None)
    # data.columns = ['en', 'zh', 's']
    # test_list = data['en'][300:320]
    #
    # for text in test_list:
    #     out = predict(
    #         text,
    #         encoder_text_max_length,
    #         encoder_input_shape,
    #         encoder_text_index,
    #         decoder_text_index,
    #         encoder_infer,
    #         decoder_infer,
    #         decoder_text_max_length,
    #         decoder_output_shape)
    #     print(text, out)