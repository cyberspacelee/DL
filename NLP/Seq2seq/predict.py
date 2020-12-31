# -*- coding:utf-8 -*-
# @Time: 2020/12/31 15:55
# @Author: explorespace
# @Email: cyberspacecloner@qq.com
# @File: predict.py


from Seq2Seq import Seq2Seq
from utils import get_data
import numpy as np
import pandas as pd

file_path = 'data/cmn.txt'
samples = 1000
encoder_data, decoder_data, target_data, encoder_input_shape, decoder_output_shape, encoder_text_index, \
    decoder_text_index, encoder_text_max_length, decoder_text_max_length = get_data(file_path, samples)

def predict(text, encoder_text_max_length, encoder_input_shape, encoder_text_index,
            decoder_text_index, encoder_infer, decoder_infer, decoder_text_max_length, decoder_output_shape):
    """predict function

    :param text:
    :param encoder_text_max_length:
    :param encoder_input_shape:
    :param encoder_text_index:
    :param decoder_text_index:
    :param encoder_infer:
    :param decoder_infer:
    :param decoder_text_max_length:
    :param decoder_output_shape:
    :return:
    """
    encode_input = np.zeros((1, encoder_text_max_length, encoder_input_shape))
    reverse_decoder = {value: key for key, value in decoder_text_index.items()}
    for char_index, char in enumerate(text):
        # encoder_text_index[char] - 1
        encode_input[0, char_index, encoder_text_index[char] - 1] = 1

    state = encoder_infer.predict(encode_input)
    # '\t' as start sign
    predict_seq = np.zeros((1, 1, decoder_output_shape))
    predict_seq[0, 0, decoder_text_index['\t'] - 1] = 1

    output = ''
    # 开始对encoder获得的隐状态进行推理
    # 每次循环用上次预测的字符作为输入来预测下一次的字符，直到预测出了终止符
    for i in range(decoder_text_max_length):  # decoder_text_max_length为句子最大长度
        # 给decoder输入上一个时刻的h,c隐状态，以及上一次的预测字符predict_seq
        yhat, h, c = decoder_infer.predict([predict_seq] + state)
        # 注意，这里的yhat为Dense之后输出的结果，因此与h不同
        char_index = np.argmax(yhat[0, -1, :])
        char = reverse_decoder[char_index + 1]
        output += char
        state = [h, c]  # 本次状态做为下一次的初始状态继续传递
        predict_seq = np.zeros((1, 1, decoder_output_shape))
        predict_seq[0, 0, char_index] = 1
        if char == '\n':  # 预测到了终止符则停下来
            break
    return output



if __name__ == '__main__':
    n_units = 64
    seq2seq = Seq2Seq(encoder_input_shape, decoder_output_shape, n_units,pretrained_path='model/model_train.h5')
    train_model, encoder_infer, decoder_infer = seq2seq.build(model_predict=True)

    data = pd.read_table(file_path, header=None)
    data.columns=['en', 'zh', 's']
    test_list = data['en'][300:320]

    for text in test_list:
        out = predict(
            text,
            encoder_text_max_length,
            encoder_input_shape,
            encoder_text_index,
            decoder_text_index,
            encoder_infer,
            decoder_infer,
            decoder_text_max_length,
            decoder_output_shape)
        print(text, out)