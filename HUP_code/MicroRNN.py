import keras

from keras import backend as K

from keras.models import Model
from keras.layers import Embedding, Input
from keras.layers import merge, Reshape, LSTM, SimpleRNN, GRU, Masking, Dense, TimeDistributed, MaxPooling1D, \
    RepeatVector, Permute, Flatten, Activation
from keras.layers.merge import Concatenate
from keras.regularizers import l2
from keras.preprocessing import sequence
from keras.optimizers import SGD

import pandas as pd
import numpy as np
from numpy import *
from numpy import random

import theano

theano.config.openmp = True
from theano import tensor as T

class MicroRNN():
    @classmethod
    def create_model_hierarchy(cls, bottom_item_list, emb_wgts_bottom_items_dict, layer_nums=1, rnn_state_size=[], bottom_emb_item_len=3, flag_embedding_trainable=1, seq_len=39, batch_size=20, rnn_type=GRU):
        RNN = rnn_type

        bottom_item_len = len(bottom_item_list)

        input = [None] * bottom_item_len
        word_num = [None] * bottom_item_len
        emb_len = [None] * bottom_item_len
        embedding_bottom_item = [None] * bottom_item_len
        embed = [None] * bottom_item_len

        layer_nums_max = 1

        rnn_embed = [None] * layer_nums_max
        rnn = [None] * layer_nums_max
        rnn_output = [None] * layer_nums_max

        flag_embedding_trainable = True if flag_embedding_trainable == 1 else False

        ##Embedding layer

        # Embedding sku, bh, cid3, dwell: 0, 1, 2, 3


        for i in range(bottom_item_len):
            bottom_item = bottom_item_list[i]
            ###input
            input[i] = Input(batch_shape=(batch_size, seq_len,), dtype='int32')


            ###Embedding

            # load embedding weights

            # emb_wgts[i] = np.loadtxt(init_wgts_file_emb[i])
            word_num[i], emb_len[i] = emb_wgts_bottom_items_dict[bottom_item].shape
            print word_num[i], emb_len[i]
            # get embedding

            embedding_bottom_item[i] = Embedding(word_num[i], emb_len[i], input_length=seq_len, trainable=flag_embedding_trainable)
            embed[i] = embedding_bottom_item[i](input[i])  # drop_out=0.2
            embedding_bottom_item[i].set_weights([emb_wgts_bottom_items_dict[bottom_item]])


        ##RNN layer

        # rnn micro
        rnn[0] = RNN(rnn_state_size[0], stateful=False, return_sequences=True)


        rnn_embed[0] = Concatenate()([embed[0], embed[1], embed[3]])

        rnn_output[0] = rnn[0](rnn_embed[0])

        # rnn final output
        rnn_out_final = rnn_output[layer_nums-1]

        ##Attention layer
        # attenion value
        out1 = TimeDistributed(Dense(rnn_state_size[-1], activation='tanh', kernel_regularizer=l2(0.01)))(rnn_out_final)
        att = TimeDistributed(Dense(1, activation='linear'))(out1)
        att = Flatten()(att)
        att = Dense(seq_len, activation="softmax")(att)

        # attention cal out
        outpermute = Permute((2, 1))(rnn_out_final)

        aout = keras.layers.Dot(axes=[2, 1])([outpermute, att])

        # model
        #model = Model(inputs=input, outputs=aout)
        model = Model(inputs=[input[0], input[1], input[3]], outputs=aout)

        # return embedding, rnn, ret_with_target, input, out
        return model
