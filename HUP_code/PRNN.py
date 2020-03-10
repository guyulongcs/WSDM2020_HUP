import random as rn
import numpy as np
rn.seed(100)
np.random.seed(200)

import keras

from keras import backend as K

from keras.models import Model
from keras.layers import Embedding, Input, Softmax
from keras.layers import merge, Reshape, LSTM, SimpleRNN, GRU, MaxPooling1D
from keras.layers import Masking, Dense, TimeDistributed, \
    RepeatVector, Permute, Flatten, Activation, Lambda
from keras.layers.merge import Concatenate
from keras.regularizers import l2
from keras.preprocessing import sequence
from keras.optimizers import SGD
from BLSTM import BLSTM

import pandas as pd

import theano

theano.config.openmp = True
from theano import tensor as T


ATT_NET_HIDSIZE=20

class PRNN():
    @classmethod
    def create_model_hierarchy(cls, bottom_item_list, emb_wgts_bottom_items_dict, layer_nums=3, rnn_state_size=[],
                               bottom_emb_item_len=3, flag_embedding_trainable=1, seq_len=39, batch_size=20,
                               mode_attention=1, drop_out_r=0., att_layer_cnt=2, bhDwellAtt=0, rnn_type="WGRU", RNN_norm="GRU", flagCid3RNNNorm=False):
        c_mask_value = 0.
        att_zero_value = -2 ^ 31

        def slice(x):
            return x[:, -1, :]

        flag_concate_sku_cid = True
        RNN = rnn_type
        MODE_BHDWELLATT = True if bhDwellAtt == 1 else False

        ATT_NET_LAYER_CNT = att_layer_cnt
        bottom_item_len = len(bottom_item_list)

        input = [None] * bottom_item_len
        word_num = [None] * bottom_item_len
        emb_len = [None] * bottom_item_len
        embedding_bottom_item = [None] * bottom_item_len
        embed = [None] * bottom_item_len

        layer_nums_max = 3

        rnn_embed = [None] * layer_nums_max
        rnn = [None] * layer_nums_max
        rnn_output = [None] * layer_nums_max

        flag_embedding_trainable = True if flag_embedding_trainable == 1 else False

        ##Embedding layer
        # Embedding sku, bh, cid3, gap, dwell: 0, 1, 2, 3, 4

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

            cur_flag_embedding_trainable = flag_embedding_trainable
            if (i == 0):
                cur_flag_embedding_trainable = False
            embedding_bottom_item[i] = Embedding(word_num[i], emb_len[i], input_length=seq_len,
                                                 trainable=cur_flag_embedding_trainable)
            embed[i] = embedding_bottom_item[i](input[i])  # drop_out=0.2
            embedding_bottom_item[i].set_weights([emb_wgts_bottom_items_dict[bottom_item]])

        # cal mask
        mask_sku = np.zeros((batch_size, seq_len))
        mask_cid3 = np.zeros((batch_size, seq_len))


        for j in range(batch_size):
            sku = input[0][j, :]
            cid3 = input[2][j, :]

            for k in range(seq_len - 1):
                if (sku[k] == 0 or sku[k] == sku[k + 1]):
                    mask_sku[j][k] = 1
                if (sku[k] == 0 or cid3[k] == cid3[k + 1]):
                    mask_cid3[j][k] = 1

        # f mask
        def f_mask_sku(x):
            x_new = x
            for j in range(batch_size):
                for k in range(seq_len):
                    if (mask_sku[j][k] == 1):
                        x_new = T.set_subtensor(x_new[j, k, :], c_mask_value)
            return x_new

        def f_mask_cid3(x):
            x_new = x
            for j in range(batch_size):
                for k in range(seq_len):
                    if (mask_cid3[j][k] == 1):
                        x_new = T.set_subtensor(x_new[j, k, :], c_mask_value)
            return x_new

        def f_mask_att_sku(x):
            x_new = x
            for j in range(batch_size):
                for k in range(seq_len):
                    if (mask_sku[j][k] == 1):
                        x_new = T.set_subtensor(x_new[j, k], att_zero_value)
            return x_new

        def f_mask_att_cid3(x):
            x_new = x
            for j in range(batch_size):
                for k in range(seq_len):
                    if (mask_cid3[j][k] == 1):
                        x_new = T.set_subtensor(x_new[j, k], att_zero_value)
            return x_new

        def K_dot(arr):
            axes = [1, 1]
            x, y = arr[0], arr[1]
            return K.batch_dot(x, y, axes=axes)

        def K_squeeze(x):
            return K.squeeze(x, axis=-1)

        Lambda_sequeeze = Lambda(lambda x: K_squeeze(x))
        ##RNN layer
        if (RNN == "BLSTM"):
            rnn[0] = BLSTM(rnn_state_size[0], interval_dim=emb_len[3], weight_dim=emb_len[1], stateful=False, return_sequences=True,
                          dropout=drop_out_r,
                          name="rnn_out_micro")
            rnn[1] = BLSTM(rnn_state_size[1], interval_dim=emb_len[3], weight_dim=emb_len[4], stateful=False, return_sequences=True,
                          dropout=drop_out_r,
                          name="rnn_out_sku")
            if(not flagCid3RNNNorm):
                rnn[2] = BLSTM(rnn_state_size[2], interval_dim=emb_len[3], weight_dim=0, stateful=False, return_sequences=True,
                          dropout=drop_out_r,
                          name="rnn_out_cid3")

        elif (RNN == "BLSTM2"):
            rnn[0] = BLSTM2(rnn_state_size[0], interval_dim=emb_len[3], weight_dim=emb_len[1], stateful=False, return_sequences=True,
                          dropout=drop_out_r,
                          name="rnn_out_micro")
            rnn[1] = BLSTM2(rnn_state_size[1], interval_dim=emb_len[3], weight_dim=emb_len[4], stateful=False, return_sequences=True,
                          dropout=drop_out_r,
                          name="rnn_out_sku")
            if (not flagCid3RNNNorm):
                rnn[2] = BLSTM2(rnn_state_size[2], interval_dim=emb_len[3], weight_dim=0, stateful=False,
                                   return_sequences=True,
                                   dropout=drop_out_r,
                                   name="rnn_out_cid3")

        elif (RNN == "TimeLSTM"):
            rnn[0] = BLSTM(rnn_state_size[0], interval_dim=emb_len[3], weight_dim=0, stateful=False, return_sequences=True,
                          dropout=drop_out_r,
                          name="rnn_out_micro")
            rnn[1] = BLSTM(rnn_state_size[1], interval_dim=emb_len[3], weight_dim=0, stateful=False,
                               return_sequences=True,
                               dropout=drop_out_r,
                               name="rnn_out_sku")
            if (not flagCid3RNNNorm):
                rnn[2] = BLSTM(rnn_state_size[2], interval_dim=emb_len[3], weight_dim=0, stateful=False,
                                   return_sequences=True,
                                   dropout=drop_out_r,
                                   name="rnn_out_cid3")
        elif (RNN == "WGRU"):
            rnn[0] = WGRU(rnn_state_size[0], weight_dim=emb_len[1], stateful=False, return_sequences=True,
                          dropout=drop_out_r,
                          name="rnn_out_micro")
            rnn[1] = WGRU(rnn_state_size[1], weight_dim=emb_len[3], stateful=False, return_sequences=True,
                          dropout=drop_out_r,
                          name="rnn_out_sku")
            if (not flagCid3RNNNorm):
                rnn[2] = WGRU(rnn_state_size[2], weight_dim=emb_len[3], tateful=False, return_sequences=True,
                            dropout=drop_out_r,
                            name="rnn_out_cid3")
        elif (RNN == "LSTM" or RNN == "GRU"):
            RNN = LSTM if RNN == "LSTM" else GRU
            rnn[0] = RNN(rnn_state_size[0], stateful=False, return_sequences=True, dropout=drop_out_r, name="rnn_out_micro")
            rnn[1] = RNN(rnn_state_size[1], stateful=False, return_sequences=True, dropout=drop_out_r, name="rnn_out_sku")
        else:
            print "%s is not valid RNN!" % RNN

        if(RNN_norm == "LSTM"):
            rnn_cid3 = LSTM
        else:
            rnn_cid3 = GRU
        if(flagCid3RNNNorm):
            rnn[2] = rnn_cid3(rnn_state_size[2], stateful=False, return_sequences=True, dropout=drop_out_r,
                          name="rnn_out_cid3")

        #rnn embed 0
        if (bottom_emb_item_len == 5):
            rnn_embed[0] = Concatenate(axis=-1)([embed[0], embed[1], embed[2], embed[3], embed[4]])
        elif (bottom_emb_item_len == 4):
            rnn_embed[0] = Concatenate(axis=-1)([embed[0], embed[1], embed[2], embed[3]])
        elif (bottom_emb_item_len == 3):
            rnn_embed[0] = Concatenate(axis=-1)([embed[0], embed[1], embed[2]])
        elif (bottom_emb_item_len == 1):
            rnn_embed[0] = embed[0]
        elif (bottom_emb_item_len == 2):
            rnn_embed[0] = Concatenate(axis=-1)([embed[0], embed[2]])
        else:
            rnn_embed[0] = Concatenate(axis=-1)([embed[0], embed[1], embed[2], embed[3]])

        #add interval, wei
        if (RNN == "WGRU"):
            rnn_embed[0] = Concatenate(axis=-1)([rnn_embed[0], embed[1]])

        if (RNN == "BLSTM" or RNN == "BLSTM2"):
            rnn_embed[0] = Concatenate(axis=-1)([rnn_embed[0], embed[3], embed[1]])

        if (RNN == "TimeLSTM"):
            rnn_embed[0] = Concatenate(axis=-1)([rnn_embed[0], embed[3]])

        #rnn micro
        rnn_output[0] = rnn[0](rnn_embed[0])

        # rnn sku
        if (flag_concate_sku_cid):
            rnn_embed[1] = Concatenate(axis=-1)([embed[0], rnn_output[0]])
        else:
            rnn_embed[1] = rnn_output[0]

        # mask sku
        #rnn embed 1
        # rnn_embed[1] = Lambda(f_mask_sku, output_shape=(seq_len, rnn_state_size[1]))(rnn_embed[1])

        if (RNN == "WGRU"):
            rnn_embed[1] = Concatenate(axis=-1)([rnn_embed[1], embed[4]])

        if (RNN == "BLSTM" or RNN == "BLSTM2"):
            rnn_embed[1] = Concatenate(axis=-1)([rnn_embed[1], embed[3], embed[4]])

        if (RNN == "TimeLSTM"):
            rnn_embed[1] = Concatenate(axis=-1)([rnn_embed[1], embed[3]])

        rnn_embed[1] = Lambda(f_mask_sku)(rnn_embed[1])
        rnn_embed[1] = Masking(mask_value=c_mask_value)(rnn_embed[1])

        rnn_output[1] = rnn[1](rnn_embed[1])

        # rnn cid3
        if (flag_concate_sku_cid):
            rnn_embed[2] = Concatenate()([embed[2], rnn_output[1]])
        else:
            rnn_embed[2] = rnn_output[1]

        if(not flagCid3RNNNorm):
            rnn_embed[2] = Concatenate(axis=-1)([rnn_embed[2], embed[3]])

        # mask cid3
        # rnn_embed[2] = Lambda(f_mask_cid3, output_shape=(seq_len, rnn_state_size[2]))(rnn_embed[2])
        rnn_embed[2] = Lambda(f_mask_cid3)(rnn_embed[2])
        rnn_embed[2] = Masking(mask_value=c_mask_value)(rnn_embed[2])
        rnn_output[2] = rnn[2](rnn_embed[2])

        # rnn final output
        rnn_out_final = rnn_output[layer_nums - 1]

        rnn_out_micro = rnn_output[0]
        rnn_out_sku = rnn_output[1]
        rnn_out_cid3 = rnn_output[2]

        # predict sku, cid3
        if (mode_attention == 0):
            # micro
            att_out_micro = Lambda(slice, output_shape=(rnn_state_size[0],))(rnn_out_micro)
            # trans to sku emb len
            out_micro_sku_emb = Dense(emb_len[0], activation="tanh")(att_out_micro)
            out_micro = out_micro_sku_emb

            # sku
            att_out_sku = Lambda(slice, output_shape=(rnn_state_size[1],))(rnn_out_sku)
            # trans to sku emb len
            out_sku_emb = Dense(emb_len[0], activation="tanh")(att_out_sku)
            out_sku = out_sku_emb

            # cid3
            att_out_cid3 = Lambda(slice, output_shape=(rnn_state_size[2],))(rnn_out_cid3)
            out_cid3_emb = Dense(emb_len[2], activation="tanh")(att_out_cid3)
            out_cid3 = out_cid3_emb
            # out_cid3 = Dense(word_num[2], activation="softmax")(out_cid3_emb)

        if (mode_attention == 2):
            # atten micro
            m_h = rnn_out_micro
            m_h_last = Lambda(slice, output_shape=(rnn_state_size[0],), name="rnn_out_micro_last")(m_h)
            m_h_r = RepeatVector(seq_len)(m_h_last)
            if(MODE_BHDWELLATT):
                m_h_c = Concatenate(axis=-1)([m_h, m_h_r, embed[1]])
            else:
                m_h_c = Concatenate(axis=-1)([m_h, m_h_r])
            if(ATT_NET_LAYER_CNT == 2):
                m_h_a_1 = TimeDistributed(Dense(ATT_NET_HIDSIZE, activation='tanh'))(m_h_c)
                m_h_a = TimeDistributed(Dense(1, activation='tanh'))(m_h_a_1)
            else:
                m_h_a = TimeDistributed(Dense(1, activation='tanh'))(m_h_c)
            m_h_a = Lambda(lambda x: x, output_shape=lambda s: s)(m_h_a)
            m_att = Flatten()(m_h_a)

            m_att_micro = Softmax(name="att_micro")(m_att)
            m_att_out = Lambda(K_dot, output_shape=(rnn_state_size[0],), name="out_micro_pre")([m_h, m_att_micro])

            # trans to sku emb len
            out_micro = Dense(emb_len[0], activation="tanh")(m_att_out)

            # attenion sku
            s_h = rnn_out_sku
            s_h_last = Lambda(slice, output_shape=(rnn_state_size[1],), name="rnn_out_sku_last")(s_h)
            s_h_r = RepeatVector(seq_len)(s_h_last)
            if (MODE_BHDWELLATT):
                s_h_c = Concatenate(axis=-1)([s_h, s_h_r, embed[4]])
            else:
                s_h_c = Concatenate(axis=-1)([s_h, s_h_r])
            if(ATT_NET_LAYER_CNT == 2):
                s_h_a_1 = TimeDistributed(Dense(ATT_NET_HIDSIZE, activation='tanh'))(s_h_c)
                s_h_a = TimeDistributed(Dense(1, activation='tanh'))(s_h_a_1)
            else:
                s_h_a = TimeDistributed(Dense(1, activation='tanh'))(s_h_c)
            s_h_a = Lambda(lambda x: x, output_shape=lambda s: s)(s_h_a)
            s_att = Flatten()(s_h_a)
            s_att = Lambda(f_mask_att_sku)(s_att)
            s_att_sku = Softmax(axis=-1, name="att_sku")(s_att)
            s_att_out = Lambda(K_dot, output_shape=(rnn_state_size[1],), name="out_sku_pre")([s_h, s_att_sku])

            # attention cid3
            c_h = rnn_out_cid3
            c_h_last = Lambda(slice, output_shape=(rnn_state_size[2],), name="rnn_out_cid3_last")(c_h)
            c_h_r = RepeatVector(seq_len)(c_h_last)
            c_h_c = Concatenate(axis=-1)([c_h, c_h_r])
            if (ATT_NET_LAYER_CNT == 2):
                c_h_a_1 = TimeDistributed(Dense(ATT_NET_HIDSIZE, activation='tanh'))(c_h_c)
                c_h_a = TimeDistributed(Dense(1, activation='tanh'))(c_h_a_1)
            else:
                c_h_a = TimeDistributed(Dense(1, activation='tanh'))(c_h_c)
            c_h_a = Lambda(lambda x: x, output_shape=lambda s: s)(c_h_a)
            c_att = Flatten()(c_h_a)
            c_att = Lambda(f_mask_att_cid3)(c_att)
            c_att_cid3 = Softmax(axis=-1, name="att_cid3")(c_att)
            c_att_out = Lambda(K_dot, output_shape=(rnn_state_size[2],), name="out_cid3_pre")([c_h, c_att_cid3])

            out_cid3 = Dense(emb_len[2], activation="tanh")(c_att_out)
            out_sku = Dense(emb_len[0], activation="tanh")(s_att_out)

        # model
        model = Model(inputs=[input[0], input[1], input[2], input[3], input[4]], outputs=[out_micro, out_sku, out_cid3])

        # return embedding, rnn, ret_with_target, input, out
        return model
