import random as rn
import numpy as np
rn.seed(100)
np.random.seed(200)


import keras


from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Embedding, Input
from keras.layers import merge, Reshape, LSTM, SimpleRNN, GRU, Masking, Dense, TimeDistributed, MaxPooling1D, \
    RepeatVector, Permute, Flatten, Activation
from keras.layers.merge import Concatenate
from keras.regularizers import l2
from keras.preprocessing import sequence
from keras.optimizers import SGD

import pandas as pd

import theano

theano.config.openmp = True
from theano import tensor as T

import re, sys
import time

from PRNN import *
#from MicroRNN import *

from preprocess.FileTool import *
from preprocess.Config import *
from preprocess.Metrics import *
from preprocess.ProTool import *
from preprocess.MLTool import *
from preprocess.Data import *


DEBUG_MODE=False


class PRNNRec():
    model_tag_list = ["PRNN", "MicroRNN"]
    def __init__(self):
        #data
        self.emb_wgts_micro_items_dict = {}
        #self.emb_wgts_micro_items_list = []
        #model
        self.hie_rnn_model = None

        # config
        self.model_tag = PRNNRec.model_tag_list[0]
        self.drop_out_rate_train_data = 0.05
        self.micro_mode = "SBCGD"
        self.micro_item_list = Config.get_micro_item_list(self.micro_mode)

        #train
        self.layer_num = 0
        self.rnn_state_size = []
        self.bottom_emb_item_len = 0
        self.flag_load_model_weights = False
        self.flag_train = False
        self.train_epoch = 0
        self.flag_test_candidate_all=0


        self.log_info_sku=[]
        self.log_info_cid3 = []


        pass

    def get_flag_from_str(self, str):
        res = True if (str == '1') else False
        return res

    def print_log_info(self):
        print "print_log_info..."

        file_log_info_sku = "log_info_sku_" + os.path.basename(self.model_save_file)
        file_log_info_cid3 = "log_info_cid3_" + os.path.basename(self.model_save_file)
        [file_log_info_sku, file_log_info_cid3] = FileTool.add_folder_fileList(self.folder_output,
                                                                               [file_log_info_sku, file_log_info_cid3])

        FileTool.printList(self.log_info_sku, file_log_info_sku)
        FileTool.printList(self.log_info_cid3, file_log_info_cid3)
        print "print_log_info done!"

    def set_files_config(self, model_tag, layer_num, rnn_state_size, bottom_emb_item_len, flag_embedding_trainable, folder_input, folder_output, exp_sig, init_emb_wgts_bootom_items_path, train_file, test_file, train_len, test_len,
                        seq_len, batch_size, model_save_file, map_file_bottom_id_to_itemsId, top_sku_recall_file, map_file_sku,
                        flag_train, train_epoch, flag_test_candidate_all, mode_attention, drop_out_r, loss_weights, att_layer_cnt, bhDwellAtt, rnn_type, RNN_norm):
        #files
        self.folder = folder_input
        self.folder_output = folder_output
        self.exp_sig = exp_sig

        print "self.micro_mode:", self.micro_mode
        print "self.micro_item_list:", self.micro_item_list
        print "init_emb_wgts_bootom_items_path:", init_emb_wgts_bootom_items_path

        [self.train_file, self.test_file, self.map_file_bottom_id_to_itemsId, self.top_sku_recall_file, self.map_file_sku] = \
            FileTool.add_folder_file_list(self.folder,
                                          [train_file, test_file, map_file_bottom_id_to_itemsId, top_sku_recall_file, map_file_sku])
        [self.model_save_file] = FileTool.add_folder_file_list(self.folder_output, [model_save_file])

        self.init_emb_wgts_bootom_items_path = FileTool.add_folder_file_list(self.folder, init_emb_wgts_bootom_items_path)

        #config
        self.model_tag = model_tag
        [self.layer_num, self.bottom_emb_item_len, self.flag_embedding_trainable, self.train_len, self.test_len, self.seq_len, self.batch_size] = [layer_num, bottom_emb_item_len, flag_embedding_trainable, train_len, test_len, seq_len, batch_size]
        self.rnn_state_size = StructureTool.format_list_to_int(rnn_state_size)
        #train
        #self.flag_load_model_weights = self.get_flag_from_str(flag_load_model_weights)
        self.flag_train = self.get_flag_from_str(flag_train)
        self.train_epoch = int(train_epoch)
        self.flag_test_candidate_all = self.get_flag_from_str(str(flag_test_candidate_all))
        self.mode_attention = int(mode_attention)
        self.drop_out_r = float(drop_out_r)
        self.loss_weights = [float(item) for item in str(loss_weights).split(',')]
        self.att_layer_cnt = int(att_layer_cnt)
        self.bhDwellAtt = int(bhDwellAtt)
        self.rnn_type = rnn_type
        self.RNN_norm = RNN_norm
        print "loss_weights:", self.loss_weights

    def load_data(self):
        FileTool.func_begin("load_data")
        self.load_data_emb_bottom_dict()
        FileTool.func_end("load_data")

    def load_data_emb_bottom_dict(self):
        FileTool.func_begin("load_data_emb_bottom_dict")
        print "self.init_emb_wgts_bootom_items_path:", self.init_emb_wgts_bootom_items_path
        print "self.micro_item_list:", self.micro_item_list
        self.emb_wgts_micro_items_dict = Data.load_micro_item_vec(self.init_emb_wgts_bootom_items_path, self.micro_item_list)
        #self.emb_wgts_micro_items_list = self.get_emb_wgts_micro_items_list()
        FileTool.func_begin("load_data_emb_bottom_dict")
        pass

    def get_emb_wgts_micro_items_list(self):
        emb_wgts_micro_items_list = []
        for item in self.micro_item_list:
            list.append(self.emb_wgts_micro_items_dict[item])
        return emb_wgts_micro_items_list

    def create_model(self):
        print "create model..."

        if(self.model_tag == "PRNN"):
            self.hie_rnn_model = PRNN.create_model_hierarchy(self.micro_item_list, self.emb_wgts_micro_items_dict,
                layer_nums=self.layer_num, rnn_state_size=self.rnn_state_size,
                bottom_emb_item_len=self.bottom_emb_item_len, flag_embedding_trainable=self.flag_embedding_trainable,
                seq_len=self.seq_len, batch_size=self.batch_size, mode_attention=self.mode_attention, drop_out_r=self.drop_out_r, att_layer_cnt=self.att_layer_cnt, bhDwellAtt=self.bhDwellAtt, rnn_type=self.rnn_type, RNN_norm=self.RNN_norm)
        elif(self.model_tag == "MicroRNN"):
            self.hie_rnn_model = MicroRNN.create_model_hierarchy(self.micro_item_list, self.emb_wgts_micro_items_dict,
                layer_nums=self.layer_num, rnn_state_size=self.rnn_state_size,
                bottom_emb_item_len=self.bottom_emb_item_len,flag_embedding_trainable=self.flag_embedding_trainable,
                seq_len=self.seq_len, batch_size=self.batch_size, rnn_type=GRU)
        print "create model done!"


    def train_model(self):
        print "train model start..."
        starttime = time.time()
        n_epoch = self.train_epoch

        nrows = self.train_len

        if(DEBUG_MODE):
            nrows=5000

        [data_micro, total_data_line_cnt] = Data.load_data_trans_bottomId_to_microItemsId(self.train_file,
                                                                                                 nrows,
                                                                                                 self.map_file_bottom_id_to_itemsId,
                                                                                                 self.micro_item_list)

        micro_item_cnt = len(self.micro_item_list)

        info_loss_list = []
        for e in range(n_epoch):
            str_epoch = "epoch/toal_epoch: %d / %d" % (e+1, n_epoch)
            print str_epoch

            train_loss = np.array([0., 0., 0., 0.])

            total_pro_cnt=0

            batch_cnt = total_data_line_cnt / batch_size
            print "train_data_line_cnt:%d, batch_cnt:%d" % (total_data_line_cnt, batch_cnt)

            data = [None] * micro_item_cnt
            x = [None] * micro_item_cnt
            y = [None] * micro_item_cnt

            for batch_id in range(batch_cnt):

                #batch_index = random.randint(0, train_data_line_cnt - batch_size)
                batch_index = batch_id * batch_size

                for i in range(micro_item_cnt):
                    micro_item = self.micro_item_list[i]
                    data[i] = data_micro[i][batch_index:batch_index+batch_size, :]

                    x[i] = np.array(data[i][:, :-1])
                    y[i] = np.array(data[i][:, -1])

                #target
                y_true_sku = y[0]
                y_true_cid3 = y[2]

                y_true_sku_emb = Data.get_embedding(y_true_sku, self.emb_wgts_micro_items_dict["sku"])
                y_true_cid3_emb = Data.get_embedding(y_true_cid3, self.emb_wgts_micro_items_dict["cid3"])

                y_true_sku_emb = np.array(y_true_sku_emb)
                y_true_cid3_emb = np.array(y_true_cid3_emb)

                if(self.model_tag == "PRNN"):
                    x_batch = [x[0], x[1], x[2], x[3], x[4]]
                elif(self.model_tag == "MicroRNN"):
                    x_batch = [x[0], x[1], x[3]]


                train_loss_batch = self.hie_rnn_model.train_on_batch(x_batch, [y_true_sku_emb, y_true_sku_emb, y_true_cid3_emb])

                train_loss_batch = np.array(train_loss_batch)

                train_loss_batch_avg =  train_loss_batch / float(batch_size)

                train_loss += train_loss_batch
                sys.stdout.flush()

                if (batch_id % 100) == 0:
                    str_batch = "batch_id/batch_cnt: %d / %d" % (batch_id+1, batch_cnt)
                    str_loss = "train_loss_batch_avg:%s" % str(train_loss_batch_avg)
                    print ', ' .join([str_epoch, str_batch, str_loss])

                #break

            total_pro_cnt += (batch_cnt * batch_size)
            avg_train_loss = train_loss / float(total_pro_cnt)

            info_loss = "EPOCH LOSS epoch:%d, loss:%s, avg_train_loss:%s" % (e+1, str(train_loss), str(avg_train_loss))
            print info_loss
            info_loss_list.append(info_loss)


        del data_micro

        print "save model:"
        print self.model_save_file

        self.hie_rnn_model.save(self.model_save_file, overwrite=True)

        endtime = time.time()

        FileTool.printList(info_loss_list, "info_loss")

        print "Train model time:"
        ProTool.get_time_interval(starttime, endtime)

        print "train done!"
        pass

    def load_model(self):
        FileTool.func_begin("load_model")
        print self.model_save_file
        self.hie_rnn_model = keras.models.load_model(self.model_save_file)

        FileTool.func_end("load_model")

    def load_map_sku_skuId(self, sku_map_file):
        print "load sku_map_file: %s!" % sku_map_file
        # sku id

        # sku => sku_id
        skumapsku2id = dict()

        # sku_id => sku
        skumapid2sku = dict()

        with open(sku_map_file, 'r') as f:
            for l in f:
                tt = l.strip().split()
                skumapsku2id[int(tt[0])] = int(tt[1])
                skumapid2sku[int(tt[1])] = int(tt[0])

        return (skumapsku2id, skumapid2sku)

    def load_map_trainItemId_sku(self, map_file):
        '''
        id2sku:
            train_item_id  =>  sku
        '''
        id2sku = dict()
        id2sku[0] = '<s>'

        '''
        sku2idlist:
            sku    =>    train_item_id list
        '''
        sku2idlist = dict()

        print "load map_file: %s ! (trainItem)" % map_file
        # [train_item, train_item_id]
        with open(map_file, 'r') as f:
            for l in f:
                tt = l.strip().split()
                # id
                i = int(tt[1])

                # sku
                sku = int(tt[0].split('+')[0])

                id2sku[i] = sku

                if sku in sku2idlist:
                    sku2idlist[sku].append(i)
                else:
                    sku2idlist[sku] = [i]

        return (id2sku, sku2idlist)

    def get_candidate_id_list(self, skuIdList, dict_top_id_recallIdList):
        #FileTool.func_begin("get_candidate_id_list")

        secondLastId = int(skuIdList[-1])

        res = set()
        for sku in skuIdList:
            sku = int(sku)
            if(sku == 0):
                continue
            cur_list = dict_top_id_recallIdList[sku]
            cur_set = set(cur_list)
            res = res.union(cur_set)

        if(secondLastId in res):
            res.remove(secondLastId)

        resList = list(res)
        return resList

        pass

    def get_all_sku_id(self):
        (row, col)  = self.emb_wgts_micro_items_dict["sku"].shape
        res = list(np.arange(1, row))
        return res

    def get_all_cid3_id(self):
        (row, col)  = self.emb_wgts_micro_items_dict["cid3"].shape
        res = list(np.arange(1, row))
        return res

    def get_log_pred(self, pred, y_true):
        flag = 0
        rank = 0
        for (y, score) in pred:
            rank += 1
            if(int(y) == int(y_true)):
                flag=1
                break
        return [flag, rank]

    def get_list_item_dis(self, list, x):
        res = -1
        l = len(list)
        i = 0
        flag = False
        for i in range(l - 1, -1, -1):
            if (int(list[i]) == int(x)):
                flag = True
                break
        if (not flag):
            dis = l
        else:
            dis = l - i - 1
        return dis

    def eval_test_result_example(self, lineNo, skuIdList, xcid3, att_sku, att_cid3, rnn_out_micro_last, dict_top_id_recallIdList, dict_emb, y_pred_emb, y_true_id, y_pred_cid3_emb, y_true_cid3_id, NList, flag_test_candidate_all):
        #cal similarity

        secondLastId = skuIdList[-1]

        if(flag_test_candidate_all):
            candidate_id_list = self.get_candidate_id_list(skuIdList, dict_top_id_recallIdList)
        else:
            candidate_id_list = dict_top_id_recallIdList[secondLastId]

        candidate_id_list_cid3 = self.get_all_cid3_id()

        #sku
        simpred = {}
        for candidate_id in candidate_id_list:
            candidate_emb = dict_emb["sku"][int(candidate_id)]
            sim = np.dot(y_pred_emb, candidate_emb) / (np.linalg.norm(y_pred_emb) * np.linalg.norm(candidate_emb))
            simpred[candidate_id] = sim

        #sorted
        simpred = sorted(simpred.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

        #cid3
        simpred_cid3 = {}
        for candidate_id_cid3 in candidate_id_list_cid3:
            candidate_emb_cid3 = dict_emb["cid3"][int(candidate_id_cid3)]
            sim = np.dot(y_pred_cid3_emb, candidate_emb_cid3) / (np.linalg.norm(y_pred_cid3_emb) * np.linalg.norm(candidate_emb_cid3))
            simpred_cid3[candidate_id_cid3] = sim

        # sorted
        simpred_cid3 = sorted(simpred_cid3.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)


        #eval top N

        #log_info  sku
        list_log_info=[]
        list_log_info.append("lineNo:%s" % str(lineNo))
        list_log_info.append("skuIdList:%s" % StructureTool.format_list_to_str_conn(skuIdList))
        list_log_info.append("secondLastSku:%s" % str(skuIdList[-1]))
        list_log_info.append("simpred:%s" % str(simpred[:20]))
        list_log_info.append("y_true_sku:%s" % str(y_true_id))
        list_log_info.append("att_sku:%s" % StructureTool.format_list_to_str_conn(att_sku))
        [flag, rank] = self.get_log_pred(simpred[:20], y_true_id)
        list_log_info.append(str(self.get_list_item_dis(skuIdList, y_true_id)))
        list_log_info.append(str(flag))
        list_log_info.append(str(rank))
        list_log_info.append("rnn_out_micro_last:%s" % StructureTool.format_list_to_str_conn(rnn_out_micro_last))


        log_info_str = "\t".join(list_log_info)
        self.log_info_sku.append(log_info_str)

        ##log_info  cid3
        list_log_info = []
        list_log_info.append("lineNo:%s" % str(lineNo))

        list_log_info.append("cid3List:%s" % StructureTool.format_list_to_str_conn(xcid3))
        list_log_info.append("secondLastCid3:%s" % str(xcid3[-1]))
        list_log_info.append("simpred_cid3:%s" % str(simpred_cid3[:10]))
        list_log_info.append("y_true_cid3:%s" % str(y_true_cid3_id))
        list_log_info.append("att_cid3:%s" % StructureTool.format_list_to_str_conn(att_cid3))
        [flag, rank] = self.get_log_pred(simpred_cid3[:10], y_true_cid3_id)
        list_log_info.append(str(self.get_list_item_dis(xcid3, y_true_cid3_id)))
        list_log_info.append(str(flag))
        list_log_info.append(str(rank))

        log_info_str = "\t".join(list_log_info)
        self.log_info_cid3.append(log_info_str)


        res_sku = self.get_recall_mrr(simpred, y_true_id, NList)
        res_cid3 = self.get_recall_mrr(simpred_cid3, y_true_cid3_id, NList)

        return [res_sku, res_cid3]


    def get_recall_mrr(self, simpred, y_true_id, NList):
        res = []
        for N in NList:
            pr = set()
            flag = 0
            for (candidate_id, score) in simpred:
                if len(pr) >= N:
                    # print "Not in Predict"
                    break
                if int(candidate_id) == int(y_true_id):
                    flag = 1
                    break
                pr.add(candidate_id)

            rank = len(pr) + 1
            # print "N, flag, rank: %d, %s, %s" % (N, str(flag), str(rank))

            res.append([flag, rank])

        # print "res: ", res

        return res

    def get_model_by_layer_name(self, layer_name):
        return Model(inputs=self.hie_rnn_model.input,
                                         outputs=self.hie_rnn_model.get_layer(layer_name).output)

    def model_output_get_data(self, model_out_layerName_list):
        dict_model = {}
        dict_data = {}
        for name in model_out_layerName_list:
            dict_model[name] = Model(inputs=self.hie_rnn_model.input,
                                            outputs=self.hie_rnn_model.get_layer(name).output)
            dict_data[name] = {}
        return (dict_model, dict_data)

    def model_output_get_dataCur_from_batch(self, dictData, index_in_batch, index_total, model_out_layerName_list):
        for name in model_out_layerName_list:
            self.dict_model_output_data[name][index_total] = dictData[name][index_in_batch]

        pass

    def model_output_preidct_batch(self, model_out_layerName_list, xbatch):
        dict_model_output = {}
        for name in model_out_layerName_list:
            dict_model_output[name] = self.dict_model_output[name].predict_on_batch(xbatch)
        return dict_model_output

    def hasAttention(self):
        return int(self.mode_attention) > 0

    def test_model(self, NList=[1, 2, 5, 8, 10, 20, 50, 100]):
        print "Test model start..."
        starttime = time.time()

        if(self.hasAttention()):
            self.att_sku_layer_model = self.get_model_by_layer_name("att_sku")
            self.att_cid3_layer_model = self.get_model_by_layer_name("att_cid3")

            #rnn output print
            model_out_layerName_list = ["rnn_out_micro", "rnn_out_sku",
                                        "rnn_out_micro_last", "rnn_out_sku_last"]

            (self.dict_model_output, self.dict_model_output_data) = self.model_output_get_data(model_out_layerName_list)



        dict_data_micro_sku_emb = {}
        print "init data start..."

        print "load top_sku_recall_file: %s!" % self.top_sku_recall_file
        dict_top_id_recallIdList = FileTool.load_data_text_np(self.top_sku_recall_file, 'int')


        print "init data done!"

        print "load test_file: %s" % test_file

        nrows = self.test_len

        if(DEBUG_MODE):
            nrows=5000

        [data_micro, total_data_line_cnt] = Data.load_data_trans_bottomId_to_microItemsId(self.test_file, nrows, self.map_file_bottom_id_to_itemsId, self.micro_item_list)
        #[data_micro, total_data_line_cnt] = Data.load_data_trans_bottomId_to_microItemsId_pickle(self.test_file, self.test_len, self.map_file_bottom_id_to_itemsId, self.micro_item_list)

        Ncnt = len(NList)

        recall = {}
        mrr = {}

        recall["sku"] = {}
        mrr["sku"] = {}

        recall["cid3"] = {}
        mrr["cid3"] = {}

        for i in range(Ncnt):
            recall["sku"][NList[i]] = 0
            mrr["sku"][NList[i]] = 0

            recall["cid3"][NList[i]] = 0
            mrr["cid3"][NList[i]] = 0

        allcase = 0

        test_data_line_cnt = total_data_line_cnt
        batch_cnt = (test_data_line_cnt / batch_size)
        print "test_data_line_cnt:%d, batch_cnt:%d" % (test_data_line_cnt, batch_cnt)

        micro_item_cnt = len(self.micro_item_list)

        data = [None] * micro_item_cnt
        x = [None] * micro_item_cnt
        y = [None] * micro_item_cnt

        lineNo=1
        for batch_id in range(batch_cnt):
            batch_index = batch_size * batch_id

            for i in range(micro_item_cnt):
                micro_item = self.micro_item_list[i]
                data[i] = data_micro[i][batch_index:batch_index + batch_size, :]
                x[i] = data[i][:, :-1]
                y[i] = data[i][:, -1]

            if (self.model_tag == "PRNN"):
                x_batch = [x[0], x[1], x[2], x[3], x[4]]
            elif(self.model_tag == "MicroRNN"):
                x_batch = [x[0], x[1], x[3]]

            # predict
            [y_pred_sku_by_micro, y_pred_sku, y_pred_cid3] = self.hie_rnn_model.predict_on_batch(x_batch)

            if (self.hasAttention()):
                dict_model_output_batch_data = self.model_output_preidct_batch(model_out_layerName_list, x_batch)


                att_sku_batch = self.att_sku_layer_model.predict_on_batch(x_batch)
                att_cid3_batch = self.att_cid3_layer_model.predict_on_batch(x_batch)


            y_true = y

            y_true_sku = y_true[0]
            y_true_cid3 = y_true[2]

            for q, out in enumerate(y_pred_sku):
                # out embedding
                #print "out:", out

                y_pred_sku_emb = y_pred_sku[q]
                y_pred_cid3_emb = y_pred_cid3[q]

                lineIndex = lineNo - 1

                att_sku = []
                att_cid3 = []
                rnn_out_micro_last = []

                if (self.hasAttention()):
                    att_sku = att_sku_batch[q]
                    att_cid3 = att_cid3_batch[q]

                    self.model_output_get_dataCur_from_batch(dict_model_output_batch_data, q, lineIndex, model_out_layerName_list)

                    rnn_out_micro_last = self.dict_model_output_data["rnn_out_micro_last"][lineIndex]

                # second last train_item_id
                xsku = x[0][q, :]
                xcid3 = x[2][q, :]
                secondLastId = x[0][q, -1]

                y_true_sku_id = y_true_sku[q]
                y_true_cid3_id = y_true_cid3[q]

                [res_sku, res_cid3] = self.eval_test_result_example(lineNo, xsku, xcid3, att_sku, att_cid3, rnn_out_micro_last, dict_top_id_recallIdList, self.emb_wgts_micro_items_dict, y_pred_sku_emb, y_true_sku_id, y_pred_cid3_emb, y_true_cid3_id, NList, self.flag_test_candidate_all)
                lineNo += 1

                self.get_metrics_recall_mrr_all(NList, res_sku, res_cid3, recall, mrr)

                allcase += 1

            if (batch_id % 10 == 0):
                print "batch_id/batch_cnt: %d / %d" % (batch_id, batch_cnt)
                # accumulate metrics after this batch
                print "Metrics Until Now:"
                self.eval_metrics_recall_mrr_all(NList, recall, mrr, allcase)


            #break


        print "\nMetrics Final:"
        self.eval_metrics_recall_mrr_all(NList, recall, mrr, allcase)

        endtime = time.time()

        print "Test model time:"
        ProTool.get_time_interval(starttime, endtime)

        print "Test model done!"

        if(self.hasAttention()):
            dict_data = self.dict_model_output_data

            file_data = FileTool.add_folder(self.folder, "pickle_data.pkl")
            FileTool.pickle_save(dict_data, file_data)

    def eval_metrics_recall_mrr_all(self, NList, recall, mrr, cnt):
        print "Evalualation sku start..."
        self.eval_metrics_recall_mrr(NList, recall["sku"], mrr["sku"], cnt)
        print "Evalualation cid3 start..."
        self.eval_metrics_recall_mrr(NList, recall["cid3"], mrr["cid3"], cnt)

    def eval_metrics_recall_mrr(self, NList, recall, mrr, cnt):
        Metrics.get_metrics_recall_mrr_NList(NList, recall, mrr, cnt)
        pass

    def get_metrics_recall_mrr_all(self, NList, res_sku, res_cid3, recall, mrr):
        self.get_metrics_recall_mrr(NList, res_sku, recall["sku"], mrr["sku"])
        self.get_metrics_recall_mrr(NList, res_cid3, recall["cid3"], mrr["cid3"])

    def get_metrics_recall_mrr(self, NList, res, recall, mrr):
        for i in range(len(NList)):
            N = NList[i]
            (flag, rank) = res[i]
            if flag == 1:
                recall[N] += 1
                mrr[N] += 1 / float(rank)
        pass

    def compile_model(self):
        FileTool.func_begin("compile_model")
        self.hie_rnn_model.compile(loss='cosine_proximity', optimizer='rmsprop', loss_weights=self.loss_weights)

    def summary_model(self):
        self.hie_rnn_model.summary()
        print "MODEL METRICS:", self.hie_rnn_model.metrics_names

if __name__ == '__main__':
    model_tag = sys.argv[1]
    layer_num = int(sys.argv[2])
    rnn_state_size = sys.argv[3].split(',')
    bottom_emb_item_len = int(sys.argv[4])
    flag_embedding_trainable = int(sys.argv[5])
    folder_input = sys.argv[6]
    folder_output = sys.argv[7]
    exp_sig = sys.argv[8]
    init_emb_wgts_bootom_items_path = sys.argv[9].split(',')
    train_file = sys.argv[10]
    test_file = sys.argv[11]
    train_len = int(sys.argv[12])
    test_len = int(sys.argv[13])
    seq_len = int(sys.argv[14])
    batch_size = int(sys.argv[15])
    model_save_file = sys.argv[16]
    map_file_bottom_id_to_itemsId = sys.argv[17]
    top_sku_recall_file = sys.argv[18]
    map_file_sku = sys.argv[19]
    flag_train = sys.argv[20]
    train_epoch = int(sys.argv[21])
    flag_test_candidate_all = sys.argv[22]
    mode_attention = int(sys.argv[23])
    drop_out_r = float(sys.argv[24])
    loss_weights = sys.argv[25]
    att_layer_cnt = int(sys.argv[26])
    bhDwellAtt = int(sys.argv[27])
    rnn_type = sys.argv[28]
    RNN_norm = sys.argv[29]


    # model
    hieRNNRec = PRNNRec()
    hieRNNRec.set_files_config(model_tag, layer_num, rnn_state_size, bottom_emb_item_len, flag_embedding_trainable, folder_input, folder_output, exp_sig, init_emb_wgts_bootom_items_path, train_file, test_file, train_len, test_len,
                        seq_len, batch_size, model_save_file, map_file_bottom_id_to_itemsId, top_sku_recall_file, map_file_sku,
                               flag_train, train_epoch, flag_test_candidate_all, mode_attention, drop_out_r, loss_weights, att_layer_cnt, bhDwellAtt, rnn_type, RNN_norm)

    hieRNNRec.load_data()

    if(hieRNNRec.flag_train):
        hieRNNRec.create_model()
        hieRNNRec.compile_model()
        hieRNNRec.summary_model()
        hieRNNRec.train_model()


    else:
        hieRNNRec.load_model()
        hieRNNRec.summary_model()

    hieRNNRec.test_model()

    hieRNNRec.print_log_info()

