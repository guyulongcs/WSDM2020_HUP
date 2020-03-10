from Config import *
import numpy as np
from numpy import *
import pandas as pd
from FileTool import *
from SessionItem import *

class Data():
    @classmethod
    def load_w2v_weights_name(cls, type):
        file = type + ".w2v"

        file = Config.add_folder(file)
        print "load_w2v_weights %s" % file

        item_wgts = np.loadtxt(file, skiprows=2)

        print "normalization..."
        item2wgt = dict()
        for l in item_wgts:
            item2wgt[str(int(l[0]))] = l[1:] / linalg.norm(l[1:])

        return item2wgt

    @classmethod
    def load_w2v_weights_id_file(cls, file):
        print "load_w2v_weights_id_file %s" % file
        item_wgts = np.loadtxt(file, skiprows=0)
        print "len:", len(item_wgts)
        return item_wgts

    @classmethod
    def load_w2v_weights_w2v_file(cls, file):
        print "load_w2v_weights_w2v_file %s" % file
        item_wgts = np.loadtxt(file, skiprows=2)
        print "normalization..."
        item2wgt = dict()
        for l in item_wgts:
            item2wgt[str(int(l[0]))] = l[1:] / linalg.norm(l[1:])

        print "len:", len(item2wgt )
        return item2wgt


    @classmethod
    def load_itemInt_idInt(cls, type):
        file = type + '.mapping'

        file = Config.add_folder(file)

        item2id = Data.load_file_map_itemInt_idInt(file)
        print "len:", len(item2id)

        return item2id

    @classmethod
    def load_id_file(cls, file):
        print "load_id %s" % file

        item_id = np.loadtxt(file, skiprows=0)

        item2id = dict()
        for l in item_id:
            item2id[str(int(l[0]))] = int(l[1])

        return item2id

    @classmethod
    def load_mapping_dict_itemStr_idInt(cls, type, reverse=False):
        file = type + '.mapping'

        file = Config.add_folder(file)

        dict = Data.load_file_map_itemStr_idInt(file, reverse)

        return dict

    @classmethod
    def load_reidx_dict_idInt_emb(cls, type):
        file = type + '.reidx'
        file = Config.add_folder(file)
        dict = Data.load_w2v_weights_id_file(file)
        return dict

    @classmethod
    def load_file_map_itemStr_idInt(cls, file, reverse=False):
        print "load_id %s" % file

        item_id = np.loadtxt(file, skiprows=0, dtype='str')

        dict = {}
        for l in item_id:
            if(not reverse):
                dict[str(l[0])] = int(l[1])
            else:
                dict[int(l[1])] = str(l[0])

        print "load file done!"
        return dict

    @classmethod
    def load_file_map_itemInt_idInt(cls, file, reverse=False):
        print "load_id %s" % file

        item_id = np.loadtxt(file, skiprows=0, dtype='str')

        dict = {}
        for l in item_id:
            if (not reverse):
                dict[int(l[0])] = int(l[1])
            else:
                dict[int(l[1])] = int(l[0])

        return dict


    @classmethod
    def load_micro_item_vec(cls, micro_item_file_list, micro_item_list):
        print "load_micro_item_vec start..."
        item2vec = {}
        for i in range(len(micro_item_list)):
            micro_item = micro_item_list[i]
            micro_item_file = micro_item_file_list[i]
            item2vec[micro_item] = Data.load_w2v_weights_id_file(micro_item_file)
        print "load_micro_item_vec done..."
        return item2vec

    @classmethod
    def load_micro_item_vec_mode(cls, mode):
        print "load_micro_item_vec start..."
        item2vec = {}
        micro_item_list = Config.get_micro_item_list(mode)
        micro_item_file_list = Config.get_micro_item_file_list_w2v(micro_item_list)
        for i in range(len(micro_item_list)):
            micro_item = micro_item_list[i]
            micro_item_file = micro_item_file_list[i]
            item2vec[micro_item] = Data.load_w2v_weights_w2v_file(micro_item_file)
        print "load_micro_item_vec done..."
        return item2vec

    @classmethod
    def load_micro_item_vec_mode_id_file(cls, mode):
        print "load_micro_item_vec start..."
        item2vec = {}
        micro_item_list = Config.get_micro_item_list(mode)
        micro_item_file_list = Config.get_micro_item_file_list(micro_item_list)
        for i in range(len(micro_item_list)):
            micro_item = micro_item_list[i]
            micro_item_file = micro_item_file_list[i]
            item2vec[micro_item] = Data.load_w2v_weights_id_file(micro_item_file)
        print "load_micro_item_vec done..."
        return item2vec

    @classmethod
    def load_micro_itemInt_idInt(cls, micro_item_list):
        print "load_micro_item_id start..."
        item2id = {}
        for micro_item in micro_item_list:
            item2id[micro_item] = Data.load_itemInt_idInt(micro_item)
        return item2id

    @classmethod
    def load_sku_cid3(cls):
        file = Config.add_folder("sku_cid3")
        dict_sku_cid3 = FileTool.load_file_map_colInt_colInt(file)
        return dict_sku_cid3

    @classmethod
    def load_data_table(cls, file, sep=' ', header=None, nrows=None, skiprows=0):
        print "load_data_pd %s start..." % file
        a = pd.read_table(file, sep=sep, header=header, nrows=nrows, skiprows=skiprows)
        a = a.fillna(value=0).values.astype('int32')
        print "load_data_pd end!"
        return a

    @classmethod
    def load_data_dict_bottomsId_microItemsId(cls, file_map_bottomItem_bottomId, micro_item_list):
        FileTool.func_begin("load_data_dict_bottomsId_microItemsId")
        dict_bottomIdInt_to_bottomItemStr = FileTool.load_file_map_idInt_to_itemStr(file_map_bottomItem_bottomId)
        print "dict_bottomIdInt_to_bottomItem :", len(dict_bottomIdInt_to_bottomItemStr)
        print dict_bottomIdInt_to_bottomItemStr[1]

        dict_bottomsId_microItemsId = {}

        for item in micro_item_list:
            dict_bottomsId_microItemsId[item]={}
            dict_bottomsId_microItemsId[item][0] = 0

        for bottomId in dict_bottomIdInt_to_bottomItemStr.keys():
            bottomItem = dict_bottomIdInt_to_bottomItemStr[bottomId]

            sessionItem = SessionItem(str(bottomItem))
            sessionItemDict = sessionItem.getDict()

            for item in micro_item_list:
                dict_bottomsId_microItemsId[item][bottomId] = int(sessionItemDict[item])

        return dict_bottomsId_microItemsId

        FileTool.func_end("load_data_dict_bottomsId_microItemsId")
        pass

    @classmethod
    def load_data_trans_bottomId_to_microItemsId_pickle(cls, file_bottomId, nrows, file_map_bottomItem_bottomId, micro_item_list):
        file_pickle = file_bottomId + "_pickle"
        data = FileTool.pickle_load(file_pickle)
        [data_micro, total_data_line_cnt] = data
        return [data_micro, total_data_line_cnt]
        pass

    @classmethod
    def load_data_trans_bottomId_to_microItemsId(cls, file_bottomId, nrows, file_map_bottomItem_bottomId, micro_item_list):
        FileTool.func_begin("load_data_trans_bottomId_to_microItemsId")
        data_bottomId = Data.load_data_table(file_bottomId, nrows=nrows)

        dict_bottomsId_microItemsId = Data.load_data_dict_bottomsId_microItemsId(file_map_bottomItem_bottomId, micro_item_list)

        (rowLen, colLen) = data_bottomId.shape
        print "data_bottomId.shape: (%d, %d)" % (rowLen, colLen)

        #dict
        data_micro_items_dict = {}

        print "create micro items id..."
        micro_item_len = len(micro_item_list)
        for item in micro_item_list:
            data_micro_items_dict[item] = np.ndarray(shape=(rowLen, colLen), dtype='int32')

        for i in range(rowLen):
            for j in range(colLen):
                bottomId = int(data_bottomId[i][j])
                for item in micro_item_list:
                    data_micro_items_dict[item][i][j] = dict_bottomsId_microItemsId[item][bottomId]

            FileTool.print_info_pro(i, rowLen)

        print "create micro items id done!"

        #list
        data_micro_items_list = []
        for item in micro_item_list:
            data_micro_items_list.append(data_micro_items_dict[item])

        FileTool.func_end("load_data_trans_bottomId_to_microItemsId")

        return [data_micro_items_list, rowLen]

    @classmethod
    def get_embedding(cls, list, emb_wgts):
        res = []
        for item in list:
            embed = emb_wgts[item]
            res.append(embed)
        return res

        pass

    @classmethod
    def get_embedding_arr(cls, list, emb_wgts):
        res = Data.get_embedding(list, emb_wgts)
        res = np.array(res)
        return res

        pass

