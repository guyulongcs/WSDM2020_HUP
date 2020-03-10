import os

import pickle
from StructureTool import *

import pandas as pd

import numpy as np

class FileTool():

    @classmethod
    def print_info_pro(cls, cnt, cnt_total=-1, base=100000):
        if(cnt % base == 0):
            if(cnt_total == -1):
                print "processed %d" % cnt
            else:
                print "processed %d/%d" % (cnt, cnt_total)

    @classmethod
    def pickle_save(cls, data, file):
        str = "pickle_save %s" % file
        FileTool.func_begin(str)
        pickle.dump(data, open(file, "wb"))
        FileTool.func_end(str)
        pass

    @classmethod
    def pickle_load(cls,  file):
        str = "pickle_load %s" % file
        FileTool.func_begin(str)
        data = pickle.load(open(file, "rb"))
        FileTool.func_end(str)
        return data
        pass


    @classmethod
    def get_file_line_seqlen_list(cls, file, sep=' '):
        seqlenList = []
        with open(file, 'r') as f:
            cnt = 0
            for l in f:
                items = l.strip().split(sep)
                seqlen = len(items)
                seqlenList.append(seqlen)
                cnt += 1
                FileTool.print_info_pro(cnt)
        return seqlenList

    @classmethod
    def read_line_to_list_str(cls, file):
        list = []
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                list.append(line)
        return list

    @classmethod
    def read_file_to_list_list(cls, file, sep=' ', lineItemFilt=""):
        print "read_file_to_list_list %s" % file
        lineList = FileTool.read_line_to_list_str(file)
        listList = StructureTool.format_listStr_to_listList(lineList, sep, lineItemFilt)
        print "read_file_to_list_list done!"
        return listList

    @classmethod
    def load_file_map_idInt_to_itemStr(cls, file_item_id, sep=' '):
        print "load_file_map_idInt_to_itemStr %s" % file_item_id

        dict = {}
        listlist = FileTool.read_file_to_list_list(file_item_id, sep)
        for list in listlist:
            dict[int(list[1])] = str(list[0])

        print "load_file_map_idInt_to_itemStr done!"
        return dict

    @classmethod
    def load_file_map_colInt_colInt(cls, file, reverseCol=False, sep=' '):
        print "load_file_map_colInt_colInt %s" % file
        dict = {}
        listlist = FileTool.read_file_to_list_list(file, sep)
        for list in listlist:
            if(not reverseCol):
                dict[int(list[0])] = int(list[1])
            else:
                dict[int(list[1])] = int(list[0])

        print "load_file_map_idInt_to_itemStr done!"
        return dict

    @classmethod
    def printListList(cls, listlist, file_out='', sep=' '):
        print file_out
        list = StructureTool.format_listlist_to_list(listlist, sep)
        if(file_out != ''):
            FileTool.printList(list, file_out)

    @classmethod
    def printListName(cls, list, name=''):
        if(name != ''):
            print name
        for item in list:
            print str(item) + "\n"

    @classmethod
    def printList(cls, list, file_out=''):
        if(file_out == ''):
            FileTool.printListName(list)
        else:
            print file_out
            with open(file_out, 'w') as f:
                for item in list:
                    str = item + "\n"
                    f.write(str)

    @classmethod
    def printDict(cls, dict, file_out=''):
        print "printDict %s..." % file_out
        listRes = StructureTool.get_dict_to_list_str(dict)
        FileTool.printList(listRes, file_out)

    @classmethod
    def printDictEmb(cls, dictEmb, file_out, sep=' '):
        print file_out
        resList = []
        for key in sorted(dictEmb.keys()):
            valueList = dictEmb[key]
            valueStrList = StructureTool.format_list_to_str(valueList)
            resList.append(valueStrList)
        FileTool.printListList(resList, file_out, sep)


    @classmethod
    def get_file_line_item_dict(cls, file, sep=' '):
        dict = {}
        with open(file, 'r') as f:
            cnt = 0
            for l in f:
                items = l.strip().split(sep)
                for item in items:
                    StructureTool.addDict(dict, item)
                cnt += 1
                FileTool.print_info_pro(cnt)
        return dict

    @classmethod
    def add_folder(cls, folder, file):
        return os.path.join(folder, file)

    @classmethod
    def add_folder_fileList(cls, folder, fileList):
        resList = [FileTool.add_folder(folder, file) for file in fileList]
        return resList

    @classmethod
    def get_file_line_seqlen_distribution(cls, folder, file_in, file_out='', sep=' '):
        print "get_file_line_seqlen_distribution %s" % file_in

        file_in = FileTool.add_folder(folder, file_in)
        file_out = FileTool.add_folder(folder, file_out)

        seqlenList = FileTool.get_file_line_seqlen_list(file_in, sep)
        dictCnt = StructureTool.get_list_distribution(seqlenList)

        FileTool.printDict(dictCnt, file_out)

        print "get_file_line_seqlen_distribution done!"

    @classmethod
    def get_file_item_distribution(cls, folder, file_in, file_out='', sep=' '):
        print "get_file_item_distribution %s" % file_in

        file_in = FileTool.add_folder(folder, file_in)
        file_out = FileTool.add_folder(folder, file_out)


        dict = FileTool.get_file_line_item_dict(file_in, sep)
        FileTool.printDict(dict, file_out)

    @classmethod
    def read_data_train(cls, train_data_path, data_len, start):
        print "read data: %s" % train_data_path
        tables = []
        batch = 10000
        while (start < data_len):
            a = pd.read_table(train_data_path, sep=' ', header=None, skiprows=start, nrows=batch)
            tables.append(a.fillna(value=0).values.astype('int32'))
            start += batch
            print "start:", start
        print "read_data end!"
        return tables

    @classmethod
    def add_folder(cls, folder, file):
        return os.path.join(folder, file)

    @classmethod
    def add_folder_file_list(cls, folder, filelist):
        res = []
        for file in filelist:
            file_new = FileTool.add_folder(folder, file)
            res.append(file_new)
        return res

    @classmethod
    def load_data_text_np(cls, file, dtype=None):
        print "load_data_text_np %s start..." % file
        if(dtype == None):
            res = np.loadtxt(file)
        else:
            res = np.loadtxt(file, dtype=dtype)
        print "load_data_text_np done!"
        return res

    @classmethod
    def func_begin(cls, str):
        print "\n%s start..." % str

    @classmethod
    def func_end(cls, str):
        print "%s end!" % str

    @classmethod
    def get_file_reverse_col(cls, file_in, file_out, sep=' '):
        print "get_file_reverse_col %s" % file_in
        listlist = FileTool.read_file_to_list_list(file_in, sep)
        dict={}
        for list in listlist:
            v1, v2 =list[0], list[1]
            dict[v2] = v1
        FileTool.printDict(dict, file_out)

    @classmethod
    def write_file_listStr(cls, file, listStr):
        with open(file, 'w') as f:
            for str in listStr:
                f.write(str + "\n")

    @classmethod
    def write_file_list_list(cls, file, listlist, sep='\t'):
        listStr = []
        for list in listlist:
            listStr.append(sep.join(list))
        FileTool.write_file_listStr(file, listStr)

    @classmethod
    def filt_file_by_lineItemCnt(cls, file_in, file_out, sep, min_cnt, max_cnt=-1):
        listlist = FileTool.read_file_to_list_list(file_in, sep)
        validlist = []

        for list in listlist:
            valid = True
            cnt = len(list)
            if(min_cnt >= 0):
                if(cnt < min_cnt):
                    valid=False
            if(max_cnt >= 0):
                if(cnt > max_cnt):
                    valid=False
            if(valid):
                validlist.append(list)
        FileTool.write_file_list_list(file_out, validlist, sep)

    @classmethod
    def get_file_line_uniq_item(cls, file_in, file_out, sep):
        listlist = FileTool.read_file_to_list_list(file_in, sep)

        validlist=[]
        for line in listlist:
            cur=[]
            prev=None
            for item in line:
                if(prev != None and item == prev):
                    continue
                cur.append(item)
                prev=item
            validlist.append(cur)
        FileTool.write_file_list_list(file_out, validlist, sep)






