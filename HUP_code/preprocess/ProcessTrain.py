from Preprocess import *

from FileTool import *

class ProcessTrain():
    @classmethod
    def split_data_train_test_file(cls, folder, file_in, file_out, max_line=-1, ratio_train=0.982):
        print "split_data_train_test_file..."
        file_out_train = file_out + '.train'
        file_out_test = file_out + '.test'

        file_in = FileTool.add_folder(folder, file_in)
        file_out_train = FileTool.add_folder(folder, file_out_train)
        file_out_test = FileTool.add_folder(folder, file_out_test)

        print "file_in:", file_in

        fout_train = open(file_out_train, 'w')
        fout_test = open(file_out_test, 'w')

        lineCnt = 0
        with open(file_in, 'r') as fin:
            for line in fin:
                if (random.random() <= ratio_train):
                    fout_train.write(line)
                else:
                    fout_test.write(line)
                lineCnt += 1
                if(max_line > 0 and lineCnt >= max_line):
                    break

        fout_train.close()
        fout_test.close()

        print "split_data_train_test_file done!"

    @classmethod
    def split_data_train_test_file_batch(cls, folder, file_in, max_line=-1, ratio_train=0.7):
        print "split_data_train_test_file_batch..."
        file_in = [file_in]
        #file_in=[file_in, file_in + ".SBD.id", file_in + ".SBCD.id"]
        N = len(file_in)

        fin = [None] * N
        file_out_train = [None] * N
        file_out_test = [None] * N
        fout_train = [None] * N
        fout_test = [None] * N

        flinelist = [None] * N

        for i in range(N):
            file_out_train[i] = file_in[i] + '.train'
            file_out_test[i] = file_in[i] + '.test'

            file_out_train[i] = FileTool.add_folder(folder, file_out_train[i])
            file_out_test[i] = FileTool.add_folder(folder, file_out_test[i])

            fout_train[i] = open(file_out_train[i], 'w')
            fout_test[i] = open(file_out_test[i], 'w')

            fin[i] = FileTool.add_folder(folder, file_in[i])

            flinelist[i] = FileTool.read_line_to_list_str(fin[i])

        totalLineCnt = len(flinelist[0])
        if(max_line > 0 and max_line < totalLineCnt):
            totalLineCnt = max_line

        for j in range(totalLineCnt):
            flag_train = (random.random() <= ratio_train)
            for i in range(N):
                line = flinelist[i][j]
                line += "\n"
                if (flag_train):
                    fout_train[i].write(line)
                else:
                    fout_test[i].write(line)

        for i in range(N):
            fout_train[i].close()
            fout_test[i].close()

        print "split_data_train_test_file_batch done!"


    @classmethod
    #get len 40 and split
    def format_data_train_test(cls, folder, file_in, file_mapping, seqlen):
        FileTool.func_begin("format_data_train_test")
        #sesseion.SBCD.id.len40.train
        ProcessTrain.format_data_train_test_parse_line_save_one(folder, file_in, "train", file_mapping, seqlen)
        #sesseion.SBCD.id.len40.test
        ProcessTrain.format_data_train_test_parse_line_save_one(folder, file_in, "test", file_mapping, seqlen)
        # sesseion.SBCD.id.len40.train.div
        ProcessTrain.format_data_train_test_parse_line_div(folder, file_in, "train", file_mapping, seqlen)
        FileTool.func_end("format_data_train_test")

    @classmethod
    def format_data_train_test_parse_line_save_one(cls, folder, file, file_tag, file_mapping, seqlen):
        FileTool.func_begin("format_data_train_test_parse_line_save_one")
        file_in = file + "." + file_tag
        file_out = file + ".len" + str(seqlen) + "." + file_tag
        file_in = FileTool.add_folder(folder, file_in)
        file_out = FileTool.add_folder(folder, file_out)
        file_mapping_f=FileTool.add_folder(folder, file_mapping)
        dict_sessionId_sessionStr = FileTool.load_file_map_idInt_to_itemStr(file_mapping_f)

        listlist = FileTool.read_file_to_list_list(file_in)
        resList = []
        for line in listlist:
            itemList = line[0:seqlen]
            itemList = ProcessTrain.pro_lineEnd_duplicateSku_listid_map(itemList, dict_sessionId_sessionStr)
            if(len(itemList) > 0):
                resList.append(itemList)
        FileTool.write_file_list_list(file_out, resList, Config.file_sep)
        FileTool.func_end("format_data_train_test_parse_line_save_one")

    @classmethod
    def format_data_train_test_parse_line_div(cls, folder, file, file_tag, file_mapping, seqlen):
        FileTool.func_begin("format_data_train_test_parse_line_div")
        file_in = file + "." + file_tag
        file_out = file + ".len" + str(seqlen) + "." + file_tag + ".div"
        file_in = FileTool.add_folder(folder, file_in)
        file_out = FileTool.add_folder(folder, file_out)
        file_mapping_f = FileTool.add_folder(folder, file_mapping)
        dict_sessionId_sessionStr = FileTool.load_file_map_idInt_to_itemStr(file_mapping_f)

        listlist = FileTool.read_file_to_list_list(file_in)
        resList = []
        for line in listlist:
            length=len(line)
            for start in range(0, length-seqlen, Config.seq_split_step):
                itemList = line[start: start+seqlen]
                itemList = ProcessTrain.pro_lineEnd_duplicateSku_listid_map(itemList, dict_sessionId_sessionStr)
                if(len(itemList) > 0):
                    resList.append(itemList)
        FileTool.write_file_list_list(file_out, resList, Config.file_sep)
        FileTool.func_end("format_data_train_test_parse_line_div")

    @classmethod
    def split_data_train_test_batch(cls, folder, file_in_batch, file_out_batch, max_line=-1, ratio_train=0.982):
        file_cnt = len(file_in_batch)

        file_out_train = []
        file_out_test = []

        fout_train = []
        fout_test = []

        fin = []

        # open file
        for i in range(file_cnt):
            file_out_train[i] = FileTool.add_folder(folder, file_out_batch[i] + '.train')
            file_out_test[i] = FileTool.add_folder(folder, file_out_batch[i] + '.test')

            fout_train[i] = open(file_out_train, 'w')
            fout_test[i] = open(file_out_test, 'w')

            file_in_batch[i] = FileTool.add_folder(folder, file_in_batch[i])
            fin[i] = open(file_in_batch[i], 'r')

        # write data
        line = []
        for j in range(max_line):
            flagTrain = False
            if random.random() <= ratio_train:
                flagTrain = True

            for i in range(file_cnt):
                line[i] = fin[i].readline()
                if (flagTrain):
                    fout_train[i].write(line[i])
                else:
                    fout_test[i].write(line[i])

        # close file
        for i in range(file_cnt):
            fin[i].close()
            fout_train[i].close()
            fout_test[i].close()

    @classmethod
    def generate_sequence_data_from_candidate_split(cls, list_list, minLen, maxLen):
        print "generate_sequence_data_from_candidate_split"
        print "list_list:", len(list_list)
        res = []
        pro = 0
        for line in list_list:
            cnt = len(line)
            if (cnt < minLen):
                continue
            split_list = StructureTool.split_list_by_maxlen(line, maxLen, Config.seq_split_step)
            res.extend(split_list)
            pro += 1
            FileTool.print_info_pro(pro)

        print "res:", len(res)
        return res

    @classmethod
    def generate_sequence_data_from_candidate_padding(cls, list_lines, maxLen, padding_str=""):
        print "generate_sequence_data_from_candidate_padding start..."
        res = []
        pro = 0
        line_total_cnt = len(list_lines)
        print "line_total_cnt:%d" % line_total_cnt
        for line in list_lines:
            cnt = len(line)
            w = []

            # padding
            if (padding_str != ""):
                if (cnt < maxLen):
                    pad_cnt = maxLen - cnt
                    w = [padding_str] * pad_cnt

            w.extend(line)
            #line = ' '.join(w)
            res.append(w)

            pro += 1
            FileTool.print_info_pro(pro, line_total_cnt)

        print "res:", len(res)
        return res
        pass

    @classmethod
    def generate_sequence_data_limitLen_padded_from_candidate(cls, list_list, minLen, maxLen, mapIdStr, padding_str="0"):
        print "generate_sequence_data_from_candidate start..."
        listList = ProcessTrain.generate_sequence_data_from_candidate_split(list_list, minLen, maxLen)
        listList_endUniqSku = ProcessTrain.filt_sequence_data_endUniSku(listList, mapIdStr)
        listlist_padded = ProcessTrain.generate_sequence_data_from_candidate_padding(listList_endUniqSku, maxLen, padding_str)
        res = listlist_padded
        return res
        pass

    @classmethod
    def filt_sequence_data_endUniSku(cls, list_list, mapIdStr):
        resList = []
        for line in list_list:
            cur = ProcessTrain.pro_lineEnd_duplicateSku_id_map(line, mapIdStr)
            resList.append(cur)
        return resList


    # raw data to session file(id of each micro item)
    @classmethod
    def get_data_session_train_itemid(cls, folder, file_in, file_out_tag, mode):
        print "get_data_session_train_itemid %s..." % file_in
        micro_item_list = Config.get_micro_item_list(mode)

        file_out = '.'.join([file_out_tag, mode])
        file_in = FileTool.add_folder(folder, file_in)
        file_out = FileTool.add_folder(folder, file_out)

        # file_out_mapping = '.'.join([file_out, "mapping"])
        # file_out_reidx = '.'.join(file_out, "redix")

        # f2 = open(file_out_mapping, 'w')
        # f3 = open(file_out_reidx, 'w')

        # f3.write(' '.join(['0'] * 70) + '\n')

        # filt data to id
        list_lines = Preprocess.get_data_session_train_filt_data_to_itemid(file_in, mode)

        # generate sequence data
        # padding_str = '+'.join(['0'] * len(micro_item_list))
        padding_str = "0"

        # file_out_base = file_out
        # list_res_base = generate_sequence_data_from_candidate(list_lines, seq_len_min, seq_len_max, padding_str="")

        FileTool.printListList(list_lines, file_out)
        # printDict(dict_sessionItem_to_id, file_out_mapping)

        pass

    @classmethod
    def is_same_sku(cls, item1, item2):
        item1 = SessionItem(item1)
        item2 = SessionItem(item2)

        return (item1.sku == item2.sku)

    @classmethod
    # process session data multi item has same sku,
    # goal: in predict, last sku not the same with before
    def pro_lineEnd_duplicateSku_listid_map(cls, list, mapIdStr):
        resList = []
        N = len(list)
        i = N - 1
        while (i > 0):
            curItem = mapIdStr[int(list[i])]
            prevItem =  mapIdStr[int(list[i-1])]
            if (ProcessTrain.is_same_sku(curItem, prevItem)):
                i -= 1
            else:
                break
        resList = list[0:i + 1]
        return resList

    @classmethod
    #process session data multi item has same sku,
    #goal: in predict, last sku not the same with before
    def pro_lineEnd_duplicateSku(cls, list):
        resList = []
        N = len(list)
        i = N-1
        while(i > 0):
            if(ProcessTrain.is_same_sku(list[i], list[i-1])):
                i -= 1
            else:
                break
        resList = list[0:i+1]
        return resList

    @classmethod
    # process session data multi item has same sku,
    # goal: in predict, last sku not the same with before
    def pro_lineEnd_duplicateSku_listid(cls, list, dictIdItemStr):
        resList = []
        N = len(list)
        i = N - 1
        while (i > 0):
            cur=dictIdItemStr[int(list[i])]
            pre=dictIdItemStr[int(list[i-1])]
            if (ProcessTrain.is_same_sku(cur, pre)):
                i -= 1
            else:
                break
        resList = list[0:i + 1]
        return resList

    #last sku must not be the same as before
    @classmethod
    def get_data_session_process_lineEnd_duplicateSku(cls, folder, file_in, file_out):
        FileTool.func_begin("get_data_session_process_lineEnd_duplicateSku")
        file_in = FileTool.add_folder(folder, file_in)
        file_out = FileTool.add_folder(folder, file_out)
        listlist = FileTool.read_file_to_list_list(file_in)

        resList=[]

        i = 0
        for list in listlist:
            list = ProcessTrain.pro_lineEnd_duplicateSku(list)
            resList.append(list)
            i+=1

            FileTool.print_info_pro(i, len(listlist))
        FileTool.printListList(resList, file_out, Config.file_sep)

        FileTool.func_end("get_data_session_process_lineEnd_duplicateSku")


    #format seq to same len
    @classmethod
    def get_file_process_sequence(cls, folder, file_id, file_mapping):

        file_id_ = FileTool.add_folder(folder, file_id)
        file_id_len_ = ".".join([file_id_, Config.seq_len_str])

        mapIdStr = FileTool.load_file_map_idInt_to_itemStr(os.path.join(folder, file_mapping), Config.file_sep)

        #get seq data: limit len, padded
        ProcessTrain.get_sequence_data_limitLen_padded(file_id_, file_id_len_, mapIdStr)

        #list_list_padded = FileTool.read_file_to_list_list(file_id_len_)


        print ""


    #get seq file with micro item set
    @classmethod
    def get_file_micro_items_sequence(cls, folder, file_id, file_mapping, seqlen):
        FileTool.func_begin("get_file_micro_items_sequence")

        file_len =  ".len" + str(seqlen)
        file_mapping_f = FileTool.add_folder(folder, file_mapping)
        dict_sessionId_sessionStr = FileTool.load_file_map_idInt_to_itemStr(file_mapping_f)

        for file_tag in [".train", ".test", ".train.div"]:
            file_id_len = file_id + file_len
            file_in = file_id_len + file_tag
            file_id_f = FileTool.add_folder(folder, file_in)

            # file_id_s = FileTool.add_folder(folder, file_id_len + ".S" + file_tag)
            # file_id_c = FileTool.add_folder(folder, file_id_len + ".C" + file_tag)
            # file_id_sbd = FileTool.add_folder(folder, file_id_len + ".SBD" + file_tag)
            # file_id_sbcd = FileTool.add_folder(folder, file_id_len + ".SBCD" + file_tag)
            file_id_sbcgd = FileTool.add_folder(folder, file_id_len + ".SBCGD" + file_tag)

            #print file_id_s

            #micro_item2id = Data.load_micro_item_id(Config.micro_item_list)

            listSessionId_list = FileTool.read_file_to_list_list(file_id_f, Config.file_sep, "0")

            list_s_list = []
            list_c_list = []
            list_sbd_list = []
            list_sbcd_list = []
            list_sbcgd_list = []
            i = 0

            print "process sequence..."
            print len(listSessionId_list)
            for listSessionId in listSessionId_list:
                list_s = []
                list_c = []
                list_sbd= []
                list_sbcd = []
                list_sbcgd = []
                for sessionId in listSessionId:
                    sessionStr = dict_sessionId_sessionStr[int(sessionId)]
                    sessionItem = SessionItem(sessionStr)
                    sessionItem_s = sessionItem.get_subId("S")
                    sessionItem_c = sessionItem.get_subId("C")
                    sessionItem_sbd = sessionItem.get_subId("SBD")
                    sessionItem_sbcd = sessionItem.get_subId("SBCD")
                    sessionItem_sbcgd = sessionItem.get_subId("SBCGD")
                    list_s.append(sessionItem_s)
                    list_c.append(sessionItem_c)
                    list_sbd.append(sessionItem_sbd)
                    list_sbcd.append(sessionItem_sbcd)
                    list_sbcgd.append(sessionItem_sbcgd)

                list_s = StructureTool.uniq_list(list_s)
                list_c = list_c
                list_sbd = StructureTool.uniq_list(list_sbd)
                list_sbcd = StructureTool.uniq_list(list_sbcd)
                list_sbcgd = StructureTool.uniq_list(list_sbcgd)


                list_s_list.append(list_s)
                list_c_list.append(list_c)
                list_sbd_list.append(list_sbd)
                list_sbcd_list.append(list_sbcd)
                list_sbcgd_list.append(list_sbcgd)

                i += 1
                FileTool.print_info_pro(i, len(listSessionId_list))

            # FileTool.printListList(list_s_list, file_id_s)
            # FileTool.printListList(list_c_list, file_id_c)
            # FileTool.printListList(list_sbd_list, file_id_sbd)
            # FileTool.printListList(list_sbcd_list, file_id_sbcd)
            FileTool.printListList(list_sbcgd_list, file_id_sbcgd)
        FileTool.func_end("get_file_micro_items_sequence")

    @classmethod
    def get_file_micro_items_sequence_train_data(cls, folder, file_id):
        FileTool.func_begin("get_file_micro_items_sequence_train_data")
        # ProcessTrain.get_file_micro_items_sequence_train_data_s(folder, file_id)
        # ProcessTrain.get_file_micro_items_sequence_train_data_c(folder, file_id)
        # ProcessTrain.get_file_micro_items_sequence_train_data_sbd(folder, file_id)
        # ProcessTrain.get_file_micro_items_sequence_train_data_sbcd(folder, file_id)
        ProcessTrain.get_file_micro_items_sequence_train_data_sbcgd(folder, file_id)
        FileTool.func_end("get_file_micro_items_sequence_train_data")

    @classmethod
    def get_file_micro_items_sequence_train_data_sbd(cls, folder, file_id):
        FileTool.func_begin("get_file_micro_items_sequence_train_data_sbd")
        ProcessTrain.build_session_data_train(folder, file_id, "SBD")
        FileTool.func_end("get_file_micro_items_sequence_train_data_sbd")

    @classmethod
    def get_file_micro_items_sequence_train_data_c(cls, folder, file_id):
        FileTool.func_begin("get_file_micro_items_sequence_train_data_c")
        ProcessTrain.build_session_data_train(folder, file_id, "C")
        FileTool.func_end("get_file_micro_items_sequence_train_data_c")

    @classmethod
    def get_file_micro_items_sequence_train_data_s(cls, folder, file_id):
        FileTool.func_begin("get_file_micro_items_sequence_train_data_s")
        ProcessTrain.build_session_data_train(folder, file_id, "S")
        FileTool.func_end("get_file_micro_items_sequence_train_data_s")

    @classmethod
    def get_file_micro_items_sequence_train_data_sbcd(cls, folder, file_id):
        FileTool.func_begin("get_file_micro_items_sequence_train_data_sbcd")
        ProcessTrain.build_session_data_train(folder, file_id, "SBCD")
        FileTool.func_end("get_file_micro_items_sequence_train_data_sbcd")

    @classmethod
    def get_file_micro_items_sequence_train_data_sbcgd(cls, folder, file_id):
        FileTool.func_begin("get_file_micro_items_sequence_train_data_sbcgd")
        ProcessTrain.build_session_data_train(folder, file_id, "SBCGD")
        FileTool.func_end("get_file_micro_items_sequence_train_data_sbcgd")

    @classmethod
    def build_session_data_train(cls, folder, file_id, mode="SBD"):
        FileTool.func_begin("build_session_data_train")
        file_id_len = ".len" + str(Config.seq_len)
        file_base = file_id + file_id_len + "." + mode

        #statis data to .reidx, .mapping
        dictItemStrId = {}
        dictItemIdStr = {}

        dict_itemId_emb = {}
        item_emb_len = Config.get_item_emb_len(mode)
        dict_itemId_emb[0] = [0.] * item_emb_len

        micro_item_vec = Data.load_micro_item_vec_mode_id_file(mode)

        map_id = 1

        i = 0
        for file_tag in [".train", ".test", ".train.div"]:
            file_in =  file_id + file_id_len + "." + mode + file_tag
            file_out = file_id + file_id_len + "." + mode + ".id" + file_tag
            file_in = FileTool.add_folder(folder, file_in)
            file_out = FileTool.add_folder(folder, file_out)
            list_list_data_id = []

            listlist = FileTool.read_file_to_list_list(file_in, Config.file_sep)
            for list in listlist:
                item_id_list = []
                for item in list:
                    if(item not in dictItemStrId):
                        dictItemStrId[item] = map_id
                        dictItemIdStr[map_id] = item
                        map_id += 1

                        sessionItem = SessionItem()
                        sessionItem.set_value(item, mode)
                        sessionItemEmb = sessionItem.getEmb(micro_item_vec, mode)
                        dict_itemId_emb[map_id] = sessionItemEmb
                    item_id = dictItemStrId[item]
                    item_id_list.append(item_id)
                list_list_data_id.append(item_id_list)

                i += 1
                FileTool.print_info_pro(i, len(listlist))

            list_list_data_id_padded = ProcessTrain.generate_sequence_data_from_candidate_padding(list_list_data_id,
                                                                                                  Config.seq_len,
                                                                                                  Config.paddingStr)
            FileTool.printListList(list_list_data_id_padded, file_out)

        file_redix = FileTool.add_folder(folder,  file_base + ".reidx")
        file_mapping = FileTool.add_folder(folder,  file_base + ".mapping")

        FileTool.printDict(dictItemStrId, file_mapping)
        FileTool.printDictEmb(dict_itemId_emb, file_redix, Config.file_sep)

        FileTool.func_end("build_session_data_train")


    @classmethod
    def get_sequence_data_limitLen_padded(cls, file_id, file_id_len, mapIdStr):
        print "get_sequence_data_limitLen_padded..."

        list_list = FileTool.read_file_to_list_list(file_id, sep=Config.file_sep)

        paddingStr=Config.paddingStr
        list_list_padded = ProcessTrain.generate_sequence_data_limitLen_padded_from_candidate(list_list,  Config.seq_len_min, Config.seq_len_max, mapIdStr, padding_str=paddingStr)

        FileTool.printListList(list_list_padded, file_id_len)

        print "get_sequence_data_limitLen_padded done!"

        #return list_list_padded

        pass

    @classmethod
    def get_data_item_uniq_items(cls):
        ProcessTrain.get_data_item_uniq("sku")
        pass

    @classmethod
    def get_data_item_uniq(cls, type_item):
        print "get_data_item_uniq %s..." % type_item
        #get data uniq
        file_uniq = type_item + ".uniq"

        file_raw = type_item + ".raw"

        #
        emb_size = Config.dict_emb_item_size[type_item]

        item2wgt = Data.load_w2v_weights_name(type_item)
        item2id =  Data.load_id_file(Config.add_folder(type_item + ".mapping"))

        print "item2wgt:", len(item2wgt)
        print "item2id:", len(item2id)
        #

        ProcessTrain.get_data_item_uniq_from_raw_filt_by_w2v(Config.folder, file_raw, file_uniq, item2wgt, item2id, Config.seq_len_min, Config.seq_len_max)
        pass


    '''
    data_raw => data_uniq
        remove duplicate items in line
        filt by w2v
    '''

    @classmethod
    def get_data_item_uniq_from_raw_filt_by_w2v(cls, folder, file_in, file_out, item2wgt, item2id, seq_len_min, seq_len_max):

        print "get_data_item_uniq_from_raw_filt_by_w2v %s..." % file_in
        i = 0

        file_in = FileTool.add_folder(folder, file_in)
        file_out = FileTool.add_folder(folder, file_out)

        list_lines = []
        #filt valid data
        with open(file_in, "r") as f:
            for l in f:
                if i % 100000 == 0:
                    print i
                events = l.strip().split()

                save = []
                #remove duplicate
                save = StructureTool.uniq_list(events)

                #print "save:", save

                #check valid
                flag = 0
                save_new = []
                for k in range(len(save)):
                    e = str(save[k])
                    if e not in item2wgt or e not in item2id:
                        #print "Error e:", e
                        flag = 1
                        #break
                    else:
                        save_new.append(str(item2id[e]))

                if(len(save_new) > 0):
                    list_lines.append(save_new)

                i+=1
                FileTool.print_info_pro(i)

        #generate sequence data
        listlist_res = ProcessTrain.generate_sequence_data_limitLen_padded_from_candidate(list_lines, Config.seq_len_min, Config.seq_len_max)
        FileTool.printListList(listlist_res, file_out)

        print "get_data_item_uniq_from_raw_filt_by_w2v done!"

    @classmethod
    def split_data_train_test(cls,  folder, file, file_mapping, max_line_cnt=-1, ratio_train=0.7):
        print "split_data_train_test %s..." % file
        #session.SBCGD.id to session.SBCGD.id.train, session.SBCGD.id.test
        ProcessTrain.split_data_train_test_file_batch(folder, file, max_line_cnt, ratio_train)


        pass

    @classmethod
    def get_map_sku_cid3(cls):
        ProcessTrain.get_map_sku_cid3_org(Config.file_data_src, Config.file_map_sku_cid3)
        ProcessTrain.get_map_sku_cid3_id(Config.file_map_sku_cid3, 'sku.mapping', 'cid3.mapping', Config.file_map_skuId_cid3Id)


    @classmethod
    def get_map_sku_cid3_org(cls, file_in, file_out):
        print "get_map_sku_cid3_org"
        file_in = Config.add_folder(file_in)
        file_out = Config.add_folder(file_out)
        listlist=FileTool.read_file_to_list_list(file_in)

        dict_sku_cid3={}

        i=0
        for list in listlist:
            for item in list:
                arr = item.split('+')
                sku = arr[0]
                cid3=arr[2]
                dict_sku_cid3[sku] = cid3

            i+=1
            if(i% 10000 == 0):
                print "%d/%d" % (i, len(listlist))
        FileTool.printDict(dict_sku_cid3, file_out)

    @classmethod
    def get_map_sku_cid3_id(cls, file_sku_cid3, file_sku, file_cid3, file_skuId_cid3Id):
        print "get_map_sku_cid3_org"
        [file_sku_cid3, file_sku, file_cid3, file_skuId_cid3Id] = Config.add_folder_fileList([file_sku_cid3, file_sku, file_cid3, file_skuId_cid3Id])

        map_skuStr_cid3Str = FileTool.load_file_map_colInt_colInt(file_sku_cid3)
        map_skuStr_skuId = FileTool.load_file_map_colInt_colInt(file_sku)
        map_cid3Str_cid3Id = FileTool.load_file_map_colInt_colInt(file_cid3)


        dict_skuId_cid3Id = {}

        i=0
        for (skuStr,cid3Str) in map_skuStr_cid3Str.items():
	    if(skuStr not in map_skuStr_skuId):
                print "sku %d not in map" % skuStr
            if(cid3Str not in map_cid3Str_cid3Id):
                print "cid3 %d not in map" % cid3Str

            if(skuStr in map_skuStr_skuId  and cid3Str in map_cid3Str_cid3Id):
                skuId = map_skuStr_skuId[skuStr]
                cid3Id = map_cid3Str_cid3Id[cid3Str]
                dict_skuId_cid3Id[skuId] = cid3Id
            i+=1
            if(i% 10000 == 0):
                print "%d/%d" % (i, len(map_skuStr_cid3Str))

        FileTool.printDict(dict_skuId_cid3Id, file_skuId_cid3Id)



    @classmethod
    def get_file_data_sbcd(cls):
        ProcessTrain.get_file_sbcd()
        ProcessTrain.get_file_sbcd_data_train()
        ProcessTrain.get_file_sbcd_data_train_micro()
        ProcessTrain.get_file_sbcd_pickle()
        pass

    @classmethod
    def analyse_file_data_sbcd(cls):
        #ProcessTrain.analyse_file_sbcd_data_item()
        ProcessTrain.analyse_file_interactions()

    @classmethod
    def analyse_file_interactions(cls):
        file_train = "sbcd.id.train.item"
        file_test = "sbcd.id.test.item"

        ProcessTrain.analyse_file_interactions_file(file_train)
        ProcessTrain.analyse_file_interactions_file(file_test)
        pass

    @classmethod
    def get_list_micro_list(cls, list, micro_type):
        res = []
        for item in list:
            if (item == Config.paddingStr):
                continue

            sessionItem = SessionItem(item)
            if(micro_type == "sku"):
                micro = sessionItem.sku
            if(micro_type == "cid3"):
                micro = sessionItem.cid3

            res.append(micro)
        return res

    @classmethod
    def get_list_seq_distinct_cnt(cls, list):
        pre = "-1"
        uni_list = []
        cnt = 0
        for item in list:
            if(item != pre):
                cnt += 1
                pre = item
        return cnt


    @classmethod
    def analyse_file_interactions_file_micro(cls, listlist, micro_type):
        total_cnt = 0
        for list in listlist:
            list_micro = ProcessTrain.get_list_micro_list(list, micro_type)
            cnt_distinct = ProcessTrain.get_list_seq_distinct_cnt(list_micro)
            total_cnt += cnt_distinct
        return total_cnt
        pass

    @classmethod
    def analyse_file_interactions_file(cls, file):
        FileTool.func_begin("analyse_file_interactions_file")
        file = Config.add_folder(file)
        listlist = FileTool.read_file_to_list_list(file)

        print "file:", file
        totalCnt_sku = ProcessTrain.analyse_file_interactions_file_micro(listlist, "sku")
        totalCnt_cid3 = ProcessTrain.analyse_file_interactions_file_micro(listlist, "cid3")

        print "file: %s, N_sku:%s, N_cid3:%s" % (str(file), str(totalCnt_sku), str(totalCnt_cid3))

        FileTool.func_end("analyse_file_interactions_file")


    @classmethod
    def analyse_file_sbcd_data_item(cls):
        FileTool.func_begin("analyse_file_sbcd_data_item")
        file_item="sbcd.id.test.item"
        file_item=Config.add_folder(file_item)

        print file_item
        listlist = FileTool.read_file_to_list_list(file_item)


        cnt = 0
        cnt_same_cid3=0
        cnt_line_multi_cid3=0
        for list in listlist:
            item1 = list[-2]
            item2 = list[-1]

            sessionItem1 = SessionItem(item1)
            sessionItem2 = SessionItem(item2)

            cnt += 1

            flag_same_cid3 = (sessionItem1.cid3 == sessionItem2.cid3)
            if(flag_same_cid3):
                cnt_same_cid3 += 1

            list_cid3_set=set()
            for item in list:
                if(item == Config.paddingStr):
                    continue
                sessionItem = SessionItem(item)
                list_cid3_set.add(sessionItem.cid3)
            if(len(list_cid3_set) > 1):
                cnt_line_multi_cid3 += 1


        str = "cnt_same_cid3, cnt, ratio: %d, %d, %f" % (cnt_same_cid3, cnt, cnt_same_cid3 / float(cnt))
        print str

        str = "cnt_line_multi_cid3, cnt, ratio: %d, %d, %f" % (cnt_line_multi_cid3, cnt, cnt_line_multi_cid3 / float(cnt))
        print str

        FileTool.func_end("analyse_file_sbcd_data_item")


    @classmethod
    def get_file_sbcd_data_train(cls):
        ProTool.CopyFile(Config.folder, "bhdwell.train", Config.folder, "sbcd.id.train")
        ProTool.CopyFile(Config.folder, "bhdwell.train.div", Config.folder, "sbcd.id.train.div")
        ProTool.CopyFile(Config.folder, "bhdwell.test", Config.folder, "sbcd.id.test")
        pass

    @classmethod
    def get_file_sbcd_data_train_micro(cls):
        FileTool.func_begin("get_file_sbcd_data_train_micro")
        file_mapping = FileTool.add_folder(Config.folder, "sbcd.id.mapping")
        dict_idInt_itemStr = Data.load_file_map_itemStr_idInt(file_mapping, True)

        ProcessTrain.get_file_sbcd_data_train_micro_file(Config.folder, "sbcd.id.train", "sbcd.id.train.cid3", dict_idInt_itemStr)
        ProcessTrain.get_file_sbcd_data_train_micro_file(Config.folder, "sbcd.id.train.div", "sbcd.id.train.div.cid3", dict_idInt_itemStr)
        ProcessTrain.get_file_sbcd_data_train_micro_file(Config.folder, "sbcd.id.test", "sbcd.id.test.cid3", dict_idInt_itemStr)
        FileTool.func_end("get_file_sbcd_data_train_micro")
        pass

    @classmethod
    def get_file_sbcd_data_train_micro_file(cls, folder, file_in, file_out, dict_idInt_itemStr):
        [file_in, file_out] = FileTool.add_folder_fileList(folder, [file_in, file_out])

        func_str = "get_file_sbcd_data_train_micro_file %s" % file_in
        FileTool.func_begin(func_str)

        listlist = FileTool.read_file_to_list_list(file_in)


        listlist_res = []
        for list in listlist:
            listNew = []
            for item in list:
                if(str(item) == Config.paddingStr):
                    listNew.append(item)
                    continue
                itemStr = dict_idInt_itemStr[int(item)]
                sessionItem = SessionItem(itemStr)
                cid3 = sessionItem.cid3
                listNew.append(cid3)
            listlist_res.append(listNew)

        FileTool.printListList(listlist_res, file_out)

        FileTool.func_end(func_str)
        pass

    @classmethod
    def get_file_id_to_item(cls):
        FileTool.func_begin("get_file_id_to_item")
        #ProcessTrain.trans_file_id_to_item(Config.folder, "sbcd.id.test", "sbcd.id.test.item", "sbcd.mapping")
        #ProcessTrain.trans_file_id_to_item(Config.folder, "sbcd.id.test", "sbcd.id.test.id", "sbcd.id.mapping")

        ProcessTrain.trans_file_id_to_item(Config.folder, "sbcd.id.train", "sbcd.id.train.item", "sbcd.mapping")
        ProcessTrain.trans_file_id_to_item(Config.folder, "sbcd.id.train", "sbcd.id.train.id", "sbcd.id.mapping")


        #ProcessTrain.trans_file_id_to_item(Config.folder, "sku.test", "sku.test.item", "sku.mapping")
        FileTool.func_end("get_file_id_to_item")

    @classmethod
    def get_file_sbcd_pickle(cls):
        FileTool.func_begin("get_file_sbcd_pickle")
        micro_item_list = Config.get_micro_item_list("SBCD")
        ProcessTrain.get_file_sbcd_pickle_bottom_to_micro(Config.folder, "sbcd.id.train", 1154771, "sbcd.id.mapping", micro_item_list)
        #ProcessTrain.get_file_sbcd_pickle_bottom_to_micro(Config.folder, "sbcd.id.train.div", 23491738, "sbcd.id.mapping", micro_item_list)
        #ProcessTrain.get_file_sbcd_pickle_bottom_to_micro(Config.folder, "sbcd.id.test", 20895, "sbcd.id.mapping", micro_item_list)
        FileTool.func_end("get_file_sbcd_pickle")
        pass

    @classmethod
    def get_file_sbcd_pickle_bottom_to_micro(cls, folder, train_file, train_len, map_file_bottom_id_to_itemsId, micro_item_list):

        [train_file, map_file_bottom_id_to_itemsId] = FileTool.add_folder_fileList(folder, [train_file, map_file_bottom_id_to_itemsId])

        [data_micro, total_data_line_cnt] = Data.load_data_trans_bottomId_to_microItemsId(train_file,
                                                                                          train_len,
                                                                                          map_file_bottom_id_to_itemsId,
                                                                                          micro_item_list)
        data = [data_micro, total_data_line_cnt]
        file_pickle = train_file + Config.file_pickle
        FileTool.pickle_save(data, file_pickle)

    @classmethod
    def trans_file_id_to_item(cls, folder, file_data_id, file_data_item, file_mapping):
        FileTool.func_begin("trans_file_id_to_item")
        print file_data_id
        [file_data_id, file_data_item, file_mapping] = FileTool.add_folder_fileList(folder, [file_data_id, file_data_item, file_mapping])
        list_list_dataId = FileTool.read_file_to_list_list(file_data_id)
        dict_idInt_itemStr = Data.load_file_map_itemStr_idInt(file_mapping, True)

        list_list_dataItem = "0"
        list_list_res = []
        for list in list_list_dataId:
            list_itemNew = []
            for item in list:
                if item == Config.paddingStr:
                    list_itemNew.append(item)
                    continue
                itemNew = dict_idInt_itemStr[int(item)]
                list_itemNew.append(itemNew)
            list_list_res.append(list_itemNew)

        FileTool.printListList(list_list_res, file_data_item)
        FileTool.func_begin("trans_file_id_to_item")




    @classmethod
    def get_file_sbcd(cls):
        FileTool.func_begin("get_file_sbcd")

        ProcessTrain.get_file_sbcd_file_map("bhdwell.mapping", "sbcd.mapping")
        ProcessTrain.get_file_sbcd_id_file_map("sbcd.mapping", "sbcd.id.mapping")
        FileTool.func_end("get_file_sbcd")

    @classmethod
    def get_file_sbcd_file_map(cls, file_in, file_out):
        FileTool.func_begin("get_file_sbcd_file_map")
        [file_in, file_out] = Config.add_folder_fileList([file_in, file_out])
        print file_in

        dict_skuInt_cid3Int = Data.load_sku_cid3()

        listlist = FileTool.read_file_to_list_list(file_in)

        listlistRes = []
        for list in listlist:
            listNew = []
            item, id = list[0], list[1]
            arr = item.split('+')
            sku, bh, dwell = arr[0], arr[1], arr[2]
            cid3 = dict_skuInt_cid3Int[int(sku)]
            itemNew = '+'.join([sku, bh, str(cid3), dwell])
            listNew = [itemNew, id]
            listlistRes.append(listNew)
        FileTool.printListList(listlistRes, file_out)

        FileTool.func_end("get_file_sbcd_file_map")

    @classmethod
    def get_file_sbcd_id_file_map(cls, file_in, file_out):
        FileTool.func_begin("get_file_sbcd_id_file_map")
        [file_in, file_out] = Config.add_folder_fileList([file_in, file_out])
        print file_in

        listlist = FileTool.read_file_to_list_list(file_in)
        dict_item_itemInt_idInt = Data.load_micro_itemInt_idInt(["sku", "base_bh", "cid3", "base_dwell"])

        listlistRes = []
        for list in listlist:
            listNew = []
            item, id = list[0], list[1]
            sessionItem = SessionItem()
            sessionItem.set_value(item, "SBCD")
            itemNew = sessionItem.get_Id(dict_item_itemInt_idInt, "SBCD")
            listNew = [itemNew, id]
            listlistRes.append(listNew)
        FileTool.printListList(listlistRes, file_out)

        FileTool.func_end("get_file_sbcd_id_file_map")

    @classmethod
    def get_file_sbcd_id(cls, dict_skuInt_cid3Int):
        FileTool.func_begin("get_file_sbcd_id")

        Data.load_micro_itemInt_idInt(["sku", "base_bh", "base_dwell", "cid3"])

        ProcessTrain.get_file_sbcd_id_file("sbcd.train", "sbcd.id.train", dict_skuInt_cid3Int)
        ProcessTrain.get_file_sbcd_id_file("sbcd.train.div", "sbcd.train.div", dict_skuInt_cid3Int)
        ProcessTrain.get_file_sbcd_id_file("sbcd.test", "sbcd.test", dict_skuInt_cid3Int)
        FileTool.func_end("get_file_sbcd_id")

    @classmethod
    def get_file_sbcd_file(cls, file_in, file_out, dict_skuInt_cid3Int):
        FileTool.func_begin("get_file_sbcd_file")
        [file_in, file_out] = Config.add_folder_fileList([file_in, file_out])
        print file_in
        listlist = FileTool.read_file_to_list_list(file_in)

        listlistRes = []
        for list in listlist:
            listNew = []
            for item in list:
                if(item == Config.paddingStr):
                    listNew.append(item)
                    continue

                arr = item.split('+')
                sku = int(arr[0])
                cid3 = dict_skuInt_cid3Int[sku]
                itemNew = '+'.join(item, str(cid3))
                listNew.append(itemNew)
            listlistRes.append(listNew)
        FileTool.printListList(listlistRes, file_out)
        FileTool.func_end("get_file_sbcd_file")

        pass

    @classmethod
    def get_file_sbcd_id_file(cls, file_in, file_out, dict_skuInt_cid3Int):
        FileTool.func_begin("get_file_sbcd_id_file")
        [file_in, file_out] = Config.add_folder_fileList([file_in, file_out])
        print file_in
        listlist = FileTool.read_file_to_list_list(file_in)

        listlistRes = []
        for list in listlist:
            listNew = []
            for item in list:
                if (item == Config.paddingStr):
                    listNew.append(item)
                    continue

                arr = item.split('+')
                sku = int(arr[0])
                cid3 = dict_skuInt_cid3Int[sku]
                itemNew = '+'.join(item, str(cid3))
                listNew.append(itemNew)
            listlistRes.append(listNew)
        FileTool.printListList(listlistRes, file_out)
        FileTool.func_end("get_file_sbcd_id_file")

        pass
