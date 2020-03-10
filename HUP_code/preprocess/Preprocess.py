from Data import *
from SessionItemBase import *
from FileTool import *
from ProTool import *
import numpy as np


class Preprocess():

    @classmethod
    def add_folder_file(cls, file_in):
        res = os.path.join(Config.folder, file_in)
        return res

    #input file header: id(0),behavior(1),sku(2),cid3(3),brand(4),price(5),dwell(6),datetime(7),timestamp(8)
    #output file header: sku+behavior+cid3+gap+dwell, connected using ' ' to form a line of session data
    @classmethod
    def extract_sessions_from_org_data(cls, folder, file_in, file_out, file_log):
        print "extract_sessions_from_org_data..."
        file_in = FileTool.add_folder(folder, file_in)
        file_out = FileTool.add_folder(folder, file_out)
        file_log = FileTool.add_folder(folder, file_log)
        print file_in

        set_id=set()
        set_bh=set()
        set_cid3=set()
        set_sessDur=set()

        linenum = 1
        count = 0

        f1 = open(file_out, 'w')
        with open(file_in, "r") as f:
            for l in f:
                try:
                    events = l.strip().strip('{(').strip(')}').split('),(')
                    el = len(events)-1
                    session = []
                    start = 0
                    flag = 0

                    sessStart=0
                    sessEnd=0
                    for i in range(el):
                        token = events[i].split(',')
                        id = int(token[0])
                        bh = token[1]
                        sku = token[2]
                        cid3 = int(token[3])

                        dwell = token[6]

                        behavior = Config.behavior2id(token[1])

                        set_id.add(id)
                        set_bh.add(bh)
                        set_cid3.add(cid3)


                        if events[i + 1].split(',')[8] == '' or token[8] == '':
                            flag = 1
                            break
                        gap = str(int(events[i + 1].split(',')[8]) - int(token[8]))
                        cid3 = token[3]
                        if i == 0:
                            present_sku = sku
                            present_dwell = dwell
                            sessStart = int(token[8])
                        if i == el-1:
                            sessEnd = int(token[8])
                        else:
                            if sku != present_sku:
                                if present_dwell != '':
                                    for j in range(start, len(session)):
                                        session[j] += '+' + present_dwell
                                    start = i
                                    present_sku = sku
                                    present_dwell = dwell
                                else:
                                    flag = 1
                                    break
                            if dwell != '':
                                if present_dwell != '':
                                    if int(dwell) > int(present_dwell):
                                        present_dwell = dwell
                                else:
                                    present_dwell = dwell
                        w = sku + '+' + behavior + '+' + cid3 + '+' + gap
                        session.append(w)
                    if present_dwell != '':
                        for j in range(start, len(session)):
                            session[j] += '+' + present_dwell
                    else:
                        flag = 1

                    if flag == 0:
                        f1.write(' '.join(session) + '\n')
                        count += 1

                    sessDur = (sessEnd - sessStart)
                    set_sessDur.add(sessDur)

                    if linenum % 100000 == 0:
                        print linenum, count
                    linenum += 1
                except:
                    continue
        f1.close()

        #wrte log
        logStr = []

        logStr.append("\nid:")
        str_tmp = "id count:%d, min:%d, max:%d" % (len(set_id), min(set_id), max(set_id))
        logStr.append(str_tmp)

        logStr.append("\nbh:")
        logStr.append(StructureTool.format_list_to_str_conn(list(set_bh)))

        logStr.append("\ncid3:")
        logStr.append(StructureTool.format_list_to_str_conn(list(set_cid3)))

        logStr.append("\nsessDur:")
        str_tmp = "dur min:%d, max:%d" % (min(set_sessDur), max(set_sessDur))
        logStr.append(str_tmp)

        FileTool.printList(logStr, file_log)


    @classmethod
    def check(cls, file_in, file_out):

        pass

    @classmethod
    def check_data(cls):

        Preprocess.check_sku_cid3()
        pass

    @classmethod
    def check_sku_cid3(cls):
        micro_item_id = Data.load_micro_itemInt_idInt(["sku"])
        micro_sku_id = micro_item_id["sku"]
        dict_sku_cid3 = Data.load_sku_cid3()

        i = 0
        for sku in micro_sku_id.keys():
            if(sku not in dict_sku_cid3):
                print "Error! sku: %d not in sku_cid3" % sku
                i += 1
        print "Error count:%d" % i

        pass

    @classmethod
    def get_data_base_data_micro_item(cls):
        Preprocess.get_data_base_data_bh_dwell()

    @classmethod
    def get_data_base_data_bh_dwell(cls):
        FileTool.func_begin("get_data_base_data_bh_dwell")
        dict_bhdwll_idInt_itemStr=Data.load_mapping_dict_itemStr_idInt("bhdwell", True)
        dict_bhdwll_idInt_emb = Data.load_reidx_dict_idInt_emb("bhdwell")

        dict_bhStr_emb={}
        dict_dwellStr_emb={}
        for bhdwll_idInt in dict_bhdwll_idInt_itemStr.keys():
            itemStr = dict_bhdwll_idInt_itemStr[bhdwll_idInt]
            itemEmb = dict_bhdwll_idInt_emb[bhdwll_idInt]
            itemArr = itemStr.split('+')
            (bhStr, dwellStr) = (itemArr[1], itemArr[2])
            bhEmb = itemEmb[30:35]
            dwellEmb = itemEmb[35:38]
            dict_bhStr_emb[bhStr] = bhEmb
            dict_dwellStr_emb[dwellStr] = dwellEmb

        dict_bhStr_id = {}
        dict_dwellStr_id = {}
        bhId = 1
        dwellId = 1
        dict_bhId_emb = {}
        dict_dwellId_emb = {}

        dict_bhId_emb[0] = [0.] * Config.dict_emb_item_size["bh"]
        dict_dwellId_emb[0] = [0.] * Config.dict_emb_item_size["dwell"]


        for bhStr in sorted(dict_bhStr_emb.keys()):
            if(bhStr not in dict_bhStr_id):
                dict_bhStr_id[bhStr] = bhId
                bhId += 1
                dict_bhId_emb[bhId] = dict_bhStr_emb[bhStr]

        for dwellStr in sorted(dict_dwellStr_emb.keys()):
            if(dwellStr not in dict_dwellStr_id):
                dict_dwellStr_id[dwellStr] = dwellId
                dwellId += 1
                dict_dwellId_emb[dwellId] = dict_dwellStr_emb[dwellStr]

        file_mapping_bh = Config.add_folder("base_bh.mapping")
        file_mapping_dwell = Config.add_folder("base_dwell.mapping")

        file_emb_bh =  Config.add_folder("base_bh.reidx")
        file_emb_dwell = Config.add_folder("base_dwell.reidx")

        FileTool.printDict(dict_bhStr_id, file_mapping_bh)
        FileTool.printDict(dict_dwellStr_id, file_mapping_dwell)

        FileTool.printDictEmb(dict_bhId_emb, file_emb_bh)
        FileTool.printDictEmb(dict_dwellId_emb, file_emb_dwell)

        FileTool.func_end("get_data_base_data_bh_dwell")

        pass
    @classmethod
    def convert_raw_time_to_id(cls, folder, file_in, file_out, type=None):
        file_in = FileTool.add_folder(folder, file_in)
        file_out = FileTool.add_folder(folder, file_out)

        fw = open(file_out, 'w')
        with open(file_in, 'r') as f:
            for l in f:
                tt = l.strip().split()
                ttnew = []
                for t in tt:
                    tnew = t
                    if (type == "dwell"):
                        tnew = Config.dwell2id(t)
                    if (type == "gap"):
                        tnew = Config.gap2id(t)

                    ttnew.append(tnew)
                fw.write(' '.join(ttnew) + '\n')

        fw.close()

    @classmethod
    def get_data_raw_data_norm(cls, file_in, file_out):

        fout = open(file_out, 'w')
        with open(file_in, "r") as f:
            for l in f:
                events = l.strip().split(' ')
                listNew = []
                for e in events:
                    sessionItem = SessionItemBase(e)
                    if(Config.get_exp_label() != "tianchi"):
                        sessionItem.normItem()
                    itemNew = sessionItem.toNormString(mode="SBCD")
                    listNew.append(itemNew)
                strNew = ' '.join(listNew)
                fout.write(strNew + '\n')
                pass
        fout.close()

    @classmethod
    def filt_item_raw_to_uniq(cls, folder):
        FileTool.func_begin("filt_item_raw_to_uniq")
        fsku=FileTool.add_folder(folder, 'sku.raw')
        fsku_uniq = FileTool.add_folder(folder, 'sku.uniq')
        FileTool.get_file_line_uniq_item(fsku, fsku_uniq, Config.file_sep)
        FileTool.func_end("filt_item_raw_to_uniq")

    @classmethod
    def get_item_raw_from_session_data(cls, folder, file_in, min_cnt, max_cnt):
        file_in = FileTool.add_folder(folder, file_in)

        if(min_cnt >= 0):
            file_in = file_in + str(min_cnt)
        if(max_cnt >= 0):
            file_in = file_in + "_" + str(max_cnt)

        print "get_item_raw_from_session_data %s..." % file_in

        fsku = open(FileTool.add_folder(folder, 'sku.raw'), 'w')
        fbh = open(FileTool.add_folder(folder, 'bh.raw'), 'w')
        fcid3 = open(FileTool.add_folder(folder, 'cid3.raw'), 'w')
        fgap = open(FileTool.add_folder(folder, 'gap.raw'), 'w')
        fdwell = open(FileTool.add_folder(folder, 'dwell.raw'), 'w')

        with open(file_in, "r") as f:
            cnt = 0
            for l in f:
                events = l.strip().split(' ')
                sku = ''
                bh = ''
                cid3 = ''
                gap = ''
                dwell = ''
                for e in events:
                    token = e.split('+')
                    sku += token[0] + ' '
                    bh += token[1] + ' '
                    cid3 += token[2] + ' '
                    gap += token[3] + ' '
                    dwell += token[4] + ' '

                fsku.write(sku + '\n')
                fbh.write(bh + '\n')
                fcid3.write(cid3 + '\n')
                fgap.write(gap + '\n')
                fdwell.write(dwell + '\n')

                cnt += 1
                FileTool.print_info_pro(cnt)

        fsku.close()
        fbh.close()
        fcid3.close()
        fgap.close()
        fdwell.close()

        print "get_item_raw_from_session_data done!"

    @classmethod
    def get_item_raw_from_session_data_tianchi(cls, folder, file_in):

        print "get_item_raw_from_session_data %s..." % file_in
        file_in = FileTool.add_folder(folder, file_in)

        fsku = open(FileTool.add_folder(folder, 'sku.raw'), 'w')
        fbh = open(FileTool.add_folder(folder, 'bh.raw'), 'w')
        fcid3 = open(FileTool.add_folder(folder, 'cid3.raw'), 'w')
        fgap = open(FileTool.add_folder(folder, 'gap.raw'), 'w')
        fdwell = open(FileTool.add_folder(folder, 'dwell.raw'), 'w')

        with open(file_in, "r") as f:
            cnt = 0
            for l in f:
                events = l.strip().split(' ')
                sku = ''
                bh = ''
                cid3 = ''
                gap = ''
                dwell = ''
                for e in events:
                    token = e.split('+')
                    sku += token[0] + ' '
                    bh += token[1] + ' '
                    cid3 += token[2] + ' '
                    gap += token[3] + ' '
                    dwell += token[4] + ' '

                fsku.write(sku + '\n')
                fbh.write(bh + '\n')
                fcid3.write(cid3 + '\n')
                fgap.write(gap + '\n')
                fdwell.write(dwell + '\n')

                cnt += 1
                FileTool.print_info_pro(cnt)

        fsku.close()
        fbh.close()
        fcid3.close()
        fgap.close()
        fdwell.close()

        print "get_item_raw_from_session_data done!"

    @classmethod
    def convert_raw_time_to_id_files(cls, folder, gap=True):
        FileTool.func_begin("convert_raw_time_to_id_files")
        Preprocess.convert_raw_time_to_id(folder, 'dwell.raw', 'dwell.id', "dwell")
        if(gap):
            Preprocess.convert_raw_time_to_id(folder, 'gap.raw', 'gap.id', "gap")
        FileTool.func_end("convert_raw_time_to_id_files")


    @classmethod
    def get_data_raw_data_unique(cls, folder, file_in, file_out):
        FileTool.func_begin("get_data_raw_data_unique")
        file_in = FileTool.add_folder(folder, file_in)
        file_out = FileTool.add_folder(folder, file_out)

        print "file_in: %s" % file_in
        print "file_out: %s" % file_out

        fw = open(file_out, 'w')
        with open(file_in, 'r') as f:
            for l in f:
                events = l.strip().split()
                previous = events[0].split('+')
                w = []
                for i in range(1, len(events)):
                    now = events[i].split('+')
                    if now[0] != previous[0] or now[1] != previous[1]:
                        w.append('+'.join(previous))
                    previous = now
                w.append('+'.join(now))
                fw.write(' '.join(w) + '\n')

        fw.close()
        FileTool.func_end("get_data_raw_data_unique")
        pass


    @classmethod
    def get_embedding_item(cls, folder, file_in, file_out, size, min_count=0):
        file_in = os.path.join(folder, file_in)
        file_out = os.path.join(folder, file_out)

        cmd = "../word2vec/word2vec -train %s -output %s -size %s -min-count %s" % (file_in, file_out, str(size), str(min_count))

        ProTool.exec_cmd(cmd)
        pass

    @classmethod
    def get_embedding_items(cls):
        Preprocess.get_embedding_item(Config.folder, "sku.uniq", "sku.w2v", Config.dict_emb_item_size["sku"], Config.min_cnt_sku[0])
        Preprocess.get_embedding_item(Config.folder, "bh.raw", "bh.w2v", Config.dict_emb_item_size["bh"])
        Preprocess.get_embedding_item(Config.folder, "cid3.raw", "cid3.w2v", Config.dict_emb_item_size["cid3"])
        Preprocess.get_embedding_item(Config.folder, "dwell.id", "dwell.w2v", Config.dict_emb_item_size["dwell"])
        Preprocess.get_embedding_item(Config.folder, "gap.id", "gap.w2v", Config.dict_emb_item_size["gap"])
        pass

    @classmethod
    def get_data_item_mapping_items(cls):
        print "get_data_item_mapping_items..."
        for type_item in Config.micro_item_list:
            Preprocess.get_data_item_mapping(type_item)

    @classmethod
    def get_data_item_mapping(cls, type_item):
        print "get_data_item_mapping %s..." % type_item
        # src file
        file_w2v = type_item + ".w2v"

        # dst file
        file_mapping = type_item + ".mapping"
        file_reidx = type_item + ".reidx"

        #
        emb_size = Config.dict_emb_item_size[type_item]

        #
        Preprocess.get_data_item_mapping_file(Config.folder, file_w2v, file_mapping, file_reidx, emb_size)

        pass

    @classmethod
    def get_data_item_mapping_file(cls, folder, file_w2v, file_mapping, file_reidx, emb_size):
        file_w2v = FileTool.add_folder(folder, file_w2v)
        file_mapping = FileTool.add_folder(folder, file_mapping)
        file_reidx = FileTool.add_folder(folder, file_reidx)

        # item embedding by id increase(0, 1,...)
        f1 = open(file_reidx, 'w')
        f1.write(' '.join(['0'] * emb_size) + '\n')

        # sku => id
        f2 = open(file_mapping, 'w')
        i = 0
        j = 1

        with open(file_w2v, "r") as f:
            for l in f:
                if i > 1:
                    events = l.strip().split()

                    # sku => id
                    f2.write(events[0] + ' ' + str(j) + '\n')
                    j += 1

                    # norm w2v vector value
                    tmp = []
                    for k in events[1:]:
                        tmp.append(float(k))
                    mochang = linalg.norm(np.array(tmp))
                    tmp = tmp / mochang

                    w = []
                    for k in tmp:
                        w.append(str(k))
                    f1.write(' '.join(w) + '\n')

                i += 1

        f1.close()
        f2.close()

    @classmethod
    def get_data_session_train_filt_data_to_itemid(cls, file_in, mode):
        print "get_data_session_train_filt_data_to_itemid %s" % file_in

        file_in = Preprocess.add_folder_file(file_in)

        #dict_sessionItem_to_id = {}
        sessionItem_id = 1

        micro_item_list = Config.get_micro_item_list(mode)
        print "micro_item_list:", micro_item_list

        micro_item2vec = Data.load_micro_item_vec_mode(mode)
        micro_item2id = Data.load_micro_itemInt_idInt(micro_item_list)

        list_lines = []
        cnt = 0
        with open(file_in, 'r') as f:
            for l in f:
                events = l.strip().split()
                w = []

                for e in events:
                    sessionItem = SessionItemBase(e)
                    sessionItem.normItem()
                    #print "e:", e
                    # check valid
                    flagItemValid = sessionItem.checkItemAllPartHasItemVec(micro_item2vec, micro_item_list)

                    if not flagItemValid:
                        continue

                    try:
                        #print "valid\n"
                        sessionItem_idStr = sessionItem.toIdString(micro_item2id, micro_item_list)

                        #if(sessionItem_idStr not in dict_sessionItem_to_id):
                        #    dict_sessionItem_to_id[sessionItem_idStr] = sessionItem_id
                        #    sessionItem_id += 1

                        curAppend = sessionItem_idStr
                        #curAppend = str(dict_sessionItem_to_id[sessionItem_idStr])
                        w.append(curAppend)
                    except:
                        print "excepttion: sessionItem.toIdString"

                if (len(w) > 0):
                    list_lines.append(w)

                cnt += 1
                FileTool.print_info_pro(cnt)

                #if(cnt > 100):
                #    break

        print "get_data_session_train_filt_data_to_itemid done!"
        return list_lines


    @classmethod
    def get_file_to_id_mapping(cls, folder, file_in, file_out, file_out_mapping, sep=' '):
        file_in = FileTool.add_folder(folder, file_in)
        file_out = FileTool.add_folder(folder, file_out)
        file_out_mapping = FileTool.add_folder(folder, file_out_mapping)

        print "get_file_to_id_mapping..."
        item2id = {}
        itemCnt = 1

        lineList = []
        cnt = 0
        with open(file_in, 'r') as fin:
            for line in fin:
                items = line.strip().split(sep)

                arr = []
                for item in items:
                    if item == "":
                        continue
                    if item not in item2id:
                        item2id[item] = itemCnt
                        itemCnt += 1
                    arr.append(str(item2id[item]))

                lineNew = sep.join(arr)
                lineList.append(lineNew)

                cnt+=1
                FileTool.print_info_pro(cnt)

        FileTool.printList(lineList, file_out)
        FileTool.printDict(item2id, file_out_mapping)

        pass


    @classmethod
    def build_raw_data(cls):
        print "build_raw_data..."

        print "processing %s..." % Config.get_exp_label()

        Preprocess.get_topsku(Config.folder, Config.file_data_raw, Config.topsku, Config.file_data_raw_topsku, Config.min_cnt_sku_limit)
        Preprocess.get_lines_min_cnt(Config.folder, Config.file_data_raw_topsku, Config.file_data_raw_topsku_len, Config.min_cnt_line_items, Config.max_cnt_line_items)
        Preprocess.get_item_raw_from_session_data(Config.folder, Config.file_data_raw_topsku_len, Config.min_cnt_line_items, Config.max_cnt_line_items)
        Preprocess.filt_item_raw_to_uniq(Config.folder)
        Preprocess.convert_raw_time_to_id_files(Config.folder)

        pass

    @classmethod
    def get_data_statis(cls):
        #FileTool.get_file_line_seqlen_distribution(Config.folder, 'u10.raw.uniq', 'u10.raw.uniq_distri')
        #FileTool.get_file_item_distribution(Config.folder, 'sku.raw', 'sku.raw_distribution')
        Preprocess.get_data_desc(Config.folder, Config.file_data_raw, Config.file_data_raw+"_ana")
        pass

    @classmethod
    def get_data_desc(cls, folder, file_in, file_out):
        FileTool.func_begin("get_data_desc")
        file_in = FileTool.add_folder(folder, file_in)
        file_out = FileTool.add_folder(folder, file_out)

        set_sku = set()
        set_bh = set()
        set_cid3 = set()
        list_gap = []
        list_dwell = []

        list_itemCnt = []

        list_timeCnt = []
        sessionCnt=0

        listlist = FileTool.read_file_to_list_list(file_in)


        microCnt = 0
        list_res = []
        for line in listlist:
            cnt = len(line)
            sessionCnt += 1
            microCnt += cnt
            list_itemCnt.append(cnt)

            seconds = 0
            for item in line:
                sessionItem = SessionItemBase(item)
                sku = sessionItem.sku
                bh = sessionItem.bh
                cid3 = sessionItem.cid3
                gap = int(sessionItem.gap)
                dwell = int(sessionItem.dwell)

                set_sku.add(sku)
                set_bh.add(bh)
                set_cid3.add(cid3)

                if(gap < 86400):
                    list_gap.append(int(gap))
                    seconds += int(gap)

                if(dwell < 60*10):
                    list_dwell.append(int(dwell))
                    seconds += int(dwell)


            list_timeCnt.append(seconds)


        list_res.append("session: %d" % sessionCnt)
        list_res.append("sku: %d" % len(set_sku))
        list_res.append("bh: %d" % len(set_bh))
        list_res.append("cid3: %d" % len(set_cid3))

        list_res.append("microCnt: %d" % microCnt)


        arr_gap = np.array(list_gap)
        list_res.append("gap: min:%d, max:%d, avg:%f" % (np.min(arr_gap), np.max(arr_gap), np.average(arr_gap)))

        arr_dwell = np.array(list_dwell)
        list_res.append("dwell: min:%d, max:%d, avg:%f" % (np.min(arr_dwell), np.max(arr_dwell), np.average(arr_dwell)))

        arr_itemCnt = np.array(list_itemCnt)
        list_res.append("itemCnt: min:%d, max:%d, avg:%f, total:%d" % (np.min(arr_itemCnt), np.max(arr_itemCnt), np.average(arr_itemCnt), np.sum(arr_itemCnt)))

        arr_timeCnt = np.array(list_timeCnt)
        list_res.append("timeCnt: min:%d, max:%d, avg:%f" % (np.min(arr_timeCnt), np.max(arr_timeCnt), np.average(arr_timeCnt)))

        FileTool.write_file_listStr(file_out, list_res)

        FileTool.func_end("get_data_desc")

    @classmethod
    def get_data_item_mapping_reverse(cls):
        FileTool.get_file_reverse_col(Config.add_folder('bhdwell.mapping'), Config.add_folder('bhdwell.mapping.reverse'))

    @classmethod
    def get_topsku(cls, folder, file_in, file_out_topsku, file_out_lines, min_sku_cnt):
        FileTool.func_begin("get_topsku")
        listlist = FileTool.read_file_to_list_list(os.path.join(folder, file_in), Config.file_sep)

        dict_sku = {}
        for line in listlist:
            for unit in line:
                item = SessionItemBase(unit)
                sku = item.sku
                StructureTool.addDict(dict_sku, sku)

        print "dict_sku:", len(dict_sku)

        valid_sku=set()
        for sku in dict_sku:
            if(dict_sku[sku] >= min_sku_cnt):
                valid_sku.add(sku)

        valid_sku_list = list(valid_sku)

        print "valid_sku: ", len(valid_sku)
        FileTool.write_file_listStr(os.path.join(folder, file_out_topsku), valid_sku_list)

        valid_lines = []
        valid_lines_len = []
        for line in listlist:
            flag=True
            if(Config.get_exp_label() != "tianchi"):
                for unit in line:
                    item = SessionItemBase(unit)
                    sku = item.sku
                    if(sku not in valid_sku):
                        flag=False
                        break
                if(flag):
                    valid_lines.append(line)
            else:
                newLine = []
                for unit in line:
                    item = SessionItemBase(unit)
                    sku = item.sku
                    if(sku  in valid_sku):
                        newLine.append(unit)
                if(len(newLine) >= Config.min_cnt_line_items_top):
                    valid_lines.append(newLine)

        print "valid_lines: ", len(valid_lines)
        FileTool.write_file_list_list(os.path.join(folder, file_out_lines), valid_lines, Config.file_sep)

        FileTool.func_end("get_topsku")

    @classmethod
    def get_lines_min_cnt(cls, folder, file_in, file_out, min_cnt, max_cnt):
        FileTool.func_begin("get_lines_min_cnt")

        file_in = os.path.join(folder, file_in)
        file_out = os.path.join(folder, file_out)
        if(min_cnt >= 0):
            file_out = file_out + str(min_cnt)
        if(max_cnt >= 0):
            file_out = file_out + "_" + str(max_cnt)


        print "file_out:", file_out

        FileTool.filt_file_by_lineItemCnt(file_in, file_out, Config.file_sep, min_cnt, max_cnt)
        FileTool.func_end("get_lines_min_cnt")

    @classmethod
    def get_recall_topsku(cls):
        cmd="python -u emb_sim_1000recall_matrix.py"
        ProTool.exec_cmd(cmd)




