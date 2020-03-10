class StructureTool():
    @classmethod
    def addDict(cls, dict, k):
        if(k not in dict):
            dict[k] = 1
        else:
            dict[k] += 1

    @classmethod
    def format_list_to_int(cls, l):
        res = [int(i) for i in l]
        return res

    @classmethod
    def format_list_to_str(cls, l):
        res = [str(i) for i in l]
        return res

    @classmethod
    def format_list_to_str_conn(cls, l, conStr=','):
        l = StructureTool.format_list_to_str(l)
        res = conStr.join(l)
        return res


    @classmethod
    def get_dict_to_list_str(cls, dict):
        res = []
        for key in sorted(dict.keys()):
            value = dict[key]
            line = "%s %s" % (str(key), str(value))
            res.append(line)
        return res

    @classmethod
    def filt_list_by_item(cls, list, itemFilt=""):
        if(itemFilt == ""):
            return list

        list = [item for item in list if (item != itemFilt)]

        return list


    @classmethod
    def format_listStr_to_listList(cls, listStr, sep=' ', lineItemFilt=""):
        listlist = []
        for str in listStr:
            arr = str.split(sep)
            arr = StructureTool.filt_list_by_item(arr, lineItemFilt)
            listlist.append(arr)
        return listlist

    @classmethod
    def format_listlist_to_list(cls, listlist, sep=' '):
        res = []
        for list in listlist:
            list = StructureTool.format_list_to_str(list)
            line = sep.join(list)
            res.append(line)
        return res

    @classmethod
    def get_list_distribution(cls, list):
        dictCnt = {}
        for item in list:
            StructureTool.addDict(dictCnt, item)

        return dictCnt

    @classmethod
    def uniq_list(cls, ss):
        last_s = ""
        ss1 = []
        for s in ss:
            if s != last_s:
                ss1.append(s)
                last_s = s
        return ss1

    @classmethod
    def split_list_by_maxlen(cls, list, maxLen, step):
        res = []
        cnt = len(list)
        start = 0
        while (start < cnt):
            if (start + maxLen > cnt):
                break
            cur = list[start:start + maxLen]
            res.append(cur)
            start += step
            break

        return res

