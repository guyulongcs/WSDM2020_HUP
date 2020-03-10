from Config import *

class SessionItemBase():
    def __init__(self, str=''):
        self.sku = ''
        self.bh = ''
        self.cid3 = ''
        self.gap = ''
        self.dwell = ''

        self.item_str = str

        arr = self.item_str.split('+')
        #print "arr:", arr
        if (len(arr) == 5):
            (self.sku, self.bh, self.cid3, self.gap, self.dwell) = (arr[0], arr[1], arr[2], arr[3], arr[4])

        if (len(arr) == 4):
            (self.sku, self.bh, self.cid3, self.dwell) = (arr[0], arr[1], arr[2], arr[3])
        pass

    def normItem(self):
        self.gap = Config.gap2id(self.gap)
        self.dwell = Config.dwell2id(self.dwell)

    def toNormString(self, mode="SBCD"):
        str = ''
        if (mode == 'SBCGD'):
            str = '+'.join([self.sku, self.bh, self.cid3, self.gap, self.dwell])
        if(mode == 'SBCD'):
            str = '+'.join([self.sku, self.bh, self.cid3, self.dwell])
        if(mode == "SBD"):
            str = '+'.join([self.sku, self.bh, self.dwell])
        return str

    def checkItemAllPartHasItemVec(self, item_vec, micro_item_list):
        flag = True
        #print "sku:", self.sku

        flag_sku = ("sku" not in micro_item_list) or (self.sku in item_vec["sku"])
        if(not flag_sku):
            print "sku", self.sku
        flag_bh = ("bh" not in micro_item_list) or (self.bh in item_vec["bh"])
        if (not flag_bh):
            print "bh:", self.bh
        flag_cid3 = ("cid3" not in micro_item_list) or (self.cid3 in item_vec["cid3"])
        if (not flag_cid3):
            print "cid3:", self.cid3
        flag_dwell = ("dwell" not in micro_item_list) or (self.dwell in item_vec["dwell"])
        if (not flag_dwell):
            print "dwell:", self.dwell
        flag_gap = ("gap" not in micro_item_list) or (self.gap in item_vec["gap"])
        if (not flag_gap):
            print "gap:", self.gap

        flag = flag_sku and flag_bh and flag_cid3 and flag_dwell and flag_gap

        if(not flag):
            print "unvalid!"

        return flag

    def toIdString(self, item_id, micro_item_list):

        dictCur={
            "sku": self.sku,
            "bh": self.bh,
            "cid3": self.cid3,
            "gap": self.gap,
            "dwell": self.dwell
        }

        resList = []

        for item in micro_item_list:
            tmp = item_id[item][int(dictCur[item])]
            resList.append(str(tmp))
        res = '+'.join(resList)

        return res
        pass
