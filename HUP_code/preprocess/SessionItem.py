from Config import *

class SessionItem():
    def __init__(self, str=''):
        self.sku = '0'
        self.bh = '0'
        self.cid3 = '0'
        self.gap='0'
        self.dwell = '0'

        self.item_str = str

        self.mode = ''
        arr = self.item_str.split('+')
        if (len(arr) == 5):
            (self.sku, self.bh, self.cid3, self.gap, self.dwell) = (arr[0], arr[1], arr[2], arr[3], arr[4])
        if (len(arr) == 4):
            (self.sku, self.bh, self.cid3, self.dwell) = (arr[0], arr[1], arr[2], arr[3])
        if (len(arr) == 3):
            (self.sku, self.bh, self.dwell) = (arr[0], arr[1], arr[2])

        pass

    def set_value(self, str, mode='SBD'):
        self.mode = mode
        arr = str.split('+')

        assert(len(arr) == len(mode))

        if (mode == "S" and len(arr) == 1):
            self.sku = arr[0]

        if (mode == "C" and len(arr) == 1):
            self.cid3 = arr[0]

        if(mode == "SBD" and len(arr) == 3):
            self.sku = arr[0]
            self.bh = arr[1]
            self.dwell = arr[2]
        if(mode == "SBCD" and len(arr) == 4):
            self.sku = arr[0]
            self.bh = arr[1]
            self.cid3 = arr[2]
            self.dwell = arr[3]
        if (mode == "SBCGD" and len(arr) == 5):
            self.sku = arr[0]
            self.bh = arr[1]
            self.cid3 = arr[2]
            self.gap = arr[3]
            self.dwell = arr[4]


    def getDict(self):
        dict = {}
        dict["sku"] = self.sku
        dict["bh"] = self.bh
        dict["cid3"] = self.cid3
        dict["gap"] = self.gap
        dict["dwell"] = self.dwell
        return dict

    def get_subId(self, mode="S"):
        res = ""
        list = []
        if(mode == "S"):
            list = [self.sku]
        if (mode == "C"):
            list = [self.cid3]
        elif(mode == "SBD"):
            list = [self.sku, self.bh, self.dwell]
        elif(mode == "SBCD"):
            list = [self.sku, self.bh, self.cid3, self.dwell]
        elif (mode == "SBCGD"):
            list = [self.sku, self.bh, self.cid3, self.gap, self.dwell]
        res = '+'.join(list)
        return res

    def get_Id(self, dict_item_itemInt_idInt, mode="SBCD"):
        res = ""
        list = []

        sku = dict_item_itemInt_idInt["sku"][int(self.sku)]
        bh = dict_item_itemInt_idInt["base_bh"][int(self.bh)]
        cid3 = dict_item_itemInt_idInt["cid3"][int(self.cid3)]
        gap = dict_item_itemInt_idInt["gap"][int(self.gap)]
        dwell = dict_item_itemInt_idInt["base_dwell"][int(self.dwell)]

        if(mode == "SBCD"):
            list = [str(sku), str(bh), str(cid3), str(dwell)]
        if (mode == "SBCGD"):
            list = [str(sku), str(bh), str(cid3), str(gap), str(dwell)]

        res = '+'.join(list)
        return res

    def getEmb(self, micro_item_vec, mode="SBD"):
        res = []

        if("S" in mode):
            emb_sku = micro_item_vec["sku"][int(self.sku)]
            res.extend(emb_sku)
        if("B" in mode):
            emb_bh = micro_item_vec["bh"][int(self.bh)]
            res.extend(emb_bh)
        if ("C" in mode):
            emb_cid3 = micro_item_vec["cid3"][int(self.cid3)]
            res.extend(emb_cid3)
        if ("G" in mode):
            emb_gap = micro_item_vec["gap"][int(self.gap)]
            res.extend(emb_gap)
        if("D" in mode):
            emb_dwell = micro_item_vec["dwell"][int(self.dwell)]
            res.extend(emb_dwell)

        return res
