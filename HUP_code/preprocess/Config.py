import os

class Config:

	#file raw data by sku, line item cnt
	min_cnt_sku_limit=50
	min_cnt_line_items=40
	max_cnt_line_items=-1

	train_ratio=0.7
	folder_base = '/export/sdb/home/guyulong/program/HUP/HUP_Data/'

	#670: JD_Computers,  737:JD_Applicances
	data_source_name=['Computers', 'Applicances']

	#select dataset
	data_source_index=0
	data_source=data_source_name[data_source_index]
	seq_len_min = 40
	seq_len_max = 40

	min_cnt_sku = [0, 15]

	topsku='topsku'
	#670: Computers
	if(data_source == "Computers"):
		folder_data = 'Computers'
		file_data_raw = 'JD_Computers'
		file_data_raw_topsku = 'JD_Computers.topsku'
		file_data_raw_topsku_len = 'JD_Computers.topsku.len'
		file_data_src = "JD_Computers.topsku.len30"
		train_ratio = 0.7
		min_cnt_line_items = 30
		seq_len_min = 30
		seq_len_max = 30



	#737: Applicances
	if (data_source == "Applicances"):
		#folder_data = 'input_737'
		folder_data = 'Applicances'
		file_data_raw ='JD_Applicances'
		file_data_raw_topsku = 'JD_Applicances.topsku'
		file_data_raw_topsku_len = 'JD_Applicances.topsku.len'
		file_data_src = 'JD_Applicances.topsku.len40'
		train_ratio = 0.7
		min_cnt_line_items = 40
		seq_len_min = 40


	seq_len = seq_len_max
	seq_split_step = 1


	folder = os.path.join(folder_base, folder_data)

	dict_emb_item_size = {
		"sku": 30,
		"bh": 5,
		"cid3": 8,
		"dwell": 5,
		"gap": 5
	}

	seq_len_str = '_'.join(["len", str(seq_len_min), str(seq_len_max), str(seq_split_step)])

	micro_item_list = ["sku", "bh", "cid3", "dwell", "gap"]

	micro_item_cnt = len(micro_item_list)

	file_sep = ' '
	paddingStr = '0'
	file_pickle="_pickle"

	@classmethod
	def add_folder(cls, file):
		res = os.path.join(Config.folder, file)
		return res

	@classmethod
	def add_folder_fileList(cls, fileList):
		resList = [Config.add_folder(file) for file in fileList]
		return resList

	@classmethod
	def behavior2id(cls, b):
		if b == 'Home_Productid':
			return '1'
		if b == 'ShopList_Productid':
			return '2'
		if b == 'HandSeckill_Productid':
			return '3'
		if b == 'Shopcart_Productid':
			return '4'
		if b == 'Searchlist_Productid':
			return '5'
		if b == 'Productdetail_CommentsPic' or b == 'Productdetail_Comment':
			return '6'
		if b == 'Productdetail_Specification' or b == 'Productdetail_DetailTab':
			return '7'
		if b == 'Productdetail_ButtomProinfo':
			return '8'
		if b == 'Productdetail_Addtocart':
			return '9'
		if b == 'order':
			return '10'

	@classmethod
	def dwell2id(cls, input):
		t = int(input)
		exp_label = Config.get_exp_label()

		if(exp_label == "tianchi"):
			tnew = int((t+1)/2)
		else:
			if t< 15:
				tnew = '1'
			elif t< 40:
				tnew = '2'
			elif t< 97:
				tnew = '3'
			elif t< 600:
				tnew = '4'
			else:
				tnew = '5'
		return str(tnew)

	@classmethod
	def gap2id(cls, input):
		d = int(input)

		exp_label = Config.get_exp_label()

		if(exp_label == "tianchi"):
			dnew = int((d+1)/5)
			return str(dnew)

		else:
			if d >= 0 and d<=1:
				return '1'
			if d >= 2 and d<=15:
				return '2'
			if d >= 16 and d<=39:
				return '3'
			if d >= 40 and d<=90:
				return '4'
			if d > 90:
				return '5'

	@classmethod
	def get_micro_item_list(cls, mode="SBCGD"):
		res = []
		if("S" in mode):
			res.append("sku")
		if("B" in mode):
			res.append("bh")
		if("C" in mode):
			res.append("cid3")
		if("G" in mode):
			res.append("gap")
		if("D" in mode):
			res.append("dwell")

		#if(mode == "SBCD"):
		#    res = micro_item_list
		return res

	@classmethod
	def get_micro_item_file_list(cls, micro_item_list):
		res = []
		for micro_item in micro_item_list:
			file = micro_item + '.reidx'
			file = Config.add_folder(file)
			res.append(file)
		return res

	@classmethod
	def get_micro_item_file_list_w2v(cls, micro_item_list):
		res = []
		for micro_item in micro_item_list:
			file = micro_item + '.w2v'
			file = Config.add_folder(file)
			res.append(file)
		return res

	@classmethod
	def get_item_emb_len(cls, mode):
		res = 0
		if("S" in mode):
			res += Config.dict_emb_item_size["sku"]
		if("B" in mode):
			res += Config.dict_emb_item_size["bh"]
		if("C" in mode):
			res += Config.dict_emb_item_size["cid3"]
		if("G" in mode):
			res += Config.dict_emb_item_size["gap"]
		if ("D" in mode):
			res += Config.dict_emb_item_size["dwell"]
		return res

	@classmethod
	def get_exp_label(cls):
		if (Config.data_source == "Computers"):
			return "Computers"
		if (Config.data_source == "Applicances"):
			return "Applicances"

