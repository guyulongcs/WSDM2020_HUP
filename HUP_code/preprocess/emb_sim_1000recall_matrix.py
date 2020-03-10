import pandas as pd
import numpy as np
import time
from numpy import *
import gc

from Config import *
from ProTool import *
N = 1000

init_wgt_path = 'sku.reidx'
file_topSku = "top%dsku" % N

file_in = Config.add_folder(init_wgt_path)
file_out = Config.add_folder(file_topSku)

print "Calculate Recall SKU By Embedding: top %d..." % N

print "file_in:", file_in
print "file_out:", file_out

w = np.loadtxt(file_in, dtype=np.float16)
#w = np.loadtxt(file_in)


print 'load finish!'

start_time=time.time()

def get_time_info(start_time):
    cur_time = time.time()
    cur_time_str = ProTool.timestamp_toString(cur_time)
    print "cur_time:", cur_time_str
    ProTool.get_time_interval(start_time, cur_time)


print "matrix mul..."
r = np.matmul(w, w.transpose())

del w
gc.collect()

print "matrix sort..."
#res = np.argsort(-r, axis=1)[:, :N]
res = (-r).argsort(axis=1)[:,:N]

print "res:"
print res.shape

print "top N out..."

np.savetxt(file_out, res, fmt='%s', delimiter=' ')

get_time_info(start_time)
print "write top file done!"
