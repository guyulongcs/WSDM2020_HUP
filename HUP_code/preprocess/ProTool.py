import os
import time
from dateutil.parser import parse

class ProTool():
    @classmethod
    def exec_cmd(cls, cmd):
        print "Cmd:%s" % cmd
        os.system(cmd)
        print "exec cmd done!"

    @classmethod
    def timestamp_toString(cls, stamp):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stamp))

    @classmethod
    def get_time_interval(cls, start_time, end_time):
        secs = (end_time - start_time)
        timeStr = time.strftime("%H H:%M M:%S s", time.gmtime(secs))
        print "Used time:%s" % timeStr

    @classmethod
    def CopyFile(cls, folder_src, file_src, folder_dst, file_dst):
        file_src = os.path.join(folder_src, file_src)
        file_dst = os.path.join(folder_dst, file_dst)
        cmd = "cp %s %s" % (file_src, file_dst)
        ProTool.exec_cmd(cmd)

    @classmethod
    def get_time_from_string(cls, str):
        dt=time.strptime(str, '%Y-%m-%d %H:%M:%S')
        return dt


    @classmethod
    def get_time_interval_hour(cls, start_time_str, end_time_str):

        start = parse(start_time_str)
        end = parse(end_time_str)
        res_seconds = (end-start).total_seconds()


        if(res_seconds > 0):
            res_hour = res_seconds / 3600
            res = int(res_hour)
        else:
            res=1

        #print "start:%s, end:%s, res_seconds:%s, res:%s" % (start_time_str, end_time_str, res_seconds, res)
        return res
