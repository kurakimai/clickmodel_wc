import sys, os, re, urllib, math, random
import urllib

from wc_common import *

root_dir = "D:/work/sogou/MOUSE"
session_file_name = root_dir + "/data/mouse_log_10000.sample"
train_percentage = 0.7
K = 1
D = 1
result_num = 10
featureList = [ "url", "left_dx", "horizontal_move_right", "v_fix_time", "fix_time", "hover_time", "action_num", "rank", "isexam", "isclick", "clicktime", "vrid" ]

out_dir = "../data/sample_1000"
if not os.path.exists(out_dir):
    os.system("mkdir " + out_dir)

feature_sep = "\t"
inner_sep = " "
QUERY_MAX_SESSION = 10000

random.seed(1)

query_list = []
query_class_map = {}
query_train_count_map = {}
query_test_count_map = {}

load_query_count = 0
session_count = 0
in_file = open(session_file_name)
tmp_train_out_file = open(out_dir + "/train_data_tmp", "w")
tmp_test_out_file = open(out_dir + "/test_data_tmp", "w")
while True:
    line = in_file.readline()
    if not line:
        break
    arr = line.strip().split("\t")
    if len(arr) != 3 + len(featureList) * result_num:
        #print len(arr)
        continue
    session_count += 1
    query = urllib.unquote(arr[0])
    url_list = []
    click_list = []
    click_time_list = []
    ld_list = []
    hmr_list = []
    vft_list = []
    ft_list = []
    ht_list = []
    an_list = []
    for i in range(0, result_num):
        url_list.append(arr[3 + len(featureList) * i])
        ld_list.append(arr[3 + 1 + len(featureList) * i])
        hmr_list.append(arr[3 + 2 + len(featureList) * i])
        vft_list.append(arr[3 + 3 + len(featureList) * i])
        ft_list.append(arr[3 + 4 + len(featureList) * i])
        ht_list.append(arr[3 + 5 + len(featureList) * i])
        an_list.append(arr[3 + 6 + len(featureList) * i])
        click_list.append(arr[3 + 9 + len(featureList) * i])
        click_time_list.append(arr[3 + 10 + len(featureList) * i])
    for i in range(0, result_num):
        if ld_list[i] == "n":
            ld_list[i] = "0"
    out_str = str(query) + "\t"
    out_str += arr_string(url_list, inner_sep) + feature_sep
    out_str += arr_string(click_list, inner_sep) + feature_sep
    out_str += arr_string(click_time_list, inner_sep) + feature_sep
    out_str += arr_string(ld_list, inner_sep) + feature_sep
    out_str += arr_string(hmr_list, inner_sep) + feature_sep
    out_str += arr_string(vft_list, inner_sep) + feature_sep
    out_str += arr_string(ft_list, inner_sep) + feature_sep
    out_str += arr_string(ht_list, inner_sep) + feature_sep
    out_str += arr_string(an_list, inner_sep) + feature_sep
    #for i in range(0, result_num):
    #    out_str += "\t" + url_list[i] + "\t" + click_list[i]
    if not query_class_map.has_key(query):
        query_class_map[query] = 1
        query_list.append(query)
        query_train_count_map[query] = 0
        query_test_count_map[query] = 0
        load_query_count += 1
    if query_train_count_map[query] + query_test_count_map[query] < QUERY_MAX_SESSION:
        if random.random() < train_percentage:
            query_train_count_map[query] += 1
            tmp_train_out_file.write(out_str + "\n")
        else:
            query_test_count_map[query] += 1
            tmp_test_out_file.write(out_str + "\n")
        #print "train: " + str(query_train_count_map[query]) + " ; test: " + str(query_test_count_map[query])
in_file.close()
tmp_train_out_file.close()
tmp_test_out_file.close()
print "session " + str(session_count)
print "query " + str(load_query_count)

query_id_map = {}
valid_count = 0
out_query_id_file = open(out_dir + "/query_id", "w")
out_class_file = open(out_dir + "/query_class", "w")
for i in range(0, len(query_list)):
    query = query_list[i]
    if query_train_count_map[query] > 0:
        query_id_map[query] = valid_count
        out_query_id_file.write(query + "\t" + str(query_id_map[query]) + "\t" + str(query_train_count_map[query]) + "\t" + str(query_test_count_map[query]) + "\n")
        out_class_file.write(str(query_id_map[query]))
        for j in range(0, K):
            out_class_file.write("\t" + str(1))
        out_class_file.write("\n")
        valid_count += 1
out_query_id_file.close()
out_class_file.close()
print "valid query " + str(valid_count)

print "begine to map url id"
del query_class_map
del query_train_count_map
del query_test_count_map

w_session_count = 0
w_click_session_count = 0
w_2click_session_count = 0
w_revisit_session_count = 0
w_query_count = 0
w_click_query_count = 0
w_2click_query_count = 0
w_revisit_query_count = 0

w_query_map = {}

url_id_map = {}
valid_url_count = 0
load_file_list = ["train_data", "test_data"]
for file_name in load_file_list:
    print "process file " + file_name
    in_file_path = out_dir + "/" + file_name + "_tmp"
    in_file = open(in_file_path)
    out_file_path = out_dir + "/" + file_name
    out_file = open(out_file_path, "w")
    while True:
        line = in_file.readline()
        if not line:
            break
        arr = line.strip().split("\t")
        query = arr[0]
        other_list_arr = arr[1:]
        url_list = string_arr(other_list_arr[0], inner_sep, "")
        click_time_list = string_arr(other_list_arr[2], inner_sep, "")
        new_url_list = []
        if query_id_map.has_key(query):
            click_count_flag = 0
            revisit_flag = 0
            current_max_click_time = 0
            for i in range(0, len(click_time_list)):
                c_time = float(click_time_list[i])
                if c_time > 0:
                    click_count_flag += 1
                    if c_time < current_max_click_time:
                        revisit_flag = 1
                    if c_time > current_max_click_time:
                        current_max_click_time = c_time
            if not w_query_map.has_key(query):
                w_query_map[query] = [0,0,0]#click, 2click, revisit
                w_query_count += 1
            
            w_session_count += 1
            if click_count_flag > 0:
                w_click_session_count += 1
                if w_query_map[query][0] == 0:
                    w_query_map[query][0] = 1
                    w_click_query_count += 1
            if click_count_flag >= 2:
                w_2click_session_count += 1
                if w_query_map[query][1] == 0:
                    w_query_map[query][1] = 1
                    w_2click_query_count += 1
            if revisit_flag == 1:
                w_revisit_session_count += 1
                if w_query_map[query][2] == 0:
                    w_query_map[query][2] = 1
                    w_revisit_query_count += 1
        
        
            new_query = query_id_map[query]
            for i in range(0, len(url_list)):
                url = url_list[i]
                if not url_id_map.has_key(url):
                    url_id_map[url] = valid_url_count
                    valid_url_count += 1
                new_url_list.append(url_id_map[url])
            other_list_arr[0] = arr_string(new_url_list, inner_sep)
            out_file.write(str(new_query) + "\t" + arr_string(other_list_arr, feature_sep) + "\n")
    in_file.close()
    out_file.close()
print "total valid url = " + str(valid_url_count)

print "w_session_count = " + str(w_session_count)
print "w_click_session_count = " + str(w_click_session_count)
print "w_2click_session_count = " + str(w_2click_session_count)
print "w_revisit_session_count = " + str(w_revisit_session_count)
print "w_query_count = " + str(w_query_count)
print "w_click_query_count = " + str(w_click_query_count)
print "w_2click_query_count = " + str(w_2click_query_count)
print "w_revisit_query_count = " + str(w_revisit_query_count)

out_url_id_file = open(out_dir + "/url_id", "w")    
for url in url_id_map.keys():
    out_url_id_file.write(str(url) + "\t" + str(url_id_map[url]) + "\n")
out_url_id_file.close()    

os.system("rm " + out_dir + "/train_data_tmp")
os.system("rm " + out_dir + "/test_data_tmp")