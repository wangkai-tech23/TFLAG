import json
import pytz
from time import mktime
import numpy as np
import os
import json


def std(t):
    t = np.array(t)
    return np.std(t)

def var(t):
    t = np.array(t)
    return np.var(t)

def mean(t):
    t = np.array(t)
    return np.mean(t)

def ns_time_to_datetime_US(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    tz = pytz.timezone('US/Eastern')
    dt = pytz.datetime.datetime.fromtimestamp(int(ns) // 1000000000, tz)
    s = dt.strftime('%Y-%m-%d %H+%M+%S')
    #s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s



def cal_anomaly_loss(edge_list):
    loss_list = []
    for i in edge_list:
        loss_list.append(i['loss']['link_loss'])
    count = 0
    loss_sum = 0
    loss_std = std(loss_list)
    loss_mean = mean(loss_list)
    print('std:',loss_std,' mean:',loss_mean)

def cal_node_set_num():
    folder_path = ['./dataset/time_windows/test_data/','./dataset/time_windows/val_data/']

    # 获取文件夹中所有的文件名
    files1 = os.listdir(folder_path[0])
    files2 = os.listdir(folder_path[1])
    files = files1+files2
    node_list = {}
    # 遍历所有文件
    node_num = {}
    for file_name in files:
        if '2018-04-06' in file_name:
            folder_path_name = folder_path[0]
        else:
            folder_path_name = folder_path[1]
        file_path = os.path.join(folder_path_name, file_name)
        
        # 打开并读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        for edge in data:
            if edge['srcmsg'] in node_num:
                node_num[edge['srcmsg']] += 1
            else:
                node_num[edge['srcmsg']] = 1
            if edge['dstmsg'] in node_num:
                node_num[edge['dstmsg']] += 1
            else:
                node_num[edge['dstmsg']] = 1
    #print(node_list)
    sorted_node_num = dict(sorted(node_num.items(), key=lambda item: item[1], reverse=True))

    return sorted_node_num



def cal_windows_time(name):


    with open('./edge_loss_{}.json'.format(name), 'r', encoding='utf-8') as file:
        edge_loss_test = json.load(file)



    # The size of time window, 60000000000 represent 1 min in nanoseconds.
    # The default setting is 15 minutes.
    time_window_size = 60000000000 * 15
    # 初始时间
    start_time = edge_loss_test[0]['time']

    grouped_data = []
    current_group = []

    # 遍历所有数据
    for entry in edge_loss_test:
        # 如果当前entry的时间在当前组的时间窗口内，则将其添加到当前组
        if entry['time'] < start_time + time_window_size:
            current_group.append(entry)
        else:
            # 当前组的时间窗口结束，将当前组添加到结果中，并开始一个新的组
            grouped_data.append(current_group)
            current_group = [entry]
            # 更新起始时间为当前entry的时间窗口开始时间
            start_time = entry['time']

    # 最后一个组如果有数据，也要添加到结果中
    if current_group:
        grouped_data.append(current_group)
    return grouped_data




test_grouped_data = cal_windows_time('test')
val_grouped_data = cal_windows_time('val')

for windows in test_grouped_data:
    time_interval = ns_time_to_datetime_US(windows[0]['time']) + "~" + ns_time_to_datetime_US(windows[-1]['time'])
    edge_list = sorted(windows, key=lambda x: x['loss']['link_loss'], reverse=True)
    with open('./dataset/time_windows/test_data/{}.json'.format(time_interval), 'w', encoding='utf-8') as json_file:
        json.dump(edge_list, json_file, ensure_ascii=False)
for windows in val_grouped_data:
    time_interval = ns_time_to_datetime_US(windows[0]['time']) + "~" + ns_time_to_datetime_US(windows[-1]['time'])
    edge_list = sorted(windows, key=lambda x: x['loss']['link_loss'], reverse=True)
    with open('./dataset/time_windows/val_data/{}.json'.format(time_interval), 'w', encoding='utf-8') as json_file:
        json.dump(edge_list, json_file, ensure_ascii=False)


grouped_data = test_grouped_data + val_grouped_data

windows_max_edge_loss = {}
for windows in grouped_data:
    time_interval = ns_time_to_datetime_US(windows[0]['time']) + "~" + ns_time_to_datetime_US(windows[-1]['time'])
    edge_list = sorted(windows, key=lambda x: x['loss']['link_loss'], reverse=True)
    windows_max_edge_loss[time_interval] = edge_list[0]['loss']
    windows_max_edge_loss[time_interval]['srcmsg'] = edge_list[0]['srcmsg']
    windows_max_edge_loss[time_interval]['dstmsg'] = edge_list[0]['dstmsg']

    
windows_max_edge_loss = dict(sorted(windows_max_edge_loss.items(), key=lambda item: item[1]['link_loss'], reverse=True))
top_5_percent_count = int(len(windows_max_edge_loss) * 0.05)
windows_max_edge_loss_top = dict(list(windows_max_edge_loss.items())[:top_5_percent_count])

#with open('windows_max_edge_loss.json', 'w', encoding='utf-8') as json_file:
#    json.dump(windows_max_edge_loss, json_file, ensure_ascii=False, indent=4)


node_num = cal_node_set_num()
min_src_msg_windows_edge = ('null',100000)
min_dst_msg_windows_edge = ('null',100000)
for key,val in windows_max_edge_loss_top.items():
    if node_num[val['srcmsg']] <= min_src_msg_windows_edge[1]:
        min_src_msg_windows_edge = (key,node_num[val['srcmsg']])
    if node_num[val['dstmsg']] <= min_dst_msg_windows_edge[1]:
        min_dst_msg_windows_edge = (key,node_num[val['dstmsg']])
    #print(val,node_num[val['srcmsg']],node_num[val['dstmsg']])
if min_src_msg_windows_edge[0] == min_dst_msg_windows_edge[0]:
    min_msg_windows_edge = min_src_msg_windows_edge[0]
else:
    min_msg_windows_edge = (min_src_msg_windows_edge[0] 
                        if abs(windows_max_edge_loss_top[min_src_msg_windows_edge[0]]['anomaly_loss']) > 
                           abs(windows_max_edge_loss_top[min_dst_msg_windows_edge[0]]['anomaly_loss']) 
                        else min_dst_msg_windows_edge[0])

print(min_msg_windows_edge)