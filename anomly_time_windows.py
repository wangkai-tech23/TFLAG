import json
import pytz
from time import mktime
import numpy as np
import os
import json




def ns_time_to_datetime_US(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    tz = pytz.timezone('US/Eastern')
    dt = pytz.datetime.datetime.fromtimestamp(int(ns) // 1000000000, tz)
    s = dt.strftime('%Y-%m-%d %H+%M+%S')
    s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s




def cal_node_set_num():
    folder_path = ['./result/time_window/test_data/','./result/time_window/val_data/']

    # 获取文件夹中所有的文件名
    files1 = os.listdir(folder_path[0])
    files2 = os.listdir(folder_path[1])
    files = files1+files2

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


    with open('./result/edge_loss/edge_loss_{}.json'.format(name), 'r', encoding='utf-8') as file:
        edge_loss_test = json.load(file)



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

def write_time_windows_json(flag,name,grouped_data):
    windows_max_edge_loss = {}
    windows_edge_verage_loss = {}
    windows_edge_verage_anomaly_loss = {}
    for windows in grouped_data:
        time_interval = ns_time_to_datetime_US(windows[0]['time']) + "~" + ns_time_to_datetime_US(windows[-1]['time'])
        edge_list = sorted(windows, key=lambda x: x['loss']['link_loss'], reverse=True)
        windows_max_edge_loss[time_interval] = edge_list[0]['loss']
        windows_max_edge_loss[time_interval]['srcmsg'] = edge_list[0]['srcmsg']
        windows_max_edge_loss[time_interval]['dstmsg'] = edge_list[0]['dstmsg']
        total_link_loss = sum([x['loss']['link_loss'] for x in windows])
        total_anomaly_loss = sum([x['loss']['anomaly_loss'] for x in windows])
        event_count = len(windows)
        verage_link_loss = total_link_loss / event_count
        verage_anomaly_loss = total_anomaly_loss / event_count
        print(time_interval,windows[0]['time'],verage_link_loss)
        windows_edge_verage_loss[time_interval] = verage_link_loss
        windows_edge_verage_anomaly_loss[time_interval] = verage_anomaly_loss

        if flag == True:
            with open('./result/time_window/{}_data/{}.json'.format(name,time_interval), 'w', encoding='utf-8') as json_file:
                json.dump(edge_list, json_file, ensure_ascii=False)
    return windows_max_edge_loss, windows_edge_verage_loss,windows_edge_verage_anomaly_loss



# 计算两天的所有时间窗口，按照loss从高到低
test_grouped_data = cal_windows_time('test')

windows_max_edge_loss_test,_,_ = write_time_windows_json(True,'test',test_grouped_data)

val_grouped_data = cal_windows_time('val')

windows_max_edge_loss_val,_,_ = write_time_windows_json(True,'val',val_grouped_data)


# 计算测试天和评估天 单独天内 所有 loss最异常的排序
windows_max_edge_loss_test = dict(sorted(windows_max_edge_loss_test.items(), key=lambda item: item[1]['link_loss'], reverse=True))
with open('./windows_max_edge_loss_test.json', 'w', encoding='utf-8') as json_file:
        json.dump(windows_max_edge_loss_test, json_file, ensure_ascii=False, indent=4)

windows_max_edge_loss_val = dict(sorted(windows_max_edge_loss_val.items(), key=lambda item: item[1]['link_loss'], reverse=True))
with open('./windows_max_edge_loss_val.json', 'w', encoding='utf-8') as json_file:
        json.dump(windows_max_edge_loss_val, json_file, ensure_ascii=False, indent=4)

# 计算两者 一起的 最异常窗口，并且 还有求一个平均窗口loss
windows_max_edge_loss,windows_edge_verage_loss_list,windows_edge_verage_anomaly_loss_list= write_time_windows_json(False,'all',test_grouped_data + val_grouped_data)
windows_edge_verage_loss_list = dict(sorted(windows_edge_verage_loss_list.items(), key=lambda item: item[1], reverse=True))
windows_edge_verage_anomaly_loss_list = dict(sorted(windows_edge_verage_anomaly_loss_list.items(), key=lambda item: item[1], reverse=True))

with open('./windows_max_edge_verage_loss.json', 'w', encoding='utf-8') as json_file:
        json.dump(windows_edge_verage_loss_list, json_file, ensure_ascii=False, indent=4)
with open('./windows_max_edge_verage_anomaly.json', 'w', encoding='utf-8') as json_file:
        json.dump(windows_edge_verage_anomaly_loss_list, json_file, ensure_ascii=False, indent=4)

# 把 前百分5的事件 排序
windows_max_edge_loss = dict(sorted(windows_max_edge_loss.items(), key=lambda item: item[1]['link_loss'], reverse=True))
top_5_percent_count = int(len(windows_max_edge_loss) * 0.05)
windows_max_edge_loss_top = dict(list(windows_max_edge_loss.items())[:top_5_percent_count])

with open('windows_max_edge_loss.json', 'w', encoding='utf-8') as json_file:
    json.dump(windows_max_edge_loss, json_file, ensure_ascii=False, indent=4)

# 计算几点出现的频率
node_num = cal_node_set_num()
with open('node_num.json', 'w', encoding='utf-8') as json_file:
    json.dump(node_num, json_file, ensure_ascii=False, indent=4)

