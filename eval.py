import json
import pytz
from time import mktime
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, roc_auc_score

def classifier_evaluation(y_test, y_test_pred):
    tn, fp, fn, tp =confusion_matrix(y_test, y_test_pred).ravel()
    print(f'tn: {tn}')
    print(f'fp: {fp}')
    print(f'fn: {fn}')
    print(f'tp: {tp}')

    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    fscore=2*(precision*recall)/(precision+recall)
    auc_val=roc_auc_score(y_test, y_test_pred)
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"fscore: {fscore}")
    print(f"accuracy: {accuracy}")
    print(f"auc_val: {auc_val}")
    return precision,recall,fscore,accuracy,auc_val


def cal_node_num(time_windows1, time_windows2):


    time_windows1_node_num = {}
    time_windows2_node_num = {}

    for edge in time_windows1:
        if edge['dstmsg'] in time_windows1_node_num:
            time_windows1_node_num[edge['dstmsg']] += 1
        else:
            time_windows1_node_num[edge['dstmsg']] = 1
    for edge in time_windows2:
        if edge['dstmsg'] in time_windows2_node_num:
            time_windows2_node_num[edge['dstmsg']] += 1
        else:
            time_windows2_node_num[edge['dstmsg']] = 1
    #print(node_list)
    time_windows1_node_num = dict(sorted(time_windows1_node_num.items(), key=lambda item: item[1], reverse=False))
    time_windows2_node_num = dict(sorted(time_windows2_node_num.items(), key=lambda item: item[1], reverse=False))

    return time_windows1_node_num,time_windows2_node_num


def cal_node_set_num(time_windows1, time_windows2): # 直接计算loss 前百分之十的交集
    
    top_10_percent_count1 = max(1, len(time_windows1) * 10 // 100)
    top_10_percent_count2 = max(1, len(time_windows2) * 10 // 100)

    # 提取前 10% 的字典
    top_10_percent1 = time_windows1[:top_10_percent_count1]
    top_10_percent2 = time_windows2[:top_10_percent_count2]

    # 提取 srcmsg 和 dstmsg 的集合
    nodes1 = {entry['dstmsg'] for entry in top_10_percent1}#.union({entry['dstmsg'] for entry in top_10_percent1})
    nodes2 = {entry['dstmsg'] for entry in top_10_percent2}#.union({entry['dstmsg'] for entry in top_10_percent2})

    # 找到交集
    common_nodes = nodes1.intersection(nodes2)
    union = nodes1.union(nodes2)
    jaccard_similarity = len(common_nodes) / len(union)
    nodes1 = set(nodes1)
    nodes2 = set(nodes2)
    #print("Common nodes in top 5%:", common_nodes)
    #print("前百分之五的独特节点数量，异常：",len(nodes1),"将要检测的:",len(nodes2))
    #print("node nums:",len(common_nodes))
    #print("笛卡尔系数",jaccard_similarity)

    return common_nodes

def cal_node_set_num_2(time_windows1, time_windows2): # 直接计算频率低的前十 的交集
    time_windows1_node_num,time_windows2_node_num = cal_node_num(time_windows1, time_windows2)
    top_10_percent_count1 = max(1, len(time_windows1_node_num) * 20 // 100)
    top_10_percent_count2 = max(1, len(time_windows2_node_num) * 20 // 100)


    #first_ten_items = list(time_windows1_node_num.items())[:top_10_percent_count1]
    #print(first_ten_items)
    # 提取前 10% 的字典
    top_10_percent1 = list(time_windows1_node_num.items())[:top_10_percent_count1]
    top_10_percent2 = list(time_windows2_node_num.items())[:top_10_percent_count2]

    # 提取 srcmsg 和 dstmsg 的集合
    nodes1 = {entry[0] for entry in top_10_percent1}#.union({entry['dstmsg'] for entry in top_10_percent1})
    nodes2 = {entry[0] for entry in top_10_percent2}#.union({entry['dstmsg'] for entry in top_10_percent2})
    # 找到交集
    common_nodes = nodes1.intersection(nodes2)
    union = nodes1.union(nodes2)
    jaccard_similarity = len(common_nodes) / len(union)
    nodes1 = set(nodes1)
    nodes2 = set(nodes2)
    #print("Common nodes in top 5%:", common_nodes)
    print("前百分之五的独特节点数量，异常：",len(nodes1),"将要检测的:",len(nodes2))
    print("node nums:",len(common_nodes))
    print("笛卡尔系数",jaccard_similarity)

    return jaccard_similarity


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


folder_path = './dataset/time_windows/test_data/'

files = os.listdir(folder_path)
index = files.index('2018-04-06 11+45+45~2018-04-06 12+00+45.json')


current_index = index
results = [files[index]]

print('向前检测++++++++++++++++++++++++++++++++')
threshold = 0
while current_index - 1 >= 0:
    now = current_index
    pre = current_index - 1
    
    with open(os.path.join(folder_path, files[now]), 'r', encoding='utf-8') as file:
        time_windows1 = json.load(file)
    
    with open(os.path.join(folder_path, files[pre]), 'r', encoding='utf-8') as file:
        time_windows2 = json.load(file)
    
    jaccard_similarity = cal_node_set_num_2(time_windows1, time_windows2)


    print(jaccard_similarity,files[now],files[pre])
    if jaccard_similarity > 0.1 and threshold == 0:
        threshold = jaccard_similarity
        
        results = [files[pre]]  + results
        current_index -= 1
    elif threshold != 0 and jaccard_similarity >= threshold * 0.85:
        print(threshold * 0.85)
        results = [files[pre]]  + results
        current_index -= 1
    else:
        break

# 重置变量以进行向后遍历
current_index = index
print('向后检测++++++++++++++++++++++++++++++++')
# 向后遍历
threshold = 0
while current_index + 1 < len(files):
    now = current_index
    next_file = current_index + 1
    
    with open(os.path.join(folder_path, files[now]), 'r', encoding='utf-8') as file:
        time_windows1 = json.load(file)
    
    with open(os.path.join(folder_path, files[next_file]), 'r', encoding='utf-8') as file:
        time_windows2 = json.load(file)
    
    jaccard_similarity = cal_node_set_num_2(time_windows1, time_windows2)

    print(jaccard_similarity,files[now],files[next_file])
    if jaccard_similarity > 0.1 and threshold == 0:
        threshold = jaccard_similarity
        results.append(files[next_file])
        current_index += 1
    elif threshold != 0 and jaccard_similarity >= threshold * 0.85:
        results.append(files[next_file])
        current_index += 1
    else:
        break
print(results)
files2 = os.listdir('./dataset/time_windows/val_data/')

all_data = files + files2

# 11:21 发送 HTTP post，漏洞利用成功，但操作员控制面板上没有 drakon 连接
# 11.22 成功，接回
# 11:33 提升
# 11:38 nrinfo
# 11:39 nrtcp 154.145.113.18 80
# 11:42 nrtcp 61.167.39.128 80
# 12:04 putfile ./deploy/archive/libdrakon.freebsd.x64.so_152.111.159.139 /var/log/devc
# 12:04 ps
# 12:08 注入 foo 123
# 12:08 注入 /var/log/devc xxx
# CADETS 坠毁，外壳丢失，无注入物
attack_list = ['2018-04-06 11+15+45~2018-04-06 11+30+41.json', 
               '2018-04-06 11+30+45~2018-04-06 11+45+41.json', 
               '2018-04-06 11+45+45~2018-04-06 12+00+45.json', 
               '2018-04-06 12+00+46~2018-04-06 12+08+59.json']
pred_lable = []
lables = []
for file in all_data:
    if file in attack_list:
        lables.append(1)
    else:
        lables.append(0)
    if file in results:
        pred_lable.append(1)
    else:
        pred_lable.append(0)
classifier_evaluation(lables, pred_lable)