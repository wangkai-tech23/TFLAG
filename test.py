import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets as dataset
import torch.utils.data
import sklearn
import numpy as np
from option import args
from model.tgat import TGAT
from utils import EarlyStopMonitor, logger_config
from tqdm import tqdm
import datetime, os
import json


import random

import numpy as np
import torch.backends.cudnn as cudnn

seed = 41
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # 如果在使用 DataLoader 并且采用了多进程加载数据的方式
    #torch.set_deterministic(True)

# 在训练开始前设置随机种子
set_random_seed(seed)

def worker_init_fn(worker_id, seed):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

rel2id = {
 1: 'EVENT_WRITE',
 2: 'EVENT_READ',
 3: 'EVENT_CLOSE',
 4: 'EVENT_OPEN',
 5: 'EVENT_EXECUTE',
 6: 'EVENT_SENDTO',
 7: 'EVENT_RECVFROM',
}

with open('./dataset/node_map.json', 'r', encoding='utf-8') as file:
    node_map = json.load(file)

crossentropyloss = nn.CrossEntropyLoss()

def cal_pos_edges_loss_multiclass(link_pred_ratio,labels):
    loss=[]
    for i in range(len(link_pred_ratio)):
        loss.append(crossentropyloss(link_pred_ratio[i].reshape(1,-1),labels[i].reshape(-1)))
    return torch.tensor(loss)




def criterion(prediction_dict, labels, model, config):
    loss_anomaly_list = []
    loss_supc_list = []
    loss_list = []

    for key, value in prediction_dict.items():
        if key != 'root_embedding' and key != 'dst_embedding' and key != 'group' and key != 'dev':
            prediction_dict[key] = value[labels > -1]

    labels = labels[labels > -1]
    logits = prediction_dict['logits']

    loss_classify = cal_pos_edges_loss_multiclass(logits, labels.long())
    loss_list += loss_classify
    loss_classify = torch.mean(loss_classify)

    loss_anomaly = torch.Tensor(0).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    loss_supc = torch.Tensor(0).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


    loss_anomaly = model.gdn.dev_loss(torch.squeeze(prediction_dict['pre_lable']), torch.squeeze(prediction_dict['anom_score']), torch.squeeze(prediction_dict['time']))
    loss_anomaly_list += loss_anomaly
    loss_anomaly = torch.mean(loss_anomaly)
    loss_supc = model.suploss(prediction_dict['root_embedding'],prediction_dict['dst_embedding'] ,prediction_dict['group'], prediction_dict['dev'])
    loss_supc_list += loss_supc
    loss_supc = loss_supc.mean()

    

    return loss_list, loss_anomaly_list, loss_supc_list

def eval_epoch(dataset, model, config, device, name):
    with open('./dataset/nodeid2msg.json', 'r', encoding='utf-8') as file:
        nodeid2msg = json.load(file)
    edge_list = []
    with torch.no_grad():
        model.eval()
        with tqdm(total=len(dataset)) as t:
            for batch_sample in dataset:
                x = model(
                    batch_sample['src_edge_feat'].to(device),
                    batch_sample['src_edge_to_time'].to(device),
                    batch_sample['src_center_node_idx'].to(device),
                    batch_sample['dst_center_node_idx'].to(device),
                    batch_sample['src_neigh_edge'].to(device),
                    batch_sample['src_node_features'].to(device),
                    batch_sample['current_time'].to(device),
                    batch_sample['labels'].to(device)
                )
                y = batch_sample['labels'].to(device)
                link_loss, anomaly_loss,supc_loss = criterion(x, y, model, config)
                src_old_idx = batch_sample['old_src_center_node_idx']
                dst_old_idx = batch_sample['old_dst_center_node_idx']
                current_time = batch_sample['current_time']
                edgetype_id = batch_sample['labels']
                for i in range(len(link_loss)):
                    srcnode = node_map[str(src_old_idx[i].item())]
                    dstnode = node_map[str(dst_old_idx[i].item())]

                    edge_time = int(current_time[i].item())

                    srcmsg = str(nodeid2msg[str(srcnode)])
                    dstmsg = str(nodeid2msg[str(dstnode)])

                    edge_type = rel2id[int(edgetype_id[i].item()) + 1]

                    link = link_loss[i]
                    anomaly = anomaly_loss[i]
                    supc = supc_loss[i]
                    temp_dic = {}
                    temp_dic['loss'] = {'link_loss': float(link), 'anomaly_loss': float(anomaly), 'supc_loss': float(supc) }
                    temp_dic['srcnode'] = srcnode
                    temp_dic['dstnode'] = dstnode
                    temp_dic['srcmsg'] = srcmsg
                    temp_dic['dstmsg'] = dstmsg
                    temp_dic['edge_type'] = edge_type
                    temp_dic['time'] = edge_time

                    edge_list.append(temp_dic)
                t.update(1)
          
    with open('./result/edge_loss/edge_loss_{}.json'.format(name), 'w', encoding='utf-8') as json_file:
        json.dump(edge_list, json_file, ensure_ascii=False)

    return 






config = args
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_valid = dataset.DygDataset(config, 'valid',[(1522728000000000000,1522987200000000000),(1522987200000000000,1523073600000000000),(1523073600000000000,1523073600000000000)])
dataset_test = dataset.DygDataset(config, 'test',[(1522728000000000000,1522987200000000000),(1522987200000000000,1523073600000000000),(1523073600000000000,1523073600000000000)])
gpus = None if config.gpus == 0 else config.gpus

collate_fn = dataset.Collate(config)

backbone = TGAT(config, device)
model = backbone.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

print(len(dataset_test))

loader_valid = torch.utils.data.DataLoader(
    dataset=dataset_valid,
    batch_size=config.batch_size,
    shuffle=False,
    #shuffle=True,
    num_workers=config.num_data_workers,
    collate_fn=collate_fn.dyg_collate_fn,
    worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed)
)


loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=config.batch_size,
    shuffle=False,
    #shuffle=True,
    num_workers=config.num_data_workers,
    collate_fn=collate_fn.dyg_collate_fn,
    worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed)
)

model.load_state_dict(torch.load('./checkpoint-2'))

eval_epoch(loader_test, model, config, device, 'test')
eval_epoch(loader_valid, model, config, device, 'val')
