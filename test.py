import torch
import torch.nn as nn
import torch.nn.functional as F
import util.datasets as dataset
import torch.utils.data
import sklearn
import numpy as np
from option import args
from model.tgat import TGAT
from util.utils import EarlyStopMonitor, logger_config
from tqdm import tqdm
import datetime, os
import json


rel2id = {
 1: 'EVENT_WRITE',
 2: 'EVENT_READ',
 3: 'EVENT_CLOSE',
 4: 'EVENT_OPEN',
 5: 'EVENT_EXECUTE',
 6: 'EVENT_SENDTO',
 7: 'EVENT_RECVFROM',
}
with open('./dataset/pre_data/node_map.json', 'r', encoding='utf-8') as file:
    node_map = json.load(file)

def cal_pos_edges_loss_multiclass(link_pred_ratio,labels):
    loss=[]
    for i in range(len(link_pred_ratio)):
        loss.append(crossentropyloss(link_pred_ratio[i].reshape(1,-1),labels[i].reshape(-1)))
    return torch.tensor(loss)

crossentropyloss = nn.CrossEntropyLoss()
def create_dataloader(dataset_type, config, collate_fn):
    datasets = dataset.DygDataset(config, dataset_type, 
                                 [(1522728000000000000, 1522987200000000000),
                                  (1522987200000000000, 1523073600000000000),
                                  (1523073600000000000, 1523073600000000000)])
    return torch.utils.data.DataLoader(
        dataset=datasets,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_data_workers,
        pin_memory=True,
        collate_fn=collate_fn.dyg_collate_fn
    )

# 创建 DataLoader


def criterion(prediction_dict, labels, model, config):
    loss_anomaly_list = []
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

    loss_anomaly = model.gdn.dev_loss(torch.squeeze(labels), torch.squeeze(prediction_dict['anom_score']), torch.squeeze(prediction_dict['time']))
    loss_anomaly_list += loss_anomaly
    

    return loss_list, loss_anomaly_list


def eval_epoch(dataset, model, config, device, name):
    with open('./dataset/nodeid2msg', 'r', encoding='utf-8') as file:
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
                link_loss, anomaly_loss= criterion(x, y, model, config)
                # idx_mapping = batch_sample['idx_mapping']
                src_old_idx = batch_sample['old_src_center_node_idx']
                dst_old_idx = batch_sample['old_dst_center_node_idx']
                current_time = batch_sample['current_time']
                edgetype_id = batch_sample['labels']
                for i in range(len(link_loss)):
                    srcnode = node_map[str(src_old_idx[i].item())]
                    dstnode = node_map[str(dst_old_idx[i].item())]

                    edge_time = int(current_time[i].item())

                    srcmsg = str(nodeid2msg[srcnode])
                    dstmsg = str(nodeid2msg[dstnode])

                    edge_type = rel2id[int(edgetype_id[i].item()) + 1]

                    link = link_loss[i]
                    anomaly = anomaly_loss[i]
                    temp_dic = {}
                    temp_dic['loss'] = {'link_loss': float(link), 'anomaly_loss': float(anomaly) }
                    temp_dic['srcnode'] = srcnode
                    temp_dic['dstnode'] = dstnode
                    temp_dic['srcmsg'] = srcmsg
                    temp_dic['dstmsg'] = dstmsg
                    temp_dic['edge_type'] = edge_type
                    temp_dic['time'] = edge_time
                    edge_list.append(temp_dic)
                t.update(1)
          
    with open('edge_loss_{}.json'.format(name), 'w', encoding='utf-8') as json_file:
        json.dump(edge_list, json_file, ensure_ascii=False)

    return 

config = args
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# log file name set
now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_base_path = f"{os.getcwd()}/train_log"
file_list = os.listdir(log_base_path)
max_num = [0] # [int(fl.split("_")[0]) for fl in file_list if len(fl.split("_"))>2] + [-1]
log_base_path = f"{log_base_path}/{max(max_num)+1}_{now_time}"
# log and path
get_checkpoint_path = lambda epoch: f'{log_base_path}saved_checkpoints/{args.data_set}-{args.mode}-{args.module_type}-{args.mask_ratio}-{epoch}.pth'
logger = logger_config(log_path=f'{log_base_path}/log.txt', logging_name='TFLAG')
logger.info(config)

gpus = None if config.gpus == 0 else config.gpus

collate_fn = dataset.Collate(config)

backbone = TGAT(config, device)
model = backbone.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

loader_train = create_dataloader('train', config, collate_fn)
loader_valid = create_dataloader('valid', config, collate_fn)
loader_test = create_dataloader('test', config, collate_fn)


model.load_state_dict(torch.load('./checkpoint-4'))

eval_epoch(loader_test, model, config, device, 'test')
eval_epoch(loader_valid, model, config, device, 'val')