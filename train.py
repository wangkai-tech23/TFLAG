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


crossentropyloss = nn.CrossEntropyLoss()
def criterion(prediction_dict, labels,model, config):

    for key, value in prediction_dict.items():
        if key != 'root_embedding' and key != 'dst_embedding' and key != 'group' and key != 'dev':
            prediction_dict[key] = value[labels > -1]

    labels = labels[labels > -1]
    logits = prediction_dict['logits']

    loss_classify = crossentropyloss(logits, labels.long())
    loss_classify = torch.mean(loss_classify)

    loss = loss_classify.clone()
    loss_anomaly = torch.Tensor(0).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    loss_supc = torch.Tensor(0).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    alpha = config.anomaly_alpha  
    beta = config.supc_alpha 
    loss_anomaly = model.gdn.dev_loss(torch.squeeze(prediction_dict['pre_lable']), torch.squeeze(prediction_dict['anom_score']), torch.squeeze(prediction_dict['time']))
    loss_anomaly = torch.mean(loss_anomaly)
    loss_supc = model.suploss(prediction_dict['root_embedding'],prediction_dict['dst_embedding'] ,prediction_dict['group'], prediction_dict['dev'])
    loss_supc = loss_supc.mean()
    loss += alpha * loss_anomaly + beta * loss_supc
    

    return loss, loss_classify, loss_anomaly, loss_supc



config = args
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dataset_train = dataset.DygDataset(config, 'train',[(1522728000000000000,1522987200000000000),(1522987200000000000,1523073600000000000),(1523073600000000000,1523073600000000000)])

gpus = None if config.gpus == 0 else config.gpus

collate_fn = dataset.Collate(config)

backbone = TGAT(config, device)
model = backbone.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

print(len(dataset_train))
#print(len(dataset_test))
loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=config.batch_size,
    shuffle=False,
    #shuffle=True,
    num_workers=config.num_data_workers,
    pin_memory=True,
    #sampler=dataset.RandomDropSampler(dataset_train, 0.75),   #for reddit
    collate_fn=collate_fn.dyg_collate_fn,
    worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed)
)


max_val_auc, max_test_auc = 0.0, 0.0
early_stopper = EarlyStopMonitor()
best_auc = [0, 0, 0]
for epoch in range(config.n_epochs):
    ave_loss = 0
    count_flag = 0
    m_loss, auc = [], []
    loss_anomaly_list = []
    loss_class_list = []
    loss_supc_list = []
    with tqdm(total=len(loader_train)) as t:
        for batch_sample in loader_train:
            count_flag += 1
            t.set_description('Epoch %i' % epoch)
            optimizer.zero_grad()
            model.train()
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
            loss, loss_classify, loss_anomaly, loss_supc = criterion(x, y ,model, config)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()

            # get training results
            with torch.no_grad():
                model = model.eval()
                m_loss.append(loss.item())


            loss_class_list.append(loss_classify.detach().clone().cpu().numpy().flatten())

            loss_anomaly_list.append(loss_anomaly.detach().clone().cpu().numpy().flatten())
            loss_supc_list.append(loss_supc.detach().clone().cpu().numpy().flatten())
            t.set_postfix(loss=np.mean(loss_class_list), loss_anomaly=np.mean(loss_anomaly_list), loss_sup=np.mean(loss_supc_list))

            t.update(1)



    print('\n epoch: {}'.format(epoch))
    print(f'train mean loss:{np.mean(m_loss)}, class loss: {np.mean(loss_class_list)}, anomaly loss: {np.mean(loss_anomaly_list)}, sup loss: {np.mean(loss_supc_list)}')
    torch.save(model.state_dict(), "./checkpoint-{}".format(epoch))
