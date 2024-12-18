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

    for key, value in prediction_dict.items():
        if key != 'root_embedding' and key != 'dst_embedding' and key != 'group' and key != 'dev':
            prediction_dict[key] = value[labels > -1]

    labels = labels[labels > -1]
    logits = prediction_dict['logits']

    loss_classify = crossentropyloss(logits, labels.long())
    loss_classify = torch.mean(loss_classify)

    loss = loss_classify.clone()
    loss_anomaly = torch.Tensor(0).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    alpha = config.anomaly_alpha  # 1e-1

    loss_anomaly = model.gdn.dev_loss(torch.squeeze(labels), torch.squeeze(prediction_dict['anom_score']), torch.squeeze(prediction_dict['time']))
    loss_anomaly = torch.mean(loss_anomaly)
    loss += alpha * loss_anomaly
    

    return loss, loss_classify, loss_anomaly



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


max_val_auc, max_test_auc = 0.0, 0.0
early_stopper = EarlyStopMonitor()
best_auc = [0, 0, 0]
for epoch in range(config.n_epochs):
    ave_loss = 0
    count_flag = 0
    m_loss, auc = [], []
    loss_anomaly_list = []
    loss_class_list = []
    dev_score_list = np.array([])
    dev_label_list = np.array([])
    
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
            loss, loss_classify, loss_anomaly = criterion(x, y, model, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()

            # get training results
            with torch.no_grad():
                model = model.eval()
                m_loss.append(loss.item())
                # pred_score = x['logits'].sigmoid()


            loss_class_list.append(loss_classify.detach().clone().cpu().numpy().flatten())
            loss_anomaly_list.append(loss_anomaly.detach().clone().cpu().numpy().flatten())
            t.set_postfix(loss=np.mean(loss_class_list), loss_anomaly=np.mean(loss_anomaly_list))
            t.update(1)



    logger.info('\n epoch: {}'.format(epoch))
    logger.info(f'train mean loss:{np.mean(m_loss)}, class loss: {np.mean(loss_class_list)}, anomaly loss: {np.mean(loss_anomaly_list)}')

    torch.save(model.state_dict(), "./checkpoint-{}".format(epoch))