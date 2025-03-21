#from option import args
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch

import json

"""with open('./dataset/edge_type.json', 'r', encoding='utf-8') as json_file:
    edge_type = json.load(json_file)
with open('./dataset/node_type.json', 'r', encoding='utf-8') as json_file:
    node_type = json.load(json_file)
"""



def preprocess(data_name):
  node_num = 0
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []
  node_map = {}
  node_cnt = 0
  node_cnt_map = {}

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0])
      i = int(e[1])
      if u not in node_cnt_map:
        node_cnt_map[u] = node_cnt
        u = node_cnt
        node_cnt += 1
      else:
        u = node_cnt_map[u]
      if i not in node_cnt_map:
        node_cnt_map[i] = node_cnt
        i = node_cnt
        node_cnt += 1
      else:
        i = node_cnt_map[i]
        

      ts = float(e[5])
      label = float(e[4]) - 1   # int(e[3])

      u_feat = int(e[2])
      i_feat = int(e[3])
      if u not in node_map:
        node_num += 1
        node_map[u] = F.one_hot(torch.tensor(u_feat), num_classes=3).float()
      if i not in node_map:
        node_num += 1
        node_map[i] = F.one_hot(torch.tensor(i_feat), num_classes=3).float()

      # print(feat)
      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}),node_map ,node_num,node_cnt_map

def reindex(df, bipartite=False):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df



if __name__=='__main__':
    dateset_dir = './result.csv'
    OUT_DF = './ml_cadets.csv'
    OUT_FEAT = './ml_cadets_edge.npy'
    OUT_NODE_FEAT = './ml_cadets_node.npy'

    df,node_map,node_num,node_cnt_map = preprocess(dateset_dir)
    print(node_num)
    new_df = reindex(df, False)
    new_node_num = {}
    for i in node_cnt_map:
      new_node_num[node_cnt_map[i] + 1] = i
    with open('./node_map.json', 'w') as json_file:
      json.dump(new_node_num, json_file, indent=4)

    max_idx = max(new_df.u.max(), new_df.i.max()) + 1
    feature_length = len(node_map[0])  # 假设 node_map[0] 存在，并且其值是独热编码后的特征

    # 初始化全零矩阵
    rand_feat = np.zeros((max_idx, feature_length))

    # 填充矩阵
    for idx, feature in node_map.items():
        rand_feat[idx+1] = feature

    new_df.to_csv(OUT_DF)
    # np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)
