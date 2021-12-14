import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLPCLFDecoder(nn.Module):
    def __init__(self, input_dim, num_ent):
        """这个部分用于解码得到最后的答案"""
        super(MLPCLFDecoder, self).__init__()
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, input_dim, bias=True)
        self.fc = nn.Linear(input_dim, num_ent, bias=True)
        self.dropout = nn.Dropout(0.0)

    def forward(self, features):
        return self.dropout(self.fc(self.act(self.fc1(features))))

class DistMultDecoder(nn.Module):
    def __init__(self, drop=0.0):
        super(DistMultDecoder, self).__init__()
        self.drop = nn.Dropout(drop)  # 输入向量dropout

    def forward(self, sub_emb, rel_emb, candidate_emb, sigmoid=False):
        """
        sub_emb: 需要打分的subject entities向量， [batch_size, ent_dim]
        rel_emb: 关系向量，[batch_size, rel_dim] rel_dim == ent_dim
        candidate_emb: 需要打分的实体向量，[num_ent, ent_dim]
        """
        sub_emb = self.drop(sub_emb)
        rel_emb = self.drop(rel_emb)
        score = torch.mm(sub_emb * rel_emb, candidate_emb.transpose(1, 0))
        if sigmoid:
            score = torch.sigmoid(score)
        return score

