import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class MLPCLFDecoder(nn.Module):
    def __init__(self, input_dim, num_ent, drop=0.1):
        """这个部分用于解码得到最后的答案"""
        super(MLPCLFDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, num_ent, bias=True)
        self.dropout = nn.Dropout(drop)

    def forward(self, features):
        return self.fc(self.dropout(features))

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

class ConvTransE(torch.nn.Module):
    def __init__(self, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):
        super(ConvTransE, self).__init__()
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)

    def forward(self, embedding, emb_rel, candidate_emb):
        embedding = F.tanh(embedding)
        candidate_emb = F.tanh(candidate_emb)
        embedding = embedding.unsqueeze(1)
        emb_rel = emb_rel.unsqueeze(1)
        batch_size = embedding.shape[0]
        stacked_inputs = torch.cat([embedding, emb_rel], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, candidate_emb.transpose(1, 0))
        return x


