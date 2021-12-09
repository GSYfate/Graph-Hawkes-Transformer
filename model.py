import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from RGCN import RGCNLayer
from transformer import TransformerEncoder

class GraphEncoder(nn.Module):
    def __init__(self, ent_dim, num_rels, num_bases, dropout=0.0):
        """这个部分用于单个snapshot的编码"""
        super(GraphEncoder, self).__init__()
        self.layer1 = RGCNLayer(ent_dim, ent_dim, num_rels, num_bases, True, torch.nn.functional.relu, True, dropout)
        self.layer2 = RGCNLayer(ent_dim, ent_dim, num_rels, num_bases, True, None, True, dropout)

    def forward(self, g):
        """g: 需要编码的snapshot图"""
        self.layer1(g)
        self.layer2(g)
        return g.ndata['h']

class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """这个部分用于序列信息的编码"""
        super(SequenceEncoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim)

    def forward(self, seq_entities, timestamps):
        output, _ = self.rnn(seq_entities)
        return output

class InforDecoder(nn.Module):
    def __init__(self, input_dim, num_ent):
        """这个部分用于解码得到最后的答案"""
        super(InforDecoder, self).__init__()
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, input_dim, bias=True)
        self.fc = nn.Linear(input_dim, num_ent, bias=True)
        self.dropout = nn.Dropout(0.0)

    def forward(self, features):
        return self.dropout(self.fc(self.act(self.fc1(features))))

class TKGraphormer(nn.Module):
    def __init__(self, config):
        super(TKGraphormer, self).__init__()
        self.n_ent = config.n_ent  # 实体的数量
        self.ent_dim = config.ent_dim  # 实体的嵌入维度
        self.n_rel = config.n_rel  # 关系的数量
        self.rel_dim = config.rel_dim  # 关系的嵌入维度
        self.lstm_hidden_dim = config.lstm_hidden_dim  #  lstm隐藏层维度

        self.ent_embeds = nn.Embedding(self.n_ent, self.ent_dim)
        self.rel_embeds = nn.Embedding(self.n_rel, self.rel_dim)

        self.graph_encoder = GraphEncoder(self.ent_dim, self.n_rel, self.ent_dim // 4, 0.0)
        # self.seq_encoder = SequenceEncoder(self.ent_dim, self.lstm_hidden_dim)
        self.seq_encoder = TransformerEncoder(self.ent_dim, self.ent_dim, 2, 5, self.ent_dim, self.ent_dim, 0.2)
        self.decoder = InforDecoder(self.lstm_hidden_dim + self.rel_dim, self.n_ent)

    # def forward(self, graph_list, query_entities, query_relations, query_timestamps, history_idx):
    #     """
    #     graph_list: 可见的是snapshots图，用于编码预测依据
    #     query_entities: 需要回答的queries中的实体, [batch_size]
    #     query_relations: 需要回答的queries中的关系
    #     query_timestamps: 需要回答的queries中的时间戳
    #     history_idx: [batch_size, history_len]
    #     """
    #     gs_h = []
    #     for g in graph_list:
    #         g.ndata['h'] = self.ent_embeds(g.ndata['id']).view(-1, self.ent_dim)
    #         gs_h.append(self.graph_encoder(g)[query_entities, :])
    #
    #     gs_h = torch.stack(gs_h, dim=0).transpose(0, 1)   #[batch_size(query_entities), graph_num, ent_dim]
    #     seq_ent_embeds = torch.stack([gs_h[i, history_idx[i]] for i in range(history_idx.shape[0])])  # [batch_size, history_len, ent_dim]
    #
    #     lstm_output = self.seq_encoder(seq_ent_embeds, query_timestamps)[:, -1, :]  # 取最后一个输出 [batch_size, lstm_hidden_dim]
    #
    #     query_rel_embeds = self.rel_embeds(query_relations)  # [batch_size, rel_dim]
    #     decoder_input = torch.cat([lstm_output, query_rel_embeds], dim=-1)
    #     output_score = self.decoder(decoder_input)  # [batch_size, num_ent]
    #     return output_score

    def forward(self, graph_list, query_entities, query_relations, query_timestamps):
        """
        graph_list: 可见的是snapshots图，用于编码预测依据
        query_entities: 需要回答的queries中的实体
        query_relations: 需要回答的queries中的关系
        query_timestamps: 需要回答的queries中的时间戳
        """
        gs_query_ent_h = []
        for g in graph_list:
            g.ndata['h'] = self.ent_embeds(g.ndata['id']).view(-1, self.ent_dim)
            g_h = self.graph_encoder(g)
            gs_query_ent_h.append(g_h[query_entities])

        query_rel_embeds = self.rel_embeds(query_relations)  # [batch_size, rel_dim]

        seq_ent_embeds = torch.stack(gs_query_ent_h, dim=1)  # [batch_size, seq_len, ent_dim]
        seq_times = torch.arange(0, seq_ent_embeds.shape[1], device=seq_ent_embeds.device).unsqueeze(0).repeat(seq_ent_embeds.shape[0], 1)
        lstm_output = self.seq_encoder(seq_ent_embeds, seq_times)[:, -1, :]  # 取最后一个输出 [batch_size, lstm_hidden_dim]
        decoder_input = torch.cat([lstm_output, query_rel_embeds], dim=-1)
        output_score = self.decoder(decoder_input)  # [batch_size, num_ent]
        return output_score

