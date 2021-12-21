import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.Decoder import *
from models.GraphEncoder import *
from models.SequenceEncoder import *

class TKGraphormer(nn.Module):
    def __init__(self, config):
        super(TKGraphormer, self).__init__()
        self.config = config
        self.n_ent = config.n_ent  # 实体的数量
        self.ent_dim = config.ent_dim  # 实体的嵌入维度
        self.n_rel = config.n_rel  # 关系的数量
        self.rel_dim = config.rel_dim  # 关系的嵌入维度
        self.lstm_hidden_dim = config.lstm_hidden_dim  #  lstm隐藏层维度

        self.ent_embeds = torch.nn.Parameter(torch.Tensor(self.n_ent, self.ent_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.ent_embeds)
        self.rel_embeds = torch.nn.Parameter(torch.Tensor(self.n_rel, self.rel_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.rel_embeds)

        self.graph_encoder = RTAGCNEncoder(self.ent_dim, self.rel_dim)

        self.time_gate_weight = nn.Parameter(torch.Tensor(self.ent_dim, self.ent_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(self.ent_dim))
        nn.init.zeros_(self.time_gate_bias)
        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.ent_dim * 2, self.ent_dim)

        self.decoder = MLPCLFDecoder(self.ent_dim + self.rel_dim, self.n_ent, 0.0)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, graph_list, query_entities, query_relations, query_timestamps):
        """
        graph_list: 可见的是snapshots图，用于编码预测依据
        query_entities: 需要回答的queries中的实体
        query_relations: 需要回答的queries中的关系
        query_timestamps: 需要回答的queries中的时间戳
        """
        # gs_query_ent_h = []
        h = F.normalize(self.ent_embeds)
        h_0 = self.rel_embeds
        for i, g in enumerate(graph_list):
            if g.r_to_e.shape[0] == 0:
                continue
            temp_e = self.ent_embeds[g.r_to_e]
            x_input = torch.zeros_like(self.rel_embeds)
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean

            x_input = torch.cat((self.rel_embeds, x_input), dim=1)
            h_0 = self.relation_cell_1(x_input, h_0)
            h_0 = F.normalize(h_0)

            g.ndata['h'] = h[g.ndata['id']].view(-1, self.ent_dim)
            g.edata['h'] = h_0[g.edata['type']].view(-1, self.rel_dim)
            g.edata['qrh'] = g.edata['h']
            current_h = self.graph_encoder(g)
            current_h = F.normalize(current_h)
            time_weight = F.sigmoid(torch.mm(h, self.time_gate_weight) + self.time_gate_bias)
            h = time_weight * current_h + (1 - time_weight) * h

        query_ent_embeds = h[query_entities].view(-1, self.ent_dim)
        query_rel_embeds = h_0[query_relations].view(-1, self.rel_dim)
        output_score = self.decoder(torch.cat([query_ent_embeds, query_rel_embeds], dim=-1))
        return output_score

    def loss(self, score, answers):
        loss = self.criterion(score, answers)
        return loss

    def seq_graph_loss(self, seq_output, seq_graph_output):
        """期望序列编码输出的下一时刻的特征，和原图的聚合后的特征相似"""
        return nn.MSELoss(seq_output, seq_graph_output)

class TemporalTransformerHawkesGraphModel(nn.Module):
    def __init__(self, config):
        super(TemporalTransformerHawkesGraphModel, self).__init__()
        self.config = config
        self.n_ent = config.n_ent  # 实体的数量
        self.ent_dim = config.ent_dim  # 实体的嵌入维度
        self.n_rel = config.n_rel  # 关系的数量
        self.rel_dim = config.rel_dim  # 关系的嵌入维度

        self.ent_embeds = nn.Embedding(self.n_ent, self.ent_dim)
        self.rel_embeds = nn.Embedding(self.n_rel, self.rel_dim)
        self.graph_encoder = RTAGCNEncoder(self.ent_dim, self.rel_dim, 0.0)
        self.seq_encoder = TransformerEncoder(self.ent_dim, self.ent_dim, 1, 2, 0.0)
        self.decoder = MLPCLFDecoder(self.ent_dim + self.rel_dim + self.ent_dim, self.n_ent, 0.0)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, query_entities, query_relations, query_timestamps, history_graphs, history_times, batch_node_ids):
        bs, hist_len = history_times.size(0), history_times.size(1)

        history_graphs.ndata['h'] = self.ent_embeds(history_graphs.ndata['id']).view(-1, self.ent_dim)
        history_graphs.edata['h'] = self.rel_embeds(history_graphs.edata['type']).view(-1, self.rel_dim)
        history_graphs.edata['qrh'] = self.rel_embeds(history_graphs.edata['query_rel']).view(-1, self.rel_dim)
        history_graphs.edata['qeh'] = self.ent_embeds(history_graphs.edata['query_ent']).view(-1, self.ent_dim)

        gh = self.graph_encoder(history_graphs)
        query_gh = gh[batch_node_ids].reshape(bs, hist_len, -1)

        query_rel_embeds = self.rel_embeds(query_relations)
        query_ent_embeds = self.ent_embeds(query_entities)

        seq_input = query_gh
        query_input = query_rel_embeds.unsqueeze(1)

        pad_mask = (history_times==-1).unsqueeze(1)
        output = self.seq_encoder(seq_input, history_times, query_input, query_timestamps.unsqueeze(1), pad_mask)
        output = output[:, -1, :]
        output = self.decoder(torch.cat([query_ent_embeds, output, query_rel_embeds], dim=-1))
        return output

    def loss(self, score, answers):
        loss = self.criterion(score, answers)
        return loss
