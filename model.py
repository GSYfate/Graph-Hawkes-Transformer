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

        self.ent_embeds = nn.Embedding(self.n_ent, self.ent_dim)
        self.rel_embeds = nn.Embedding(self.n_rel, self.rel_dim)

        if config.graphEncoder == 'RGTEncoder':
            self.graph_encoder = RGTEncoder(self.ent_dim, self.rel_dim, 0.0)
        else:
            self.graph_encoder = RGCNEncoder(self.ent_dim, self.n_rel, self.ent_dim // 4, 0.0)

        if config.sequenceEncoder == 'TransformerEncoder':
            self.seq_encoder = TransformerEncoder(self.ent_dim, self.ent_dim, 2, 5, self.ent_dim, self.ent_dim, 0.2)
        else:
            self.seq_encoder = LSTMEncoder(self.ent_dim, self.lstm_hidden_dim)

        if config.decoder == 'DistMultDecoder':
            self.decoder = DistMultDecoder(0.0)
        else:
            self.decoder = MLPCLFDecoder(self.lstm_hidden_dim + self.rel_dim, self.n_ent)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, graph_list, query_entities, query_relations, query_timestamps):
        """
        graph_list: 可见的是snapshots图，用于编码预测依据
        query_entities: 需要回答的queries中的实体
        query_relations: 需要回答的queries中的关系
        query_timestamps: 需要回答的queries中的时间戳
        """
        # gs_query_ent_h = []
        gs_ent_h = []
        for g in graph_list:
            g.ndata['h'] = self.ent_embeds(g.ndata['id']).view(-1, self.ent_dim)
            g.edata['h'] = self.rel_embeds(g.edata['type']).view(-1, self.rel_dim)
            g_h = self.graph_encoder(g)
            gs_ent_h.append(g_h)
            # gs_query_ent_h.append(g_h[query_entities])

        query_rel_embeds = self.rel_embeds(query_relations)  # [batch_size, rel_dim]

        # seq_ent_embeds = torch.stack(gs_query_ent_h, dim=1)  # [batch_size, seq_len, ent_dim]

        seq_ent_embeds = torch.stack(gs_ent_h, dim=0).transpose(0, 1)  # [ent_num, seq_len, ent_dim]

        seq_times = torch.arange(0, seq_ent_embeds.shape[1], device=seq_ent_embeds.device).unsqueeze(0).repeat(
            seq_ent_embeds.shape[0], 1)
        # seq_output = self.seq_encoder(seq_ent_embeds, seq_times)[:, -1, :]  # 取最后一个输出 [batch_size, lstm_hidden_dim]

        seq_output = self.seq_encoder(seq_ent_embeds, seq_times)  #[ent_num, seq_len, lstm_hidden_dim]

        query_ents_rep = seq_output[query_entities][:, -1, :]  # [batch_size, lstm_hidden_dim]

        if self.config.decoder == 'DistMultDecoder':
            output_score = self.decoder(query_ents_rep, query_rel_embeds, seq_output[:, -1, :])
        else:
            decoder_input = torch.cat([query_ents_rep, query_rel_embeds], dim=-1)
            output_score = self.decoder(decoder_input)  # [batch_size, num_ent]
        return output_score

    def loss(self, score, answers):
        loss = self.criterion(score, answers)
        return loss

    def seq_graph_loss(self, seq_output, seq_graph_output):
        """期望序列编码输出的下一时刻的特征，和原图的聚合后的特征相似"""
        return nn.MSELoss(seq_output, seq_graph_output)
