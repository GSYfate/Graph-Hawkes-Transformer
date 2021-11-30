import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphEncoder(nn.Module):
    def __init__(self):
        """这个部分用于单个snapshot的编码"""
        super(GraphEncoder, self).__init__()

    def forward(self, graph_list, ent_emb, rel_emb):
        """
        graph_list: 需要编码的snapshots图
        ent_emb: 图中所有实体的初始化嵌入表征
        rel_emb: 图中所有关系的初始化嵌入表征
        """
        g_hs = []
        for g in graph_list:
            g.ndata['h'] = ent_emb
            g_hs.append(g.ndata.pop('h'))
        return g_hs

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
        self.fc = nn.Linear(input_dim, num_ent)

    def forward(self, features):
        return self.fc(features)

class TKGraphormer(nn.Module):
    def __init__(self, config):
        super(TKGraphormer, self).__init__()
        self.n_ent = config.n_ent  # 实体的数量
        self.ent_dim = config.ent_dim # 实体的嵌入维度
        self.n_rel = config.n_rel  # 关系的数量
        self.rel_dim = config.rel_dim  # 关系的嵌入维度
        self.lstm_hidden_dim = config.lstm_hidden_dim  #  lstm隐藏层维度

        self.ent_embeds = nn.Embedding(self.n_ent, self.ent_dim)
        self.rel_embeds = nn.Embedding(self.n_rel, self.rel_dim)

        self.graph_encoder = GraphEncoder()
        self.seq_encoder = SequenceEncoder(self.ent_dim, self.lstm_hidden_dim)
        self.decoder = InforDecoder(self.lstm_hidden_dim + self.rel_dim, self.n_ent)

    def forward(self, graph_list, query_entities, query_relations, query_timestamps):
        """
        graph_list: 可见的是snapshots图，用于编码预测依据
        query_entities: 需要回答的queries中的实体
        query_relations: 需要回答的queries中的关系
        query_timestamps: 需要回答的queries中的时间戳
        """
        # TODO: 这里忽略图聚合，后续再加. 用一个实体编码序列代替
        query_ent_embeds = self.ent_embeds(query_entities)  # [batch_size, ent_dim]
        query_rel_embeds = self.rel_embeds(query_relations)  # [batch_size, rel_dim]
        seq_ent_embeds = query_ent_embeds.unsqueeze(1).repeat(1, 3, 1)  # [batch_size, seq_len: 3, ent_dim]
        lstm_output = self.seq_encoder(seq_ent_embeds, query_timestamps)[:, -1, :]  # 取最后一个输出
        decoder_input = torch.cat([lstm_output, query_rel_embeds], dim=-1)
        output_score = self.decoder(decoder_input).softmax(dim=-1) # [batch_size, num_ent]
        return output_score

if __name__ == '__main__':
    from collections import namedtuple
    Config = namedtuple('config', ['n_ent', 'ent_dim', 'n_rel', 'rel_dim', 'lstm_hidden_dim'])
    config = Config(n_ent=10, ent_dim=50, n_rel=5, rel_dim=50, lstm_hidden_dim=50)
    model = TKGraphormer(config)
    graph_list = None
    query_entities = torch.tensor([0, 1])
    query_relations = torch.tensor([1, 2])
    query_timestamps = torch.tensor([1, 2])
    print(model.forward(graph_list, query_entities, query_relations, query_timestamps))
