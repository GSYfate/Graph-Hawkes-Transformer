import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.GraphEncoder import RTAGCNEncoder
from models.SequenceEncoder import TransformerEncoder
import torch_scatter

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

class TemporalTransformerHawkesGraphModel(nn.Module):
    def __init__(self, config):
        super(TemporalTransformerHawkesGraphModel, self).__init__()
        self.config = config
        self.n_ent = config.n_ent  # 实体的数量
        self.n_rel = config.n_rel  # 关系的数量
        self.d_model = config.d_model  # 嵌入维度
        self.dropout_rate = 0.2
        self.transformer_layer_num = 1
        self.transformer_head_num = 1

        self.ent_embeds = nn.Embedding(self.n_ent, self.d_model)
        self.rel_embeds = nn.Embedding(self.n_rel, self.d_model)
        self.graph_encoder = RTAGCNEncoder(self.d_model, self.dropout_rate)
        self.seq_encoder = TransformerEncoder(self.d_model, self.d_model, self.transformer_layer_num,
                                              self.transformer_head_num, self.dropout_rate)

        self.seq_linear = nn.Linear(self.d_model * 3, self.d_model, bias=False)

        self.linear_inten_layer = nn.Linear(self.d_model * 3, self.d_model, bias=False)
        self.Softplus = nn.Softplus(beta=10)
        self.dropout = nn.Dropout(self.dropout_rate)

        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = LabelSmoothingCrossEntropy(eps=0.1)

    def forward(self, query_entities, query_relations, query_timestamps, history_graphs, history_times, batch_node_ids):
        bs, hist_len = history_times.size(0), history_times.size(1)

        history_graphs.ndata['h'] = self.ent_embeds(history_graphs.ndata['id']).view(-1, self.d_model)
        history_graphs.edata['h'] = self.rel_embeds(history_graphs.edata['type']).view(-1, self.d_model)
        history_graphs.edata['qrh'] = self.rel_embeds(history_graphs.edata['query_rel']).view(-1, self.d_model)
        history_graphs.edata['qeh'] = self.ent_embeds(history_graphs.edata['query_ent']).view(-1, self.d_model)

        gh = self.graph_encoder(history_graphs)

        #### graph mean pool
        # query_gh = gh.reshape([bs, hist_len, -1, self.d_model])
        # node_type = history_graphs.ndata['id'].reshape([bs, hist_len, -1])
        # node_mask = (node_type == self.n_ent)
        # query_gh = query_gh.masked_fill(node_mask.unsqueeze(-1), 0)
        # query_gh = torch.sum(query_gh, dim=2)  # [bs, hist_len, d_model]
        # query_gh = query_gh / torch.sum(~node_mask, dim=-1).unsqueeze(-1)
        query_gh = gh[batch_node_ids].reshape(bs, hist_len, -1)

        query_rel_embeds = self.rel_embeds(query_relations)
        query_ent_embeds = self.ent_embeds(query_entities)

        # seq_input = torch.cat([query_gh, query_rel_embeds.unsqueeze(1).repeat(1, hist_len, 1),
        #                        query_ent_embeds.unsqueeze(1).repeat(1, hist_len, 1)], dim=-1)
        # seq_input = F.leaky_relu(self.seq_linear(seq_input))
        seq_input = query_gh
        query_input = query_rel_embeds.unsqueeze(1)

        pad_mask = (history_times==-1).unsqueeze(1)
        output = self.seq_encoder(seq_input, history_times, query_input, query_timestamps.unsqueeze(1), pad_mask)
        output = output[:, -1, :]

        inten_raw = self.linear_inten_layer(
            self.dropout(torch.cat((query_ent_embeds, output, query_rel_embeds), dim=-1)))  # [bs, d_model]

        global_intes = inten_raw.mm(self.ent_embeds.weight.transpose(0, 1))  # [bs, ent_num]
        # global_intes = nn.Dropout(1.0)(global_intes)

        local_h = gh.reshape([bs, -1, self.d_model])  # [bs, max_nodes_num * seq_len, d_model]
        local_intes = torch.matmul(inten_raw.unsqueeze(1), local_h.transpose(1, 2))[:, -1, :]   # [bs, max_nodes_num * seq_len]

        intens = self.Softplus(torch.cat([global_intes, local_intes], dim=-1))

        local_type = history_graphs.ndata['id'].reshape([bs, -1])
        global_type = torch.arange(self.n_ent, device=intens.device).unsqueeze(0).repeat(bs, 1)
        type = torch.cat([global_type, local_type], dim=-1)

        return intens, type

    def loss(self, intens, type, answers):
        intens = torch_scatter.scatter(intens, type, dim=-1, reduce="mean")
        loss = self.loss_fn(intens[:, :-1], answers)
        return loss

    def predict(self, query_entities, query_relations, query_timestamps, history_graphs, history_times, batch_node_ids):
        intens, type = self.forward(query_entities, query_relations, query_timestamps, history_graphs, history_times, batch_node_ids)
        # intens[:, :self.n_ent] = 0
        output = torch_scatter.scatter(intens, type, dim=-1, reduce="max")  # Link Prediction 得分

        return output[:, :-1]

