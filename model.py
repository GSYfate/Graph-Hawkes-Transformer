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
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

class TemporalTransformerHawkesGraphModel(nn.Module):
    def __init__(self, config):
        super(TemporalTransformerHawkesGraphModel, self).__init__()
        self.config = config
        self.n_ent = config.n_ent  # 实体的数量
        self.n_rel = config.n_rel  # 关系的数量
        self.d_model = config.d_model  # 嵌入维度
        self.dropout_rate = config.dropout
        self.transformer_layer_num = config.seqTransformerLayerNum
        self.transformer_head_num = config.seqTransformerHeadNum
        self.PAD_TIME = -1
        self.PAD_ENTITY = self.n_ent

        self.ent_embeds = nn.Embedding(self.n_ent, self.d_model)
        self.rel_embeds = nn.Embedding(self.n_rel, self.d_model)
        self.graph_encoder = RTAGCNEncoder(self.d_model, self.dropout_rate)
        self.seq_encoder = TransformerEncoder(self.d_model, self.d_model, self.transformer_layer_num,
                                              self.transformer_head_num, self.dropout_rate)

        self.linear_inten_layer = nn.Linear(self.d_model * 3, self.d_model, bias=False)
        self.time_inten_layer = nn.Linear(self.d_model * 3, self.d_model, bias=False)
        self.Softplus = nn.Softplus(beta=10)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.time_predict_layer = nn.Linear(self.d_model * 4, 1, bias=False)

        self.lp_loss_fn = LabelSmoothingCrossEntropy(eps=0.1)
        self.tp_loss_fn = nn.MSELoss()

    def forward(self, query_entities, query_relations, history_graphs, history_times, batch_node_ids):
        bs, hist_len = history_times.size(0), history_times.size(1)

        history_graphs.ndata['h'] = self.ent_embeds(history_graphs.ndata['id']).view(-1, self.d_model)
        history_graphs.edata['h'] = self.rel_embeds(history_graphs.edata['type']).view(-1, self.d_model)
        history_graphs.edata['qrh'] = self.rel_embeds(history_graphs.edata['query_rel']).view(-1, self.d_model)
        history_graphs.edata['qeh'] = self.ent_embeds(history_graphs.edata['query_ent']).view(-1, self.d_model)

        gh = self.graph_encoder(history_graphs)

        self.query_rel_embeds = self.rel_embeds(query_relations)
        self.query_ent_embeds = self.ent_embeds(query_entities)

        self.seq_input = gh[batch_node_ids].reshape(bs, hist_len, -1)
        seq_query_input = self.query_rel_embeds.unsqueeze(1)  # [bs, 1, d_model]
        seq_query_time = history_times[:, -1].view(-1, 1)  # [bs, 1]

        self.pad_mask = (history_times == -1).unsqueeze(1)
        self.history_times = history_times
        output = self.seq_encoder(self.seq_input, history_times, seq_query_input, seq_query_time, self.pad_mask)
        self.output = output[:, -1, :]

        inten_raw = self.linear_inten_layer(
            self.dropout(torch.cat((self.query_ent_embeds, self.output, self.query_rel_embeds), dim=-1)))  # [bs, d_model]

        global_intes = inten_raw.mm(self.ent_embeds.weight.transpose(0, 1))  # [bs, ent_num]

        local_h = gh.reshape([bs, -1, self.d_model])  # [bs, max_nodes_num * seq_len, d_model]
        local_intes = torch.matmul(inten_raw.unsqueeze(1), local_h.transpose(1, 2))[:, -1, :]   # [bs, max_nodes_num * seq_len]

        intens = self.Softplus(torch.cat([global_intes, local_intes], dim=-1))

        local_type = history_graphs.ndata['id'].reshape([bs, -1])
        global_type = torch.arange(self.n_ent, device=intens.device).unsqueeze(0).repeat(bs, 1)
        type = torch.cat([global_type, local_type], dim=-1)
        return intens, type

    def link_prediction_loss(self, intens, type, answers):
        intens = torch_scatter.scatter(intens, type, dim=-1, reduce="mean")
        loss = self.lp_loss_fn(intens[:, :-1], answers)
        return loss

    def time_prediction_loss(self, estimate_dt, dur_last):
        loss_dt = self.tp_loss_fn(estimate_dt, dur_last)
        return loss_dt

    def predict_e(self, intens, type):
        output = torch_scatter.scatter(intens, type, dim=-1, reduce="max")   # Link Prediction 得分 也是intensity
        return output[:, :-1]

    def predict_t(self, tail_ent, dur_last, time_scale=24, timestep=0.1, hmax=5):
        n_samples = int(hmax / timestep) + 1  # add 1 to accomodate zero
        #
        dur_last = dur_last / time_scale
        dur_non_zero_idx = (dur_last > 0).nonzero().squeeze()
        dur_last = dur_last[dur_non_zero_idx].type(torch.float)
        if dur_last == torch.Size([]):
            return torch.tensor([0.]), torch.tensor([0.])

        dt = torch.linspace(0, hmax, n_samples, device=dur_last.device).repeat(dur_last.shape[0], 1)  # [bs, n_sample]

        seq_query_input = self.query_rel_embeds[dur_non_zero_idx].unsqueeze(1).repeat(1, n_samples, 1)  # [bs , n_sample, d_model]
        seq_query_time = self.history_times[dur_non_zero_idx, -1].unsqueeze(1).repeat(1, n_samples) + dt  # [bs, n_sample]
        sampled_seq_output = self.seq_encoder(self.seq_input[dur_non_zero_idx], self.history_times[dur_non_zero_idx],
                                              seq_query_input, seq_query_time, self.pad_mask[dur_non_zero_idx])  # [bs, n_sample, d_model]

        inten_layer_input = torch.cat((self.query_ent_embeds[dur_non_zero_idx].unsqueeze(1).repeat(1, n_samples, 1),
                                    sampled_seq_output, self.query_rel_embeds[dur_non_zero_idx].unsqueeze(1).repeat(1, n_samples, 1)), dim=-1)
        inten_raw = self.time_inten_layer(self.dropout(inten_layer_input))  # [bs, n_sample, d_model]
        o = self.ent_embeds(tail_ent[dur_non_zero_idx]).unsqueeze(1).repeat(1, n_samples, 1)  # [bs, d_model]
        intensity = self.Softplus((inten_raw * o).sum(dim=2))  # [bs, n_sample]

        integral_ = torch.cumsum(timestep * intensity, dim=1)
        density = (intensity * torch.exp(-integral_))
        t_pit = dt * density  # [bs, n_sample]
        estimate_dt = (timestep * 0.5 * (t_pit[:, 1:] + t_pit[:, :-1])).sum(dim=1)  # shape: n_batch
        # estimate_dt = self.time_predict_layer(torch.cat([self.query_ent_embeds, self.query_ent_embeds, self.output, self.ent_embeds(tail_ent)], dim=-1))
        # print(estimate_dt)
        return estimate_dt, dur_last

