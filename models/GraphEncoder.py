import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class RTAGCNLayer(nn.Module):
    def __init__(self, d_model, drop=0.1):
        super(RTAGCNLayer, self).__init__()
        self.d_model = d_model
        self.msg_fc = nn.Linear(self.d_model * 2, self.d_model, bias=False)

        self.qw = nn.Linear(self.d_model, self.d_model, bias=False)
        self.kw = nn.Linear(self.d_model * 2, self.d_model, bias=False)
        self.temp = self.d_model ** 0.5

        self.layer_norm1 = nn.LayerNorm(self.d_model, eps=1e-6)
        self.dropout = nn.Dropout(drop)

        nn.init.xavier_uniform_(self.qw.weight)
        nn.init.xavier_uniform_(self.kw.weight)
        nn.init.xavier_uniform_(self.msg_fc.weight)

    def msg_func(self, edges):
        msg = self.msg_fc(torch.cat([edges.src['h'], edges.data['h']], dim=-1))
        msg = F.leaky_relu(msg)   # 这个激活函数确实需要，但是没有测试用哪种激活函数比较好
        msg = self.dropout(msg)
        q = self.qw(edges.data['qrh']) / self.temp
        k = self.kw(torch.cat([edges.src['h'], edges.data['h']], dim=-1))
        att = torch.sum(q * k, dim=-1).unsqueeze(1)
        return {'msg': msg, 'att': att}

    def reduce_func(self, nodes):
        res = nodes.data['h']
        alpha = self.dropout(F.softmax(nodes.mailbox['att'], dim=1))
        h = torch.sum(alpha * nodes.mailbox['msg'], dim=1)
        h = h + res
        h = self.layer_norm1(h)
        return {'h': h}

    def forward(self, g):
        g.update_all(self.msg_func, self.reduce_func)
        return g

class RTAGCNEncoder(nn.Module):
    def __init__(self, d_model, drop=0.1):
        super(RTAGCNEncoder, self).__init__()
        self.layer1 = RTAGCNLayer(d_model, drop)
        self.layer2 = RTAGCNLayer(d_model, drop)

    def forward(self, g):
        """g: 需要编码的snapshot图"""
        self.layer1(g)
        self.layer2(g)
        return g.ndata['h']
