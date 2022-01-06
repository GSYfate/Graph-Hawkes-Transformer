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


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=False,
                 activation=None, self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.num_rels = num_rels    # 关系数量，这里用了取反关系，r + num_r
        self.num_bases = num_bases   # base数量，用于减少计算量
        self.bias = bias   # 信息聚合是否有偏置
        self.self_loop = self_loop   # 是否需要自环边
        self.activation = activation   # 激活函数
        self.out_feat = out_feat   # 输出特征维度

        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        # 关系参数矩阵
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # 偏置参数
        if self.bias:
            self.bias_parm = nn.Parameter(torch.zeros(out_feat))

        # 自环参数
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(-1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}

    def forward(self, g):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            loop_message = self.dropout(loop_message)

        self.propagate(g)

        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias_parm
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr
        return g

class RGCNEncoder(nn.Module):
    def __init__(self, ent_dim, num_rels, num_bases, dropout=0.0):
        """这个部分用于单个snapshot的编码"""
        super(RGCNEncoder, self).__init__()
        self.layer1 = RGCNLayer(ent_dim, ent_dim, num_rels, num_bases, True, torch.nn.functional.relu, True, dropout)
        self.layer2 = RGCNLayer(ent_dim, ent_dim, num_rels, num_bases, True, None, True, dropout)

    def forward(self, g):
        """g: 需要编码的snapshot图"""
        self.layer1(g)
        self.layer2(g)
        return g.ndata['h']
