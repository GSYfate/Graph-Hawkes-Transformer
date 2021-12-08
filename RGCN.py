import torch
import torch.nn as nn
import dgl.function as fn
import dgl

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=False,
                 activation=None, self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.num_rels = num_rels    #关系数量，这里用了取反关系，r + num_r
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

if __name__ == '__main__':
    import numpy as np
    num_nodes = 10
    src = np.array([0, 1, 2, 3])
    dst = np.array([1, 3, 3, 4])
    rel = np.array([0, 1, 1, 0])
    num_rels = 2
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))  # 加入取反边
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)

    ent_embeds = torch.nn.Embedding(num_nodes, 10)
    g.ndata['h'] = ent_embeds(g.ndata['id']).view(-1, 10)
    print(g.ndata['h'])
    model = RGCNLayer(10, 10, num_rels * 2, 1, True, torch.nn.functional.relu, True, 0.0)
    model.forward(g)
    print(g.ndata['h'])