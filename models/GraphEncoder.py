import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, skip_connect=False, dropout=0.0, layer_norm=False):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.layer_norm = layer_norm

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            # self.loop_weight = nn.Parameter(torch.eye(out_feat), requires_grad=False)

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(self.skip_connect_weight,
                                    gain=nn.init.calculate_gain('relu'))

            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if self.layer_norm:
            self.normalization_layer = nn.LayerNorm(out_feat, elementwise_affine=False)

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, prev_h=[]):
        if self.self_loop:
            #print(self.loop_weight)
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)
        # self.skip_connect_weight.register_hook(lambda g: print("grad of skip connect weight: {}".format(g)))
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1
            # print("skip_ weight")
            # print(skip_weight)
            # print("skip connect weight")
            # print(self.skip_connect_weight)
            # print(torch.mm(prev_h, self.skip_connect_weight))

        self.propagate(g)  # 这里是在计算从周围节点传来的信息

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        # print(len(prev_h))
        if len(prev_h) != 0 and self.skip_connect:   # 两次计算loop_message的方式不一样，前者激活后再加权
            previous_node_repr = (1 - skip_weight) * prev_h
            if self.activation:
                node_repr = self.activation(node_repr)
            if self.self_loop:
                if self.activation:
                    loop_message = skip_weight * self.activation(loop_message)
                else:
                    loop_message = skip_weight * loop_message
                node_repr = node_repr + loop_message
            node_repr = node_repr + previous_node_repr
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.layer_norm:
                node_repr = self.normalization_layer(node_repr)
            if self.activation:
                node_repr = self.activation(node_repr)
            # print("node_repr")
            # print(node_repr)
        g.ndata['h'] = node_repr
        return node_repr


class UnionRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1,  bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
        super(UnionRGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect
        self.ob = None
        self.sub = None

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(self.skip_connect_weight,gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def forward(self, g, prev_h, emb_rel):
        self.rel_emb = emb_rel
        # self.sub = sub
        # self.ob = ob
        if self.self_loop:
            #loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            # masked_index = torch.masked_select(torch.arange(0, g.number_of_nodes(), dtype=torch.long), (g.in_degrees(range(g.number_of_nodes())) > 0))
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
                (g.in_degrees(range(g.number_of_nodes())) > 0))
            loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1

        # calculate the neighbor message with weight_neighbor
        self.propagate(g)
        node_repr = g.ndata['h']

        # print(len(prev_h))
        if len(prev_h) != 0 and self.skip_connect:  # 两次计算loop_message的方式不一样，前者激活后再加权
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr

    def msg_func(self, edges):
        # if reverse:
        #     relation = self.rel_emb.index_select(0, edges.data['type_o']).view(-1, self.out_feat)
        # else:
        #     relation = self.rel_emb.index_select(0, edges.data['type_s']).view(-1, self.out_feat)
        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        edge_type = edges.data['type']
        edge_num = edge_type.shape[0]
        node = edges.src['h'].view(-1, self.out_feat)
        # node = torch.cat([torch.matmul(node[:edge_num // 2, :], self.sub),
        #                  torch.matmul(node[edge_num // 2:, :], self.ob)])
        # node = torch.matmul(node, self.sub)

        # after add inverse edges, we only use message pass when h as tail entity
        # 这里计算的是每个节点发出的消息，节点发出消息时其作为头实体
        # msg = torch.cat((node, relation), dim=1)
        msg = node + relation
        # calculate the neighbor message with weight_neighbor
        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}

class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, encoder_name="", opn="sub", rel_emb=None, use_cuda=False, analysis=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_basis = num_basis
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.skip_connect = skip_connect
        self.self_loop = self_loop
        self.encoder_name = encoder_name
        self.use_cuda = use_cuda
        self.run_analysis = analysis
        self.skip_connect = skip_connect
        print("use layer :{}".format(encoder_name))
        self.rel_emb = rel_emb
        self.opn = opn
        # create rgcn layers
        self.build_model()
        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):

            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        print("h before GCN message passing")
        print(g.ndata['h'])
        print("h behind GCN message passing")
        for layer in self.layers:
            layer(g)
        print(g.ndata['h'])
        return g.ndata.pop('h')



class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')

class RGTLayer(nn.Module):
    def __init__(self, ent_dim, rel_dim, d_hid, n_heads, dropout=0.0):
        super(RGTLayer, self).__init__()
        self.n_heads = n_heads   # att head数量
        self.d_k = ent_dim // n_heads
        self.temperature = self.d_k ** 0.5

        self.w_qs = nn.Linear(ent_dim*2 + rel_dim, self.n_heads * self.d_k, bias=False)
        self.w_ks = nn.Linear(ent_dim*2 + rel_dim, self.n_heads * self.d_k, bias=False)
        self.w_vs = nn.Linear(ent_dim*2 + rel_dim, self.n_heads * self.d_k, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.n_heads * self.d_k, ent_dim)
        self.layer_norm = nn.LayerNorm(ent_dim)

        self.w_1 = nn.Linear(ent_dim, d_hid)
        self.w_2 = nn.Linear(d_hid, ent_dim)

    def msg_func(self, edges):
        node_src = edges.src['h']
        node_dst = edges.dst['h']
        edge_h = edges.data['h']
        feat = torch.cat([node_src, node_dst, edge_h], dim=-1)
        v = self.w_vs(feat)
        q = self.w_qs(feat)
        k = self.w_ks(feat)
        return {'v': v, 'q': q, 'k': k}

    def reduce_func(self, nodes):
        residual = nodes.mailbox['q']  # [N, D, self.n_heads * self.d_k]
        N, D = residual.size(0), residual.size(1)
        q = nodes.mailbox['q'].view(N, D, self.n_heads, self.d_k)
        k = nodes.mailbox['k'].view(N, D, self.n_heads, self.d_k)
        v = nodes.mailbox['v'].view(N, D, self.n_heads, self.d_k)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(F.softmax(attn, dim=-1))  # [N, n_heads, D]
        output = torch.matmul(attn, v)  # [N, n_heads, D, d_k]

        output = output.transpose(1, 2).contiguous().view(N, D, -1)  # [N, D, n_heads * d_k]
        output = self.dropout(self.fc(output))  # [N, D, ent_dim]
        output += residual
        output = self.layer_norm(output)  # [N, D, ent_dim]
        x = torch.mean(output, dim=1)  # [N, ent_dim]

        residual = x
        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual
        x = self.layer_norm(x)

        return {'h': x}

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), self.reduce_func, self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}

    def forward(self, g):
        loop_message = g.ndata['h']
        self.propagate(g)
        g.ndata['h'] += loop_message
        return g

class RGTEncoder(nn.Module):
    def __init__(self, ent_dim, rel_dim, dropout=0.0):
        super(RGTEncoder, self).__init__()
        self.layer1 = RGTLayer(ent_dim, rel_dim, ent_dim, 5, dropout)
        self.layer2 = RGTLayer(ent_dim, rel_dim, ent_dim, 5, dropout)

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


class RTAGCNLayer(nn.Module):
    def __init__(self, node_h_dim, edge_h_dim, drop=0.1):
        super(RTAGCNLayer, self).__init__()
        self.node_h_dim = node_h_dim
        self.edge_h_dim = edge_h_dim
        self.msg_fc = nn.Linear(self.node_h_dim + self.edge_h_dim, self.node_h_dim, bias=False)

        self.qw = nn.Linear(self.edge_h_dim, self.node_h_dim, bias=False)
        self.kw = nn.Linear(self.node_h_dim + self.edge_h_dim, self.node_h_dim, bias=False)
        self.temp = self.node_h_dim ** 0.5

        self.layer_norm1 = nn.LayerNorm(self.node_h_dim, eps=1e-6)
        self.output_fc = nn.Linear(self.node_h_dim * 2, self.node_h_dim)
        self.dropout = nn.Dropout(drop)

    def msg_func(self, edges):
        msg = self.msg_fc(torch.cat([edges.src['h'], edges.data['h']], dim=-1))
        q = self.qw(edges.data['qrh']) / self.temp
        k = self.kw(torch.cat([edges.src['h'], edges.data['h']], dim=-1))
        att = torch.sum(q * k, dim=-1).unsqueeze(1)
        return {'msg': F.leaky_relu(msg), 'att': att}

    def reduce_func(self, nodes):
        res = nodes.data['h']
        alpha = F.softmax(nodes.mailbox['att'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['msg'], dim=1)
        h = h + res
        return {'h': h}

    def forward(self, g):
        g.update_all(self.msg_func, self.reduce_func)
        return g

class RTAGCNEncoder(nn.Module):
    def __init__(self, node_h_dim, edge_h_dim, drop=0.1):
        super(RTAGCNEncoder, self).__init__()
        self.layer1 = RTAGCNLayer(node_h_dim, edge_h_dim, drop)
        self.layer2 = RTAGCNLayer(node_h_dim, edge_h_dim, drop)

    def forward(self, g):
        """g: 需要编码的snapshot图"""
        self.layer1(g)
        self.layer2(g)
        return g.ndata['h']

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
    rel_embeds = torch.nn.Embedding(num_rels*2, 10)
    g.ndata['h'] = ent_embeds(g.ndata['id']).view(-1, 10)
    g.edata['h'] = rel_embeds(g.edata['type']).view(-1, 10)
    print(g.ndata['h'])
    model = RGTLayer(10, 10, 10, 2, 0.0)
    model.forward(g)
    print(g.ndata['h'])