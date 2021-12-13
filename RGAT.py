import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x = torch.sum(output, dim=1)  # [N, ent_dim]

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
