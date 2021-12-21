import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.Transformer import *

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """这个部分用于序列信息的编码"""
        super(LSTMEncoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, seq_entities, timestamps, tgt, tgt_time):
        output, _ = self.rnn(seq_entities)
        return output

class TimeEncoding(nn.Module):
    def __init__(self, dim_t):
        super(TimeEncoding, self).__init__()
        # self.w = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_t))).float())
        self.w = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_t))).float(), requires_grad=True)
        # nn.init.xavier_uniform_(self.w)

    def forward(self, dt):
        batch_size = dt.size(0)
        seq_len = dt.size(1)
        dt = dt.view(batch_size, seq_len, 1)
        t_cos = torch.cos(self.w.view(1, 1, -1) * dt)
        t_sin = torch.sin(self.w.view(1, 1, -1) * dt)
        return t_cos, t_sin

class TempMultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """
    def __init__(self, n_head, d_model, d_k, d_v, dim_t, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_t = dim_t

        self.time_encoding = TimeEncoding(dim_t)
        self.q2tw = nn.Linear(d_model, n_head * dim_t, bias=False)
        nn.init.xavier_uniform_(self.q2tw.weight)

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=(d_k + dim_t) ** 0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, q_time, k_time, mask=None):
        """
        q: [batch_size, seq_len, d_model]
        k: [batch_size, seq_len, d_model]
        v: [batch_size, seq_len, d_model]
        q_time: [batch_size, seq_len]
        k_time: [batch_size, seq_len]
        v_time: [batch_size, seq_len]
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # residual = q

        qtw = self.q2tw(q).view(sz_b, len_q, n_head, self.d_t).transpose(1, 2)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k).transpose(1, 2)  # [batch_size, n_head, seq_len, d_k]
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k).transpose(1, 2)  # [batch_size, n_head, seq_len, d_k]
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v).transpose(1, 2)  # [batch_size, n_head, seq_len, d_v]

        q_time_cos, q_time_sin = self.time_encoding(q_time)   # [batch_size, seq_len, d_t]
        k_time_cos, k_time_sin = self.time_encoding(k_time)

        q_time_cos, q_time_sin = qtw * q_time_cos.unsqueeze(1), qtw * q_time_sin.unsqueeze(1)
        k_time_cos, k_time_sin = qtw * k_time_cos.unsqueeze(1), qtw * k_time_sin.unsqueeze(1)

        q = torch.cat([q, q_time_cos, q_time_sin], dim=-1)
        k = torch.cat([k, k_time_cos, k_time_sin], dim=-1)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        # output += residual

        # output = self.layer_norm(output)
        return output, attn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention_layer = TempMultiHeadAttention(n_head, d_model, d_model, d_model, d_model, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_inner, dropout, False)

    def forward(self, src, src_time, tgt, tgt_time, mask=None):
        """src: [batch_size, seq_len, d_model],
        src_time: [batch_size, seq_len]
        tgt: [batch_size, seq_len, d_model]
        tgt_time: [batch_size, seq_len]
        """
        output, _ = self.attention_layer(tgt, src, src, tgt_time, src_time, mask)
        output = self.ff(output)
        return output

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_inner, n_layers, n_head, dropout):
        super(TransformerEncoder, self).__init__()
        self.n_head = n_head
        self.layer_stack = nn.ModuleList([
            TransformerEncoderLayer(d_model, d_inner, n_head, dropout)
            for _ in range(n_layers)])

    def forward(self, src, src_time, tgt, tgt_time, mask=None):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, dim]
            timestamps: [batch_size, seq_len]  int
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        for enc_layer in self.layer_stack:
            tgt = enc_layer(src, src_time, tgt, tgt_time, mask)
        return tgt
