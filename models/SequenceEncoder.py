import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.Transformer import EncoderLayer

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """这个部分用于序列信息的编码"""
        super(LSTMEncoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim)

    def forward(self, seq_entities, timestamps):
        output, _ = self.rnn(seq_entities)
        return output

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout):
        super().__init__()
        self.d_model = d_model

        # positionvector, used for temporal encoding
        self.position_vec = torch.nn.Parameter(
            torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)]), requires_grad=False)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time):
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def forward(self, features, times):
        """
        features: 经过聚合后每个时间上的实体特征，[batch_size, seq_len, feature_dim(ent_dim)]
        times: 序列对应的时间戳 [batch_size, seq_len]
        """
        tem_enc = self.temporal_enc(times)

        for enc_layer in self.layer_stack:
            features += tem_enc
            features, _ = enc_layer(features)
        return features