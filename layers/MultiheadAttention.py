import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1234)

class MultiheadAttention(nn.Module):
    def __init__(self, d_model=200, num_head=4):
        super(MultiheadAttention, self).__init__()

        self.d_k = d_model
        self.num_head = num_head
        self.d_spl = d_model // num_head

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(self.d_spl, self.d_spl)

    def forward(self, q, k, v, mask):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        num_split = self.d_k // self.num_head

        split_q = torch.split(q, num_split, dim=2)
        split_k = torch.split(k, num_split, dim=2)
        split_v = torch.split(v, num_split, dim=2)

        buff = []
        buff_weights = []

        mask = mask.unsqueeze(1)
        for sq, sk, sv in zip(split_q, split_k, split_v):
            weights = torch.matmul(sq, sk.transpose(1, 2)
                                   ) / math.sqrt(num_split)
            weights = weights.masked_fill(mask == 0, -1e9)

            normalized_weights = F.softmax(weights, dim=-1)

            output = torch.matmul(normalized_weights, sv)
            output = self.out(output)

            buff.append(output)
            buff_weights.append(weights)

        output = torch.cat(buff, dim=2)
        normalized_weights = torch.cat(buff_weights, dim=2)
        return output, normalized_weights
