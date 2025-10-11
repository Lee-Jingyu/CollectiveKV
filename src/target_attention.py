# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import torch
from torch import nn
from sklearn.decomposition import PCA
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import os
from datetime import datetime
import time

def cosine_similarity(tensor):
    normalized_tensor = F.normalize(tensor, p=2, dim=1)
    # 计算余弦相似度矩阵
    similarity_matrix = torch.mm(normalized_tensor, normalized_tensor.t())
    return similarity_matrix
  
def load_balance_kl(probs: torch.Tensor,
                    eps: float = 1e-9) -> torch.Tensor:
    """
    KL divergence load balance loss.

    Args:
        probs: (B, S, M) router probabilities before softmax
        eps: numerical stability

    Returns:
        scalar loss
    """
    probs = F.softmax(probs, dim=-1)  # (B, S, M)
    avg_probs = probs.mean(dim=(0, 1))  # (M,)
    avg_probs = avg_probs.clamp_min(eps)
    M = avg_probs.size(0)
    uniform_prob = 1.0 / M

    loss = (avg_probs * (avg_probs / uniform_prob).log()).sum()
    return loss


def loss_peak(router_logits: torch.Tensor,
              topk_indices: torch.Tensor,
              eps: float = 1e-9) -> torch.Tensor:
    """
    Encourages the probability of the selected top-1 expert to approach 1.

    Args:
        router_logits (torch.Tensor): Shape (B, S, M), router network raw outputs before softmax.
        topk_indices (torch.Tensor): Shape (B, S, 1), top-1 expert indices from torch.topk (k=1).
        eps (float): Small constant to avoid log(0).

    Returns:
        torch.Tensor: Scalar loss.
    """
    # Compute softmax probabilities
    probs = F.softmax(router_logits, dim=-1)  # (B, S, M)

    # Gather probabilities of the selected top-1 expert
    top1_probs = probs.gather(-1, topk_indices.detach())  # (B, S, 1)

    # Negative log likelihood of the top-1 expert
    loss = -(top1_probs.clamp_min(eps).log()).mean()

    return loss


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention 
        Ref: https://zhuanlan.zhihu.com/p/47812375
    """
    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, Q, K, V, scale=None, mask=None):
        # mask: 0 for masked positions
        scores = torch.matmul(Q, K.transpose(-1, -2))
        if scale:
            scores = scores / scale
        if mask is not None:
            mask = mask.view_as(scores)
            scores = scores.masked_fill_(mask.float() == 0, -1.e9) # fill -inf if mask=0
        attention = scores.softmax(dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.matmul(attention, V)
        return output, attention


# baseline
class MultiHeadTargetAttention_baseline(nn.Module):
    def __init__(self,
                 input_dim=64,
                 attention_dim=64,
                 num_heads=1,
                 dropout_rate=0,
                 use_scale=True,
                 use_qkvo=True):
        super(MultiHeadTargetAttention_baseline, self).__init__()
        if not use_qkvo:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.use_qkvo = use_qkvo
        if use_qkvo:
            self.W_q = nn.Linear(input_dim, attention_dim, bias=False)
            self.W_k = nn.Linear(input_dim, attention_dim, bias=False)
            self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
            self.W_o = nn.Linear(attention_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention(dropout_rate)
        self.test_time = []
        self.count = 0
        

    def forward(self, target_item, history_sequence, mask=None):
        """
        target_item: b x emd
        history_sequence: b x len x emb
        mask: mask of history_sequence, 0 for masked positions
        """
        b, s, d = history_sequence.shape
        # linear projection
        if self.use_qkvo:
            query = self.W_q(target_item)
            key = self.W_k(history_sequence)
            value = self.W_v(history_sequence)
        else:
            query, key, value = target_item, history_sequence, history_sequence
        
        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        if mask is not None:
            mask = mask.view(batch_size, 1, 1, -1).expand(-1, self.num_heads, -1, -1)

        # scaled dot product attention
        output, _ = self.dot_attention(query, key, value, scale=self.scale, mask=mask)
        # concat heads
        output = output.transpose(1, 2).contiguous().view(-1, self.num_heads * self.head_dim)
        if self.use_qkvo:
            output = self.W_o(output)
        return output



class MultiHeadTargetAttention_CollectiveKV(nn.Module):
    def __init__(self,
                 input_dim=64,
                 attention_dim=64,
                 num_heads=1,
                 dropout_rate=0,
                 use_scale=True,
                 use_qkvo=True,
                 global_pool_size=10000,
                 usr_dim=1,
                 loss_balance_weight=1.0,
                 loss_peak_weight=1.0,
                 share_k=True,
                 share_v=True):
        '''
        Params:
        ------
        usr_dim: dimension of user-specific key and value
        share_k: whether to activate key sharing
        share_v: whether to activate value sharing
        ------
        两个超参数的额外说明：
        1、usr_dim + global_dim = attention_dim
            这里usr_dim需要多次尝试才能找到最优，经验表明，usr_dim过大表现不好，建议设置在10维以内。
        2、global_pool_size: 全局共享KV pool的大小，增加该值可能会略微提升效果，但会显著增加训练显存占用。
            可能与user数正相关。建议从较低的值开始尝试。
        '''
        super().__init__()
        if not use_qkvo:
            attention_dim = input_dim
        assert attention_dim % num_heads == 0, \
               "attention_dim={} is not divisible by num_heads={}".format(attention_dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.scale = self.head_dim ** 0.5 if use_scale else None
        self.use_qkvo = use_qkvo
        if use_qkvo:
            self.W_q = nn.Linear(input_dim, attention_dim, bias=False)            
            self.usr_dim = usr_dim
            self.global_dim = attention_dim - usr_dim
            
            self.share_k = share_k
            self.share_v = share_v
            if share_k:
                self.W_k = nn.Linear(input_dim, self.usr_dim, bias=False)  # global key
                self.router_K = nn.Linear(input_dim, global_pool_size, bias=False)  # gate for global value
                self.global_K_pool = nn.Parameter(torch.randn(global_pool_size, self.global_dim), requires_grad=True)
            else:
                self.W_k = nn.Linear(input_dim, attention_dim, bias=False) 
            
            if share_v:
                self.W_v = nn.Linear(input_dim, self.usr_dim, bias=False)  # global value
                self.router_V = nn.Linear(input_dim, global_pool_size, bias=False)  # gate for global value
                self.global_V_pool = nn.Parameter(torch.randn(global_pool_size, self.global_dim), requires_grad=True)
            else:
                self.W_v = nn.Linear(input_dim, attention_dim, bias=False)
            
            self.sigmoid = nn.Sigmoid()
            
            self.W_o = nn.Linear(attention_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention(dropout_rate)
        self.sparse_W_q = None
        self.loss_balance_weight = loss_balance_weight
        self.loss_peak_weight = loss_peak_weight
        self.test_time = []
        self.count = 0
        

    def forward(self, target_item, history_sequence, mask=None, use_silu=True):
        """
        target_item: b x emd
        history_sequence: b x len x emb
        mask: mask of history_sequence, 0 for masked positions       
        """
        device = target_item.device
        b, s, emb = history_sequence.size()
        batch_size = target_item.size(0)
        query = self.W_q(target_item)  
        if self.use_qkvo:
            if not self.share_k:
                key = self.W_k(history_sequence)
                loss_balance = 0
                loss_peak = 0
            else:
                # usr specific key
                usr_key = F.silu(self.W_k(history_sequence))    # can be cached
                # global key
                router_K = self.router_K(history_sequence)  # b x s x global_pool_size
                if self.training:
                    # 训练时要乘上权重，保证梯度回传到router
                    # 为了与测试时保持一致，增加loss_peak使sigmoid权重接近1
                    # 同时引入了loss_balance进行负载均衡
                    topk_weights_K, topk_indices_K = torch.topk(router_K, k=1, dim=-1)  
                    selected_experts_K = self.global_K_pool[topk_indices_K].squeeze()  
                    topk_weights_K = self.sigmoid(topk_weights_K)  # b x s x 1
                    global_key = topk_weights_K * selected_experts_K
                    key = torch.cat([global_key, usr_key], dim=-1)
                    
                    loss_peak = -topk_weights_K.log().mean() # 鼓励最大值接近1
                    loss_peak = self.loss_peak_weight * loss_peak
                    loss_balance = load_balance_kl(router_K)
                    loss_balance = self.loss_balance_weight * loss_balance
                else:
                    topk_indices_K = torch.argmax(router_K, dim=-1)       # can be cached             
                    if b == 1:
                        global_key = self.global_K_pool[topk_indices_K]  
                    else:
                        global_key = self.global_K_pool[topk_indices_K].squeeze()  
                    key = torch.cat([global_key, usr_key], dim=-1)                
            
            if not self.share_v:
                value = self.W_v(history_sequence)
            else:
                # usr specific value
                usr_value = F.silu(self.W_v(history_sequence))  # can be cached
                # global value
                router_V = self.router_V(history_sequence)  # b x s x global_pool_size
                if self.training: # 与key是一样的
                    topk_weights_V, topk_indices_V = torch.topk(router_V, k=1, dim=-1)  
                    selected_experts_V = self.global_V_pool[topk_indices_V].squeeze()  
                    topk_weights_V = self.sigmoid(topk_weights_V)  # b x s x 1
                    global_value = topk_weights_V * selected_experts_V

                    # 增加loss监督，使最大的weight接近1，同时尽量负载均衡
                    loss_peak_v = -topk_weights_V.log().mean() # 鼓励最大值接近1
                    loss_peak += self.loss_peak_weight * loss_peak_v
                    loss_balance_v = load_balance_kl(router_V)
                    loss_balance += self.loss_balance_weight * loss_balance_v
                else:
                    topk_indices_V = torch.argmax(router_V, dim=-1)      # can be cached
                    if b == 1:
                        selected_experts_V = self.global_V_pool[topk_indices_V]
                    else:
                        selected_experts_V = self.global_V_pool[topk_indices_V].squeeze()  
                    global_value = selected_experts_V
                value = torch.cat([global_value, usr_value], dim=-1)
        else:
            key, value = history_sequence, history_sequence
        

        if use_silu:
            query = F.silu(query)
            key = F.silu(key)
            value = F.silu(value)
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        if mask is not None:
            mask = mask.view(batch_size, 1, 1, -1).expand(-1, self.num_heads, -1, -1)

        # scaled dot product attention
        output, _ = self.dot_attention(query, key, value, scale=self.scale, mask=mask)
        # concat heads
        output = output.transpose(1, 2).contiguous().view(-1, self.num_heads * self.head_dim)
        if self.use_qkvo:
            output = self.W_o(output)
        
        if self.training:
            return output, loss_peak, loss_balance
        else:
            return output
