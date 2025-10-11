# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
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
import numpy as np
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.utils import not_in_whitelist
from torch.utils.tensorboard import SummaryWriter
from .target_attention import MultiHeadTargetAttention_baseline, MultiHeadTargetAttention_CollectiveKV


class SIM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="SIM", 
                 gpu=-1, 
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_dropout=0,
                 attention_dim=64,
                 num_heads=1,
                 gsu_type="soft",
                 short_seq_len=50,
                 topk=50,
                 alpha=1,
                 beta=1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0,
                 batch_norm=False,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(SIM, self).__init__(feature_map,
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.topk = topk
        self.short_seq_len = short_seq_len
        self.alpha = alpha
        self.beta = beta
        assert gsu_type == "soft", "Only support soft search currently!"
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim) 
        self.accumulation_steps = accumulation_steps
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.W_a = nn.Linear(self.item_info_dim, attention_dim, bias=False)
        self.W_b = nn.Linear(self.item_info_dim, attention_dim, bias=False)
        self.method = kwargs['method']
        if kwargs['method'] == 'baseline':
            self.short_attention = MultiHeadTargetAttention_baseline(self.item_info_dim,
                                                        attention_dim,
                                                        num_heads,
                                                        attention_dropout)
        elif kwargs['method'] == 'CollectiveKV':
            self.short_attention = MultiHeadTargetAttention_CollectiveKV(self.item_info_dim,
                                                                attention_dim,
                                                                num_heads,
                                                                attention_dropout,
                                                                usr_dim=kwargs['usr_dim'],
                                                                global_pool_size=kwargs['pool_size'],
                                                                loss_balance_weight=kwargs['loss_balance_weight'],
                                                                loss_peak_weight=kwargs['loss_peak_weight'],
                                                                share_k=kwargs['share_k'],
                                                                share_v=kwargs['share_v'])
            self.writer = SummaryWriter(log_dir=f"./loss_logs/SIM/{kwargs['expid']}")
        else:
            self.short_attention = MultiHeadTargetAttention_baseline(self.item_info_dim,
                                                        attention_dim,
                                                        num_heads,
                                                        attention_dropout)
        self.long_attention = MultiHeadTargetAttention_baseline(self.item_info_dim,
                                                       attention_dim,
                                                       num_heads,
                                                       attention_dropout)
        input_dim = feature_map.sum_emb_out_dim() + self.item_info_dim
        self.dnn_aux = MLP_Block(input_dim=input_dim,
                                 output_dim=1,
                                 hidden_units=dnn_hidden_units,
                                 hidden_activations=dnn_activations,
                                 output_activation=self.output_activation, 
                                 dropout_rates=net_dropout,
                                 batch_norm=batch_norm)
        input_dim = feature_map.sum_emb_out_dim() + self.item_info_dim * 2
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
        self.count = 0

    def forward(self, inputs):
        return_dict = {}
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        emb_list = []
        if batch_dict: # not empty
            emb_out = self.embedding_layer(batch_dict, flatten_emb=True)
            emb_list.append(emb_out)
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)
        target_emb = item_feat_emb[:, -1, :]

        # short interest attention
        short_seq_emb = item_feat_emb[:, -self.short_seq_len:-1, :]
        short_mask = mask[:, -self.short_seq_len:-1] if mask.shape[1] >= self.short_seq_len else mask
        if self.method == 'CollectiveKV' and self.training:
            short_interest_emb, loss_peak, loss_balance = self.short_attention(target_emb, short_seq_emb, short_mask)
            return_dict['loss_peak'] = loss_peak
            return_dict['loss_balance'] = loss_balance
        else:
            short_interest_emb = self.short_attention(target_emb, short_seq_emb, short_mask)

        # first stage
        long_seq_emb = item_feat_emb[:, 0:-1, :]
        q = self.W_a(target_emb).unsqueeze(1)
        k = self.W_b(long_seq_emb)
        qk = torch.bmm(q, k.transpose(-1, -2)).squeeze(1) * mask
        pooled_u_rep = torch.bmm(qk.unsqueeze(1), long_seq_emb).squeeze(1)
        emb_list += [target_emb, pooled_u_rep]
        y_aux = self.dnn_aux(torch.cat(emb_list, dim=-1))
        topk = min(self.topk, qk.shape[1]) # make sure input seq_len >= topk
        topk_index = qk.topk(topk, dim=1, largest=True, sorted=True)[1]
        topk_emb = torch.gather(long_seq_emb, 1, 
                                topk_index.unsqueeze(-1).expand(-1, -1, long_seq_emb.shape[-1]))
        topk_mask = torch.gather(mask, 1, topk_index)

        # second stage
        long_interest_emb = self.long_attention(target_emb, topk_emb, topk_mask)
        emb_list = emb_list[0:-1] + [short_interest_emb, long_interest_emb]
        feature_emb = torch.cat(emb_list, dim=-1)
        y_pred = self.dnn(feature_emb)
        return_dict["y_pred"] = y_pred
        return_dict["y_aux"] = y_aux
        return return_dict

    def add_loss(self, return_dict, y_true):
        loss_gsu = self.loss_fn(return_dict["y_aux"], y_true, reduction='mean')
        loss_esu = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        return self.alpha * loss_gsu + self.beta * loss_esu
    
    def get_inputs(self, inputs, feature_source=None):
        batch_dict, item_dict, mask = inputs
        X_dict = dict()
        for feature, value in batch_dict.items():
            if feature in self.feature_map.labels:
                continue
            feature_spec = self.feature_map.features[feature]
            if feature_spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            X_dict[feature] = value.to(self.device)
        for item, value in item_dict.items():
            item_dict[item] = value.to(self.device)
        return X_dict, item_dict, mask.to(self.device)

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        batch_dict = inputs[0]
        y = batch_dict[labels[0]].to(self.device)
        return y.float().view(-1, 1)
                
    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def train_step(self, batch_data):
        self.count += 1
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss_main = self.add_loss(return_dict, y_true)
        loss_peak = return_dict.get("loss_peak", 0.0)  # add loss_peak if exists
        loss_balance = return_dict.get("loss_balance", 0.0)  # add loss_balance if exists
        loss = loss_main + loss_peak + loss_balance
        if self.method == 'CollectiveKV' and self.training:
            self.writer.add_scalar("Loss/total", loss.item(), global_step=self.count)
            self.writer.add_scalar("Loss/main", loss_main.item(), global_step=self.count)
            self.writer.add_scalar("Loss/peak", loss_peak.item(), global_step=self.count)
            self.writer.add_scalar("Loss/balance", loss_balance.item(), global_step=self.count)
        loss = loss / self.accumulation_steps
        loss.backward()
        if (self._batch_index + 1) % self.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss
