# =========================================================================
# Copyright (C) 2025. The FuxiCTR Library. All rights reserved.
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
import torch.nn.functional as F
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.utils import not_in_whitelist
from torch.utils.tensorboard import SummaryWriter
from .target_attention import MultiHeadTargetAttention_baseline, MultiHeadTargetAttention_CollectiveKV


class SDIM(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="SDIM", 
                 gpu=-1, 
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_dim=64,
                 use_qkvo=True,
                 num_heads=1,
                 use_scale=True,
                 attention_dropout=0,
                 reuse_hash=True,
                 num_hashes=1,
                 hash_bits=4,
                 learning_rate=1e-3,
                 embedding_dim=10, 
                 net_dropout=0, 
                 batch_norm=False,
                 l2_norm=False,
                 short_seq_len=50,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(SDIM, self).__init__(feature_map,
                                   model_id=model_id, 
                                   gpu=gpu,
                                   embedding_regularizer=embedding_regularizer, 
                                   net_regularizer=net_regularizer,
                                   **kwargs)
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.reuse_hash = reuse_hash
        self.num_hashes = num_hashes
        self.hash_bits = hash_bits
        self.short_seq_len = short_seq_len
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim) 
        self.accumulation_steps = accumulation_steps
        self.l2_norm = l2_norm
        self.powers_of_two = nn.Parameter(torch.tensor([2.0 ** i for i in range(hash_bits)]), 
                                          requires_grad=False)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
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
                                                                loss_peak_weight=kwargs['loss_peak_weight'])
            self.writer = SummaryWriter(log_dir="./loss_logs/routing_loss")
        else:
            self.short_attention = MultiHeadTargetAttention_baseline(self.item_info_dim,
                                                        attention_dim,
                                                        num_heads,
                                                        attention_dropout)
        self.random_rotations = nn.Parameter(
            torch.randn(1, self.item_info_dim, self.num_hashes, self.hash_bits), 
            requires_grad=False
        )
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
        # short interest attention
        target_emb = item_feat_emb[:, -1, :]
        short_seq_emb = item_feat_emb[:, -self.short_seq_len:-1, :]
        short_mask = mask[:, -self.short_seq_len:-1] if mask.shape[1] >= self.short_seq_len else mask
        if self.method == 'CollectiveKV' and self.training:
            short_interest_emb, loss_peak, loss_balance = self.short_attention(target_emb, short_seq_emb, short_mask)
            return_dict['loss_peak'] = loss_peak
            return_dict['loss_balance'] = loss_balance
        else:
            short_interest_emb = self.short_attention(target_emb, short_seq_emb, short_mask)
        # long interest attention
        long_seq_emb = item_feat_emb[:, 0:-1, :]
        long_interest_emb = self.lsh_attentioin(self.random_rotations,
                                                target_emb, long_seq_emb, mask)
        emb_list += [target_emb, long_interest_emb, short_interest_emb]
        feature_emb = torch.cat(emb_list, dim=-1)
        y_pred = self.dnn(feature_emb)
        return_dict["y_pred"] = y_pred
        return return_dict

    def lsh_attentioin(self, random_rotations, target_item, history_sequence, mask):
        if self.reuse_hash:
            random_rotations = random_rotations.repeat(target_item.size(0), 1, 1, 1)
        else:
            random_rotations = torch.randn(
                target_item.size(0), target_item.size(1), self.num_hashes, 
                self.hash_bits, device=target_item.device
            )
        sequence_bucket = self.lsh_hash(history_sequence, random_rotations)
        target_bucket = (
            self.lsh_hash(target_item.unsqueeze(1), random_rotations)
            .repeat(1, sequence_bucket.shape[1], 1)
        )
        collide_mask = (
            # both hash collision and not masked
            ((sequence_bucket == target_bucket) * mask.unsqueeze(-1))
            .float()
            .permute(2, 0, 1) # num_hashes x B x seq_len
        )
        _, collide_index = torch.nonzero(collide_mask.flatten(start_dim=1), as_tuple=True)
        offsets = collide_mask.sum(dim=-1).flatten().cumsum(dim=0)
        offsets = torch.cat([torch.zeros(1, device=offsets.device), offsets]).long()
        attn_out = F.embedding_bag(collide_index, history_sequence.reshape(-1, target_item.size(1)), 
                                   offsets, mode='sum', include_last_offset=True) # (num_hashes x B) x d
        if self.l2_norm:
            attn_out = F.normalize(attn_out, dim=-1)
        attn_out = attn_out.view(self.num_hashes, -1, target_item.size(1)).mean(dim=0) # B x d
        return attn_out
        
    def lsh_hash(self, vecs, random_rotations):
        """ See the tensorflow-lsh-functions for reference:
            https://github.com/brc7/tensorflow-lsh-functions/blob/main/lsh_functions.py
            
            Input: vecs, with shape B x seq_len x d
            Output: hash_bucket, with shape B x seq_len x num_hashes
        """
        # shape B x seq_len x num_hashes x hash_bits x 1
        rotated_vecs = torch.einsum("bld,bdht->blht", vecs, random_rotations).unsqueeze(-1)
        rotated_vecs = torch.cat([-rotated_vecs, rotated_vecs], dim=-1)
        hash_code = torch.argmax(rotated_vecs, dim=-1).float()
        hash_bucket = torch.matmul(hash_code, self.powers_of_two.unsqueeze(-1)).squeeze(-1)
        return hash_bucket

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
        loss_main = self.compute_loss(return_dict, y_true)
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
