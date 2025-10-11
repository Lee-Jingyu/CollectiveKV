import torch
import torch.nn as nn
from fuxictr.pytorch.models import BaseModel
from tqdm import tqdm
import sys
import logging
import numpy as np


class ArkBase(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="ArkBase", 
                 task="binary_classification", 
                 gpu=-1, 
                 monitor="AUC", 
                 save_best_only=True, 
                 monitor_mode="max", 
                 early_stop_patience=2, 
                 eval_steps=None, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 reduce_lr_on_plateau=True, 
                 **kwargs):
        super(ArkBase, self).__init__(feature_map, 
                                      model_id=model_id, 
                                      task=task, 
                                      gpu=gpu, 
                                      monitor=monitor, 
                                      save_best_only=save_best_only, 
                                      monitor_mode=monitor_mode, 
                                      early_stop_patience=early_stop_patience, 
                                      eval_steps=eval_steps, 
                                      embedding_regularizer=embedding_regularizer, 
                                      net_regularizer=net_regularizer, 
                                      reduce_lr_on_plateau=reduce_lr_on_plateau, 
                                      **kwargs)

    def compute_loss(self, return_dict):
        preds, labels, valid_len = return_dict["preds"], return_dict["labels"], return_dict["valid_len"]
        padding_mask = ~self.sequence_mask(valid_len, max_len=preds.shape[1]) # get False as paddings
        y_pred = preds.masked_select(padding_mask)
        y_true = labels.masked_select(padding_mask).float()
        loss = self.loss_fn(y_pred, y_true, reduction='mean')
        loss += self.regularization_loss()
        return loss
    
    def get_inputs(self, inputs):
        feature_h_seq = inputs["feature_h_seq"].to(self.device)
        label_seq = inputs["label_seq"].to(self.device)
        seq_len = inputs[0]['seq_len'].to(self.device)
        return feature_h_seq, label_seq, seq_len
    
    def sequence_mask(self, lens, max_len=None):
        if max_len is None:
            max_len = lens.max()
        # 1's for masked positions
        mask = torch.arange(max_len, device=lens.device)[None, :] >= lens[:, None]
        mask = mask.flip(dims=[1]) # padding on left
        return mask

    def train_step(self, batch_data):
        # import time; t1 = time.time()
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        loss = self.compute_loss(return_dict)
        if return_dict.get('orth_losses') is not None:
            orth_loss = torch.mean(torch.stack(return_dict['orth_losses']))
            loss += orth_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        # print(time.time() - t1)
        return loss

    def evaluate(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            group_id = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                preds, labels = return_dict["preds"], return_dict["labels"]
                test_len = batch_data["test_len"].to(self.device)
                test_mask = self.sequence_mask(test_len, max_len=preds.shape[1])
                preds = preds.masked_select(~test_mask)
                labels = labels.masked_select(~test_mask)
                y_pred.extend(preds.data.cpu().numpy().reshape(-1))
                y_true.extend(labels.data.cpu().numpy().reshape(-1))
                if self.feature_map.group_id is not None:
                    group_batch = self.get_group_id(batch_data).unsqueeze(1).repeat(1, test_mask.shape[1])
                    group_id.extend(group_batch.masked_select(~test_mask.cpu()).numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            group_id = np.array(group_id) if len(group_id) > 0 else None
            if metrics is not None:
                val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
            else:
                val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            return val_logs
        

class PositionEncoding(nn.Module):
    def __init__(
        self,
        embedding_dim,
        dropout=0.0,
        max_len=5000
    ):
        super(PositionEncoding, self).__init__()
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_emb):
        batch_size, seq_len, _ = seq_emb.shape
        positions = torch.arange(seq_len).repeat(batch_size, 1).to(seq_emb.device)
        return self.dropout(seq_emb + self.position_embedding(positions))
