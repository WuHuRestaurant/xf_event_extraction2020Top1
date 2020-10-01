# coding=utf-8
"""
@author: Oscar
@license: (C) Copyright 2019-2022, ZJU.
@contact: 499616042@qq.com
@software: pycharm
@file: model_utils.py
@time: 2020/9/1 16:19
"""
import os
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel


class ConditionalLayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape,
                 eps=1e-12):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(normalized_shape))

        self.weight_dense = nn.Linear(normalized_shape * 2, normalized_shape, bias=False)
        self.bias_dense = nn.Linear(normalized_shape * 2, normalized_shape, bias=False)

        self.reset_weight_and_bias()

    def reset_weight_and_bias(self):
        """
        此处初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
        """
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

        nn.init.zeros_(self.weight_dense.weight)
        nn.init.zeros_(self.bias_dense.weight)

    def forward(self, inputs, cond=None):
        assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'
        cond = torch.unsqueeze(cond, 1)  # (b, 1, h*2)

        weight = self.weight_dense(cond) + self.weight  # (b, 1, h)
        bias = self.bias_dense(cond) + self.bias  # (b, 1, h)

        mean = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
        outputs = inputs - mean  # (b, s, h)

        variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)  # (b, s, 1)

        outputs = outputs / std  # (b, s, h)

        outputs = outputs * weight + bias

        return outputs


class BaseModel(nn.Module):
    def __init__(self,
                 bert_dir,
                 dropout_prob=0.1):
        super(BaseModel, self).__init__()
        config_path = os.path.join(bert_dir, 'config.json')

        assert os.path.exists(bert_dir) and os.path.exists(config_path), \
            'pretrained bert file does not exist'

        self.bert_module = BertModel.from_pretrained(bert_dir)

        self.bert_config = self.bert_module.config

        self.dropout_layer = nn.Dropout(dropout_prob)

    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.zeros_(module.bias)
                    nn.init.ones_(module.weight)

    @staticmethod
    def _batch_gather(data: torch.Tensor, index: torch.Tensor):
        """
        实现类似 tf.batch_gather 的效果
        :param data: (bs, max_seq_len, hidden)
        :param index: (bs, n)
        :return: a tensor which shape is (bs, n, hidden)
        """
        index = index.unsqueeze(-1).repeat_interleave(data.size()[-1], dim=-1)  # (bs, n, hidden)
        return torch.gather(data, 1, index)


# trigger 提取器
class TriggerExtractor(BaseModel):
    def __init__(self,
                 bert_dir,
                 dropout_prob=0.1,
                 use_distant_trigger=False,
                 **kwargs):
        super(TriggerExtractor, self).__init__(bert_dir=bert_dir,
                                               dropout_prob=dropout_prob)

        self.use_distant_trigger = use_distant_trigger

        out_dims = self.bert_config.hidden_size

        if use_distant_trigger:
            embedding_dim = kwargs.pop('embedding_dims', 256)
            self.distant_trigger_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
            out_dims += embedding_dim

        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)

        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        self.classifier = nn.Linear(mid_linear_dims, 2)

        self.activation = nn.Sigmoid()

        self.criterion = nn.BCELoss()

        init_blocks = [self.mid_linear, self.classifier]

        if use_distant_trigger:
            init_blocks += [self.distant_trigger_embedding]

        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                distant_trigger=None,
                labels=None):

        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        seq_out = bert_outputs[0]

        if self.use_distant_trigger:
            assert distant_trigger is not None, \
                'When using distant trigger features, distant trigger should be implemented'

            distant_trigger_feature = self.distant_trigger_embedding(distant_trigger)
            seq_out = torch.cat([seq_out, distant_trigger_feature], dim=-1)

        seq_out = self.mid_linear(seq_out)

        logits = self.activation(self.classifier(seq_out))

        out = (logits,)

        if labels is not None:
            loss = self.criterion(logits, labels.float())
            out = (loss,) + out

        return out


# sub & obj 提取器
class Role1Extractor(BaseModel):
    def __init__(self,
                 bert_dir,
                 dropout_prob=0.1,
                 use_trigger_distance=False,
                 **kwargs):
        super(Role1Extractor, self).__init__(bert_dir=bert_dir,
                                             dropout_prob=dropout_prob)

        out_dims = self.bert_config.hidden_size

        self.use_trigger_distance = use_trigger_distance

        self.conditional_layer_norm = ConditionalLayerNorm(out_dims, eps=self.bert_config.layer_norm_eps)

        if use_trigger_distance:
            embedding_dim = kwargs.pop('embedding_dims', 256)

            self.trigger_distance_embedding = nn.Embedding(num_embeddings=512, embedding_dim=embedding_dim)

            out_dims += embedding_dim

            self.layer_norm = nn.LayerNorm(out_dims, eps=self.bert_config.layer_norm_eps)

        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)

        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        self.obj_classifier = nn.Linear(mid_linear_dims, 2)
        self.sub_classifier = nn.Linear(mid_linear_dims, 2)

        self.activation = nn.Sigmoid()

        self.criterion = nn.BCELoss()

        init_blocks = [self.mid_linear, self.obj_classifier, self.sub_classifier]

        if use_trigger_distance:
            init_blocks += [self.trigger_distance_embedding, self.layer_norm]

        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                trigger_index,
                trigger_distance=None,
                labels=None):

        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        seq_out, pooled_out = bert_outputs[0], bert_outputs[1]

        trigger_label_feature = self._batch_gather(seq_out, trigger_index)

        trigger_label_feature = trigger_label_feature.view([trigger_label_feature.size()[0], -1])

        seq_out = self.conditional_layer_norm(seq_out, trigger_label_feature)

        if self.use_trigger_distance:
            assert trigger_distance is not None, \
                'When using trigger distance features, trigger distance should be implemented'

            trigger_distance_feature = self.trigger_distance_embedding(trigger_distance)
            seq_out = torch.cat([seq_out, trigger_distance_feature], dim=-1)
            seq_out = self.layer_norm(seq_out)
            # seq_out = self.dropout_layer(seq_out)

        seq_out = self.mid_linear(seq_out)

        obj_logits = self.activation(self.obj_classifier(seq_out))
        sub_logits = self.activation(self.sub_classifier(seq_out))

        logits = torch.cat([obj_logits, sub_logits], dim=-1)
        out = (logits,)

        if labels is not None:
            masks = torch.unsqueeze(attention_masks, -1)

            labels = labels.float()
            obj_loss = self.criterion(obj_logits * masks, labels[:, :, :2])
            sub_loss = self.criterion(sub_logits * masks, labels[:, :, 2:])

            loss = obj_loss + sub_loss

            out = (loss,) + out

        return out


# time & loc 提取器
class Role2Extractor(BaseModel):
    def __init__(self,
                 bert_dir,
                 dropout_prob=0.1,
                 use_trigger_distance=False,
                 **kwargs):
        super(Role2Extractor, self).__init__(bert_dir=bert_dir,
                                             dropout_prob=dropout_prob)

        out_dims = self.bert_config.hidden_size

        self.use_trigger_distance = use_trigger_distance

        self.conditional_layer_norm = ConditionalLayerNorm(out_dims, eps=self.bert_config.layer_norm_eps)

        if use_trigger_distance:
            embedding_dim = kwargs.pop('embedding_dims', 256)

            self.trigger_distance_embedding = nn.Embedding(num_embeddings=512, embedding_dim=embedding_dim)

            out_dims += embedding_dim

            self.layer_norm = nn.LayerNorm(out_dims, eps=self.bert_config.layer_norm_eps)

        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)

        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        self.classifier = nn.Linear(mid_linear_dims, 10)

        self.crf_module = CRF(num_tags=10, batch_first=True)

        init_blocks = [self.mid_linear, self.classifier]

        if use_trigger_distance:
            init_blocks += [self.trigger_distance_embedding, self.layer_norm]

        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                trigger_index,
                trigger_distance=None,
                labels=None):

        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        seq_out, pooled_out = bert_outputs[0], bert_outputs[1]

        trigger_label_feature = self._batch_gather(seq_out, trigger_index)

        trigger_label_feature = trigger_label_feature.view([trigger_label_feature.size()[0], -1])

        seq_out = self.conditional_layer_norm(seq_out, trigger_label_feature)

        if self.use_trigger_distance:
            assert trigger_distance is not None, \
                'When using distant trigger features, trigger distance should be implemented'

            trigger_distance_feature = self.trigger_distance_embedding(trigger_distance)

            seq_out = torch.cat([seq_out, trigger_distance_feature], dim=-1)
            seq_out = self.layer_norm(seq_out)

        seq_out = self.mid_linear(seq_out)

        emissions = self.classifier(seq_out)

        if labels is not None:
            tokens_loss = -1. * self.crf_module(emissions=emissions,
                                                tags=labels.long(),
                                                mask=attention_masks.byte(),
                                                reduction='mean')

            out = (tokens_loss,)

        else:
            tokens_out = self.crf_module.decode(emissions=emissions, mask=attention_masks.byte())
            out = (tokens_out,)

        return out


# tense & polarity 分类器
class AttributionClassifier(BaseModel):
    def __init__(self,
                 bert_dir,
                 dropout_prob=0.1):
        super(AttributionClassifier, self).__init__(bert_dir=bert_dir,
                                                    dropout_prob=dropout_prob)

        out_dims = self.bert_config.hidden_size

        self.pooling_layer = nn.AdaptiveMaxPool1d(output_size=1)

        self.tense_classifier = nn.Linear(out_dims * 3, 4)
        self.polarity_classifier = nn.Linear(out_dims * 3, 3)

        self.criterion = nn.CrossEntropyLoss()

        init_blocks = [self.tense_classifier, self.polarity_classifier]

        self._init_weights(init_blocks)

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                trigger_index,
                pooling_masks,
                labels=None):
        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        seq_out = bert_outputs[0]

        trigger_label_feature = self._batch_gather(seq_out, trigger_index)

        trigger_label_feature = trigger_label_feature.view([trigger_label_feature.size()[0], -1])

        seq_out = torch.transpose(seq_out, -1, -2)  # (bs, hidden, seq_len)
        pooling_masks = torch.unsqueeze(pooling_masks, 1)
        seq_out = seq_out + (1 - pooling_masks) * (-1e7)  # mask 无关区域

        pooled_out = self.pooling_layer(seq_out).squeeze(-1)

        logits = torch.cat([pooled_out, trigger_label_feature], dim=-1)

        polarity_logits = self.polarity_classifier(self.dropout_layer(logits))
        tense_logits = self.tense_classifier(self.dropout_layer(logits))

        out = (torch.softmax(polarity_logits, dim=-1), torch.softmax(tense_logits, dim=-1),)

        if labels is not None:
            labels = labels.long()

            tense_loss = self.criterion(tense_logits, labels[:, 0])
            polarity_loss = self.criterion(polarity_logits, labels[:, 1])

            loss = polarity_loss + tense_loss

            out = (loss,) + out

        return out


def build_model(task_type, bert_dir, **kwargs):
    assert task_type in ['trigger', 'role1', 'role2', 'attribution'], 'task mismatch'

    if task_type == 'trigger':
        model = TriggerExtractor(bert_dir=bert_dir,
                                 dropout_prob=kwargs.pop('dropout_prob', 0.1),
                                 use_distant_trigger=kwargs.pop('use_distant_trigger'))

    elif task_type == 'role1':
        model = Role1Extractor(bert_dir=bert_dir,
                               dropout_prob=kwargs.pop('dropout_prob', 0.1),
                               use_trigger_distance=kwargs.pop('use_trigger_distance'))
    elif task_type == 'role2':
        model = Role2Extractor(bert_dir=bert_dir,
                               dropout_prob=kwargs.pop('dropout_prob', 0.1),
                               use_trigger_distance=kwargs.pop('use_trigger_distance'))

    else:
        model = AttributionClassifier(bert_dir=bert_dir,
                                      dropout_prob=kwargs.pop('dropout_prob', 0.1))

    return model
