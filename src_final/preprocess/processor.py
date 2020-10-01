# coding=utf-8
"""
@author: Oscar
@license: (C) Copyright 2019-2022, ZJU.
@contact: 499616042@qq.com
@software: pycharm
@file: processor.py
@time: 2020/9/1 10:52
"""
import json
import random
import logging
from tqdm import tqdm
from transformers import BertTokenizer


logger = logging.getLogger(__name__)


__all__ = ['TriggerProcessor', 'RoleProcessor', 'AttributionProcessor', 'ROLE2_TO_ID',
           'fine_grade_tokenize', 'search_label_index', 'convert_examples_to_features']


ROLE2_TO_ID = {
    "O": 0,
    "B-time": 1,
    "I-time": 2,
    "E-time": 3,
    "S-time": 4,
    "B-loc": 5,
    "I-loc": 6,
    "E-loc": 7,
    "S-loc": 8,
    "X": 9
}


class BaseExample:
    def __init__(self,
                 set_type,
                 text,
                 label=None):
        self.set_type = set_type
        self.text = text
        self.label = label


class TriggerExample(BaseExample):
    def __init__(self,
                 set_type,
                 text,
                 distant_triggers=None,
                 label=None):
        super(TriggerExample, self).__init__(set_type=set_type,
                                             text=text,
                                             label=label)
        self.distant_triggers = distant_triggers


class RoleExample(BaseExample):
    def __init__(self,
                 set_type,
                 text,
                 trigger_location,
                 label=None):
        super(RoleExample, self).__init__(set_type=set_type,
                                          text=text,
                                          label=label)

        self.trigger_location = trigger_location  # trigger location in the text


class AttributionExample(BaseExample):
    def __init__(self,
                 set_type,
                 text,
                 trigger,
                 label=None):
        super(AttributionExample, self).__init__(set_type=set_type,
                                                 text=text,
                                                 label=label)
        self.trigger = trigger


class BaseFeature:
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 labels=None):
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.labels = labels


class TriggerFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 distant_trigger_label=None,
                 labels=None):
        super(TriggerFeature, self).__init__(token_ids=token_ids,
                                             attention_masks=attention_masks,
                                             token_type_ids=token_type_ids,
                                             labels=labels)
        self.distant_trigger_label = distant_trigger_label


class RoleFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 trigger_loc,
                 trigger_distance=None,
                 labels=None):
        """
        attribution detection use two handcrafted feature：
        1、trigger label： 1 for the tokens which are trigger, 0 for not;
        2、trigger distance: the relative distance of other tokens and the trigger tokens
        """
        super(RoleFeature, self).__init__(token_ids=token_ids,
                                          attention_masks=attention_masks,
                                          token_type_ids=token_type_ids,
                                          labels=labels)
        self.trigger_loc = trigger_loc
        self.trigger_distance = trigger_distance


class AttributionFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 trigger_loc,
                 pooling_masks,
                 labels=None):
        super(AttributionFeature, self).__init__(token_ids=token_ids,
                                                 attention_masks=attention_masks,
                                                 token_type_ids=token_type_ids,
                                                 labels=labels)
        self.trigger_loc = trigger_loc
        self.pooling_masks = pooling_masks


class BaseProcessor:
    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            examples = json.load(f)
        return examples


class TriggerProcessor(BaseProcessor):

    @staticmethod
    def _example_generator(raw_examples, set_type):
        examples = []
        callback_info = []

        for _ex in raw_examples:
            text = _ex['sentence']
            tmp_triggers = []

            for _event in _ex['events']:
                tmp_triggers.append((_event['trigger']['text'], int(_event['trigger']['offset'])))

            examples.append(TriggerExample(set_type=set_type,
                                           text=text,
                                           label=tmp_triggers,
                                           distant_triggers=_ex['distant_triggers']))

            callback_info.append((text, tmp_triggers, _ex['distant_triggers']))

        if set_type == 'dev':
            return examples, callback_info
        else:
            return examples

    def get_train_examples(self, raw_examples):
        return self._example_generator(raw_examples, 'train')

    def get_dev_examples(self, raw_examples):
        return self._example_generator(raw_examples, 'dev')


class RoleProcessor(BaseProcessor):

    @staticmethod
    def _example_generator(raw_examples, set_type):
        examples = []
        callback_info = []

        type_nums = 0
        type_weight = {'object': 0,
                       'subject': 0,
                       'time': 0,
                       'loc': 0}

        for _ex in raw_examples:
            text = _ex['sentence']

            for _event in _ex['events']:
                tmp_trigger = _event['trigger']

                # 加 1 是为了 CLS 偏置，保证 trigger loc 与真实的对应
                tmp_trigger_start = tmp_trigger['offset'] + 1
                tmp_trigger_end = tmp_trigger['offset'] + len(tmp_trigger['text'])

                examples.append(RoleExample(set_type=set_type,
                                            text=text,
                                            trigger_location=[tmp_trigger_start, tmp_trigger_end],
                                            label=_event['arguments']))

                gt_labels = {'object': [],
                             'subject': [],
                             'time': [],
                             'loc': []}
                for _role in _event['arguments']:
                    gt_labels[_role['role']].append((_role['text'], _role['offset']))

                    type_nums += 1
                    type_weight[_role['role']] += 1

                callback_info.append((text, tmp_trigger['text'], gt_labels))

        for key in type_weight.keys():
            type_weight[key] /= type_nums

        if set_type == 'dev':
            return examples, (callback_info, type_weight)
        else:
            return examples

    def get_train_examples(self, raw_examples):
        return self._example_generator(raw_examples, 'train')

    def get_dev_examples(self, raw_examples):
        return self._example_generator(raw_examples, 'dev')


class AttributionProcessor(BaseProcessor):

    @staticmethod
    def _example_generator(raw_examples, set_type):
        examples = []
        callback_info = []

        for _ex in raw_examples:
            text = _ex['sentence']

            for _event in _ex['events']:
                tmp_trigger = _event['trigger']

                label = [_event['tense'], _event['polarity']]

                examples.append(AttributionExample(set_type=set_type,
                                                   text=text,
                                                   trigger=(tmp_trigger['text'], tmp_trigger['offset']),
                                                   label=label))

                callback_info.append((text, tmp_trigger['text'], label))

        if set_type == 'dev':
            return examples, callback_info
        else:
            return examples

    def get_train_examples(self, raw_examples):
        return self._example_generator(raw_examples, 'train')

    def get_dev_examples(self, raw_examples):
        return self._example_generator(raw_examples, 'dev')


def search_label_index(tokens, label_tokens):
    """
    search label token indexes in all tokens
    :param tokens: tokens for raw text
    :param label_tokens: label which are split by the cjk extractor
    :return:
    """
    index_list = []  # 存放搜到的所有的index

    # 滑动窗口搜索 labels 在 token 中的位置
    for index in range(len(tokens) - len(label_tokens) + 1):
        if tokens[index: index + len(label_tokens)] == label_tokens:
            start_index = index
            end_index = start_index + len(label_tokens) - 1
            index_list.append((start_index, end_index))

    return index_list


def fine_grade_tokenize(raw_text, tokenizer):
    """
    序列标注任务 BERT 分词器可能会导致标注偏移，
    用 char-level 来 tokenize
    """
    tokens = []

    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[INV]')
            else:
                tokens.append(_ch)

    return tokens


def convert_trigger_example(ex_idx, example: TriggerExample, max_seq_len, tokenizer: BertTokenizer):
    """
    convert trigger examples to trigger features
    """
    set_type = example.set_type
    raw_text = example.text
    raw_label = example.label
    distant_triggers = example.distant_triggers

    tokens = fine_grade_tokenize(raw_text, tokenizer)

    labels = [[0] * 2 for i in range(len(tokens))]  # start / end

    distant_trigger_label = [0] * len(tokens)

    # tag labels
    for _label in raw_label:
        tmp_start = _label[1]
        tmp_end = _label[1] + len(_label[0]) - 1

        labels[tmp_start][0] = 1
        labels[tmp_end][1] = 1

    # tag distant triggers
    for _trigger in distant_triggers:
        tmp_trigger_tokens = fine_grade_tokenize(_trigger, tokenizer)
        tmp_index_list = search_label_index(tokens, tmp_trigger_tokens)

        assert len(tmp_index_list)

        for _index in tmp_index_list:
            for i in range(_index[0], _index[1] + 1):
                distant_trigger_label[i] = 1

    if len(labels) > max_seq_len - 2:
        labels = labels[:max_seq_len - 2]
        distant_trigger_label = distant_trigger_label[:max_seq_len - 2]

    pad_labels = [[0] * 2]
    labels = pad_labels + labels + pad_labels

    distant_trigger_label = [0] + distant_trigger_label + [0]

    if len(labels) < max_seq_len:
        pad_length = max_seq_len - len(labels)

        labels = labels + pad_labels * pad_length
        distant_trigger_label = distant_trigger_label + [0] * pad_length

    assert len(labels) == max_seq_len
    assert len(distant_trigger_label) == max_seq_len

    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)

    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    if ex_idx < 3 and set_type == 'train':
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        logger.info(f'text: {" ".join(tokens)}')
        logger.info(f"token_ids: {token_ids}")
        logger.info(f"attention_masks: {attention_masks}")
        logger.info(f"token_type_ids: {token_type_ids}")
        logger.info(f'distant trigger: {distant_trigger_label}')

    feature = TriggerFeature(token_ids=token_ids,
                             attention_masks=attention_masks,
                             token_type_ids=token_type_ids,
                             distant_trigger_label=distant_trigger_label,
                             labels=labels)

    return feature


def convert_role1_example(ex_idx, example: RoleExample, max_seq_len, tokenizer: BertTokenizer):
    """
    convert role examples to sub & obj features
    """
    set_type = example.set_type
    raw_text = example.text
    raw_label = example.label
    trigger_loc = example.trigger_location

    if set_type == 'train' and trigger_loc[0] > max_seq_len:
        logger.info('Skip this example where the tag is longer than max sequence length')
        return None

    tokens = fine_grade_tokenize(raw_text, tokenizer)

    labels = [[0] * 4 for i in range(len(tokens))]  # sub / obj

    # sub / obj 为空时丢弃该 example
    sub_flag = False
    obj_flag = False

    # tag labels
    for role in raw_label:
        if role['role'] in ['time', 'loc']:
            continue

        if role['role'] == 'subject':
            sub_flag = True
        elif role['role'] == 'object':
            obj_flag = True

        if role['role'] == 'object':
            role_type_idx = 0
        else:
            role_type_idx = 2

        role_start = role['offset']
        role_end = role_start + len(role['text']) - 1

        labels[role_start][role_type_idx] = 1  # start 位置标注为 1
        labels[role_end][role_type_idx + 1] = 1  # end 位置标注为 1

    if set_type == 'train' and (not sub_flag or not obj_flag):
        return None

    if len(labels) > max_seq_len - 2:
        labels = labels[:max_seq_len - 2]

    pad_labels = [[0] * 4]
    labels = pad_labels + labels + pad_labels

    if len(labels) < max_seq_len:
        pad_length = max_seq_len - len(labels)
        labels = labels + pad_labels * pad_length

    # build trigger distance features
    trigger_distance = [511] * max_seq_len
    for i in range(max_seq_len):
        if trigger_loc[0] <= i <= trigger_loc[1]:
            trigger_distance[i] = 0
            continue
        elif i < trigger_loc[0]:
            trigger_distance[i] = trigger_loc[0] - i
        else:
            trigger_distance[i] = i - trigger_loc[1]

    assert len(labels) == max_seq_len

    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)

    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    for i in range(trigger_loc[0], trigger_loc[1] + 1):
        token_type_ids[i] = 1

    if ex_idx < 3 and set_type == 'train':
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        logger.info(f'text: {" ".join(tokens)}')
        logger.info(f"token_ids: {token_ids}")
        logger.info(f"attention_masks: {attention_masks}")
        logger.info(f"token_type_ids: {token_type_ids}")
        logger.info(f'trigger location: {trigger_loc}')

    feature = RoleFeature(token_ids=token_ids,
                          attention_masks=attention_masks,
                          token_type_ids=token_type_ids,
                          trigger_loc=trigger_loc,
                          trigger_distance=trigger_distance,
                          labels=labels)

    return feature


def convert_role2_example(ex_idx, example: RoleExample, max_seq_len, tokenizer: BertTokenizer):
    """
    convert role examples to time & loc features
    """
    set_type = example.set_type
    raw_text = example.text
    raw_label = example.label
    trigger_loc = example.trigger_location

    if trigger_loc[0] > max_seq_len:
        logger.info('Skip this example where the tag is longer than max sequence length')
        return None

    tokens = fine_grade_tokenize(raw_text, tokenizer)

    labels = [0] * len(tokens)  # time / loc

    flag = False

    # tag labels
    for role in raw_label:
        role_type = role['role']

        if role_type in ['subject', 'object']:
            continue

        flag = True

        role_start = role['offset']
        role_end = role_start + len(role['text']) - 1

        if role_start == role_end:
            labels[role_start] = ROLE2_TO_ID['S-' + role_type]
        else:
            labels[role_start] = ROLE2_TO_ID['B-' + role_type]
            labels[role_end] = ROLE2_TO_ID['E-' + role_type]
            for i in range(role_start + 1, role_end):
                labels[i] = ROLE2_TO_ID['I-' + role_type]

    # 负样本以一定概率剔除，保证类别均衡，增大后续召回率
    if set_type == 'train' and not flag and random.random() > 0.3:
        return None

    if len(labels) > max_seq_len - 2:
        labels = labels[:max_seq_len - 2]

    pad_labels = [ROLE2_TO_ID['O']]
    labels = [ROLE2_TO_ID['X']] + labels + [ROLE2_TO_ID['X']]

    if len(labels) < max_seq_len:
        pad_length = max_seq_len - len(labels)
        labels = labels + pad_labels * pad_length

    # build trigger distance features
    trigger_distance = [511] * max_seq_len
    for i in range(max_seq_len):
        if trigger_loc[0] <= i <= trigger_loc[1]:
            trigger_distance[i] = 0
            continue
        elif i < trigger_loc[0]:
            trigger_distance[i] = trigger_loc[0] - i
        else:
            trigger_distance[i] = i - trigger_loc[1]

    assert len(labels) == max_seq_len

    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)

    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    for i in range(trigger_loc[0], trigger_loc[1] + 1):
        token_type_ids[i] = 1

    if ex_idx < 3 and set_type == 'train':
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        logger.info(f'text: {" ".join(tokens)}')
        logger.info(f'trigger location: {trigger_loc}')
        logger.info(f'labels: {labels}')

    feature = RoleFeature(token_ids=token_ids,
                          attention_masks=attention_masks,
                          token_type_ids=token_type_ids,
                          trigger_loc=trigger_loc,
                          trigger_distance=trigger_distance,
                          labels=labels)

    return feature


def convert_attribution_example(ex_idx, example: AttributionExample, max_seq_len,
                                tokenizer: BertTokenizer, polarity2id, tense2id):
    """
    convert attribution example to attribution feature
    """
    set_type = example.set_type
    raw_text = example.text
    raw_label = example.label
    trigger = example.trigger

    tokens = fine_grade_tokenize(raw_text, tokenizer)

    trigger_loc = (trigger[1] + 1, trigger[1] + len(trigger[0]))

    labels = [tense2id[raw_label[0]], polarity2id[raw_label[1]]]

    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)

    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    window_size = 20

    # 左右各取 20 的窗口作为 trigger 触发的语境
    pooling_masks_range = range(max(1, trigger_loc[0] - window_size),
                                min(min(1 + len(raw_text), max_seq_len - 1), trigger_loc[1] + window_size))

    pooling_masks = [0] * max_seq_len
    for i in pooling_masks_range:
        pooling_masks[i] = 1
    for i in range(trigger_loc[0], trigger_loc[1] + 1):
        pooling_masks[i] = 0

    if ex_idx < 3 and set_type == 'train':
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        logger.info(f'text: {" ".join(tokens)}')
        logger.info(f"token_ids: {token_ids}")
        logger.info(f"attention_masks: {attention_masks}")
        logger.info(f"token_type_ids: {token_type_ids}")
        logger.info(f'trigger loc: {trigger_loc}')
        logger.info(f'labels: {labels}')

    feature = AttributionFeature(token_ids=token_ids,
                                 attention_masks=attention_masks,
                                 token_type_ids=token_type_ids,
                                 trigger_loc=trigger_loc,
                                 pooling_masks=pooling_masks,
                                 labels=labels)

    return feature


def convert_examples_to_features(task_type, examples, bert_dir, max_seq_len, **kwargs):
    assert task_type in ['trigger', 'role1', 'role2', 'attribution']

    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    logger.info(f'Vocab nums in this tokenizer is: {tokenizer.vocab_size}')

    features = []

    for i, example in enumerate(tqdm(examples, desc=f'convert examples')):
        if task_type == 'trigger':

            feature = convert_trigger_example(
                ex_idx=i,
                example=example,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer,
            )

        elif task_type == 'role1':
            feature = convert_role1_example(
                ex_idx=i,
                example=example,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer
            )

        elif task_type == 'role2':
            feature = convert_role2_example(
                ex_idx=i,
                example=example,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer
            )

        else:
            feature = convert_attribution_example(
                ex_idx=i,
                example=example,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer,
                polarity2id=kwargs.get('polarity2id'),
                tense2id=kwargs.get('tense2id')
            )

        if feature is None:
            continue

        features.append(feature)

    return features
