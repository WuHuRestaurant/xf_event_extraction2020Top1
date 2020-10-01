# coding=utf-8
"""
@author: Oscar
@license: (C) Copyright 2019-2022, ZJU.
@contact: 499616042@qq.com
@software: pycharm
@file: convert_raw_data.py
@time: 2020/8/31 21:42
"""
import os
import copy
import json
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold


def load_examples(file_path):
    with open(file_path, encoding='utf-8') as f:
        examples = json.load(f)
    return examples


def save_info(data_dir, data, desc):
    with open(os.path.join(data_dir, f'{desc}.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_pair(before, role, nexts, pair=("《", "》")):
    pre = ""
    sub = ""
    if pair[0] in before or pair[1] in nexts:
        if pair[0] in before and pair[1] not in before[before.rfind(pair[0]):]:
            pre = before[before.rfind(pair[0]):]
        if pair[1] in nexts and pair[0] not in nexts[:nexts.find(pair[1]) + 1]:
            sub = nexts[:nexts.find(pair[1]) + 1]
    if "《" in pair:
        if (not (len(pre) > 1 and len(sub) > 1)) and (not len(pre) + len(sub) == 0):
            return pre, sub
        else:
            return "", ""
    if "”" in pair or "(" in pair or "'" in pair or "（" in pair:
        if (not (len(pre) > 1 and len(sub) > 1)) and (not len(pre) + len(sub) == 0) and (
                len(pre) == 0 or len(sub) == 0):
            new_role = pre + role + sub
            if pair[0] in new_role and pair[1] in new_role:
                return pre, sub
            else:
                return "", ""
        else:
            return "", ""


def clean_data(raw_pre_examples):
    nums = 0
    for idx, example in enumerate(raw_pre_examples):
        text = example['sentence']
        for jdx, event in enumerate(example['events']):
            for kdx, argument in enumerate(event['arguments']):
                if argument['role'] == 'subject' or argument['role'] == 'object':
                    for pair in [("'", "'"), ("“", "”"), ("(", ")"), ("《", "》"), ("（", "）")]:
                        before = text[max(argument['offset'] - 40, 0):argument['offset']]
                        nexts = text[argument['offset'] + argument['length']:min(
                            argument['offset'] + argument['length'] + 40, len(text))]
                        pre, sub = find_pair(before, argument['text'], nexts, pair)
                        if pre != "" or sub != "":
                            new_text = pre + argument['text'] + sub
                            new_offset = argument['offset'] - len(pre)
                            new_length = len(new_text)
                            assert pair[0] in new_text and pair[1] in new_text
                            assert text[new_offset:new_offset + new_length] == new_text
                            argument['text'] = new_text
                            argument['offset'] = new_offset
                            argument['length'] = new_length
                            nums += 1
    return raw_pre_examples, nums


def convert_raw_data(data_dir, save_data=False, save_dict=False):
    """
    1、10 折交叉验证构造带标签数据的 distant trigger （长度大于一，并且出现次数大于一）
    2、将复赛数据 8 : 2 划分训练 / 验证数据；
    3、构建先验知识词典 ---- (tense_prob, polarity_prob)；
    4、构造 trigger 词典 （所有长度等于2的 trigger，去重后构造一个 trigger dict）
    """
    random.seed(321)

    raw_dir = os.path.join(data_dir, 'raw_data')
    mid_dir = os.path.join(data_dir, 'mid_data')
    if not os.path.exists(mid_dir):
        os.mkdir(mid_dir)

    stack_examples = load_examples(os.path.join(raw_dir, 'raw_stack.json'))
    stack_examples, nums = clean_data(stack_examples)
    print(f'Clean {nums} final data')

    test_examples = load_examples(os.path.join(raw_dir, 'sentences.json'))
    preliminary_examples = load_examples(os.path.join(raw_dir, 'preliminary_pred_triggers_pred_roles.json'))
    preliminary_examples, nums = clean_data(preliminary_examples)
    print(f'Clean {nums} preliminary data')

    kf = KFold(10)

    triggers = {}

    nums = 0

    # 10折交叉构造 distant trigger
    for _now_id, _candidate_id in kf.split(stack_examples):
        now = [stack_examples[_id] for _id in _now_id]
        candidate = [stack_examples[_id] for _id in _candidate_id]

        now_triggers = {}

        for _ex in now:
            for _event in _ex['events']:
                tmp_trigger = _event['trigger']['text']
                # distant trigger 选取长度为2的
                if len(tmp_trigger) != 2:
                    continue
                if tmp_trigger in now_triggers:
                    now_triggers[tmp_trigger] += 1
                else:
                    now_triggers[tmp_trigger] = 1

        for _ex in candidate:
            tmp_sent = _ex['sentence']
            candidate_triggers = []
            for _t in now_triggers.keys():
                if _t in tmp_sent:
                    candidate_triggers.append(_t)

            for _event in _ex['events']:
                tmp_trigger = _event['trigger']['text']
                # distant trigger 选取长度为2的
                if len(tmp_trigger) != 2:
                    continue
                if tmp_trigger in triggers:
                    triggers[tmp_trigger] += 1
                else:
                    triggers[tmp_trigger] = 1

            _ex['distant_triggers'] = candidate_triggers

            if len(candidate_triggers) > nums:
                nums = len(candidate_triggers)

    print(nums)
    nums = 0

    for _ex in preliminary_examples:
        tmp_sent = _ex['sentence']
        candidate_triggers = []
        for _t in triggers.keys():
            if _t in tmp_sent:
                candidate_triggers.append(_t)

        _ex['distant_triggers'] = candidate_triggers[:10]
        if len(_ex['distant_triggers']) > nums:
            nums = len(_ex['distant_triggers'])

    print(nums)
    nums = 0

    # 构造 test 的 distant trigger
    for _ex in test_examples:
        tmp_sent = _ex['sentence']
        candidate_triggers = []
        for _t in triggers.keys():
            if _t in tmp_sent:
                candidate_triggers.append(_t)

        _ex['distant_triggers'] = candidate_triggers
        if len(candidate_triggers) > nums:
            nums = len(candidate_triggers)

    triggers = dict(sorted(triggers.items(), key=lambda x: x[1], reverse=True))

    tense = {}
    polarity = {}
    counts = 0.

    print(nums)

    for _ex in tqdm(stack_examples, desc='raw data convert'):
        _ex.pop('words')

        for _event in _ex['events']:
            tmp_tense = _event['tense']
            tmp_polarity = _event['polarity']
            counts += 1

            if tmp_tense not in tense:
                tense[tmp_tense] = 1
            else:
                tense[tmp_tense] += 1

            if tmp_polarity not in polarity:
                polarity[tmp_polarity] = 1
            else:
                polarity[tmp_polarity] += 1

    def build_map(info):
        info = {key: info[key] / counts for key in info.keys()}
        info2id = {'map': {}, 'prob': []}
        for idx, key in enumerate(info.keys()):
            info2id['map'][key] = idx
            info2id['prob'].append(info[key])
        return info2id

    tense2id = build_map(tense)
    polarity2id = build_map(polarity)
    triggers_dict = {key: idx + 1 for idx, key in enumerate(triggers.keys())}

    train, dev = train_test_split(stack_examples, shuffle=True, random_state=456, test_size=0.2)

    print(len(train), len(dev))

    if save_data:
        print('Save data')
        save_info(raw_dir, stack_examples, 'stack')
        save_info(raw_dir, train, 'train')
        save_info(raw_dir, dev, 'dev')
        save_info(raw_dir, test_examples, 'test')
        save_info(raw_dir, preliminary_examples, 'preliminary_stack')
    else:
        print('Do not save data')

    if save_dict:
        save_info(mid_dir, tense2id, 'tense2id')
        save_info(mid_dir, polarity2id, 'polarity2id')
        save_info(mid_dir, triggers_dict, 'triggers_dict')


def split_preliminary_trigger_data(base_dir, save_dir):
    """
    初赛的数据与复赛数据不一致，进行处理
    trigger 的数据种类大致划分为三种
    first： 只有一个 trigger + 预测 trigger 正确的
    second：只有一个 trigger + 预测 trigger 错误的
    third： 多个 trigger 的
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    preliminary_examples = load_examples(os.path.join(base_dir, 'preliminary_stack.json'))
    first_examples = []
    second_examples = []
    third_examples = []

    for _ex in tqdm(preliminary_examples, desc='process preliminary examples'):
        assert len(_ex['pred_triggers']) in [1, 0]

        if len(_ex['events']) > 1:
            third_examples.append(_ex)
        else:
            if not len(_ex['pred_triggers']):
                first_examples.append(_ex)
                continue

            gt_trigger = _ex['events'][0]['trigger']
            pred_trigger = _ex['pred_triggers'][0]
            if gt_trigger['text'] == pred_trigger['text'] and gt_trigger['offset'] == pred_trigger['offset']:
                first_examples.append(_ex)
            else:
                second_examples.append(_ex)

    print(f'First trigger examples nums: {len(first_examples)}')
    print(f'Second trigger examples nums: {len(second_examples)}')
    print(f'Third trigger examples nums: {len(third_examples)}')

    save_info(save_dir, first_examples, 'trigger_first')
    save_info(save_dir, second_examples, 'trigger_second')
    save_info(save_dir, third_examples, 'trigger_third')


def split_preliminary_trigger_third_data(bert_dir, base_dir, ckpt_path):
    """
    划分策略，不增强的模型或者用 first 增强的模型对 third 进行预测，
    关闭 force_one_trigger
    model 的参数直接在函数中手动设置
    """
    import sys
    import torch
    sys.path.append('../../')
    from transformers import BertTokenizer
    from src_final.utils.model_utils import TriggerExtractor
    from src_final.utils.evaluator import pointer_trigger_decode
    from src_final.preprocess.processor import fine_grade_tokenize, search_label_index

    tokenizer = BertTokenizer.from_pretrained(bert_dir)

    print('Load model ckpt')
    model = TriggerExtractor(bert_dir, use_distant_trigger=True)
    model.load_state_dict(torch.load(os.path.join(ckpt_path, 'model.pt'),
                                     map_location=torch.device('cpu')))

    device = torch.device('cuda:1')

    model.to(device)

    trigger_third = load_examples(os.path.join(base_dir, 'trigger_third.json'))

    print(f'raw third example nums: {len(trigger_third)}')

    trigger_third_new = []

    num_1, num_2 = 0, 0

    model.eval()

    with torch.no_grad():
        for _ex in tqdm(trigger_third, desc='process trigger third'):
            text = _ex['sentence']
            distant_triggers = _ex['distant_triggers']

            text_tokens = fine_grade_tokenize(text, tokenizer)

            assert len(text_tokens) == len(text)

            trigger_encode_dict = tokenizer.encode_plus(text=text_tokens,
                                                        max_length=512,
                                                        pad_to_max_length=False,
                                                        is_pretokenized=True,
                                                        return_token_type_ids=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')

            model_inputs = {'token_ids': trigger_encode_dict['input_ids'],
                            'attention_masks': trigger_encode_dict['attention_mask'],
                            'token_type_ids': trigger_encode_dict['token_type_ids']}

            distant_trigger_label = [0] * len(text)
            for _trigger in distant_triggers:
                tmp_trigger_tokens = fine_grade_tokenize(_trigger, tokenizer)
                tmp_index_list = search_label_index(text_tokens, tmp_trigger_tokens)

                assert len(tmp_index_list)

                for _index in tmp_index_list:
                    for i in range(_index[0], _index[1] + 1):
                        # 采用 1 / 0
                        distant_trigger_label[i] = 1
            if len(distant_trigger_label) > 510:
                distant_trigger_label = distant_trigger_label[:510]
            distant_trigger_label = [0] + distant_trigger_label + [0]

            distant_trigger_label = torch.tensor([distant_trigger_label]).long()
            model_inputs['distant_trigger'] = distant_trigger_label

            for key in model_inputs.keys():
                model_inputs[key] = model_inputs[key].to(device)

            trigger_pred_logits = model(**model_inputs)[0][0]
            trigger_pred_logits = trigger_pred_logits.cpu().numpy()[1:1+len(text)]

            trigger_pred = pointer_trigger_decode(trigger_pred_logits, text, distant_triggers,
                                                  start_threshold=0.5,
                                                  end_threshold=0.45)

            if not len(trigger_pred):
                continue

            trigger_labels = [x['trigger'] for x in _ex['events']]

            tmp_ex = {'sentence': text, 'distant_triggers': distant_triggers,
                      'events': []}

            for _label in trigger_labels:
                if _label['text'] == trigger_pred[0][0] and _label['offset'] == trigger_pred[0][1]:
                    tmp_ex['events'].append({'trigger': {'text': _label['text'],
                                                         'offset': _label['offset'],
                                                         'length': _label['length']}})
                    break

            if len(tmp_ex['events']):
                num_1 += 1
                trigger_third_new.append(tmp_ex)
            else:
                num_2 += 1

    print(f'Use {num_1} third examples, drop {num_2} examples')

    save_info(base_dir, trigger_third_new, 'trigger_third_new')


def split_preliminary_role_data(base_dir, save_dir):
    """
    将 role 的数据进行切分，与 trigger 的划分相似
    first： 模型预测的 sub + obj 与 labels 全部相同
    second：模型预测的 sub + obj 被 labels 中的包含或者 labels 中的包含预测的
    third： 模型预测的与 labels 中的无关
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    preliminary_examples = load_examples(os.path.join(base_dir, 'preliminary_stack.json'))

    first_roles = []
    second_roles = []
    third_roles = []

    for _ex in tqdm(preliminary_examples, desc='process preliminary examples with roles'):
        tmp_first_event = {'sentence': _ex['sentence'],
                           'events': [], 'pred_events': []}

        tmp_second_event = copy.deepcopy(tmp_first_event)
        tmp_third_event = copy.deepcopy(tmp_first_event)

        for _event in _ex['events']:
            for _pred in _ex['pred_events']:
                # label 和预测的 trigger 匹配
                if _pred['trigger']['text'] == _event['trigger']['text'] \
                        and _pred['trigger']['offset'] == _event['trigger']['offset']:

                    flag1 = True
                    flag2 = False
                    wrong = False
                    for _aug in _event['arguments']:
                        if _aug['role'] not in ['subject', 'object']:
                            continue

                        tag1 = False

                        tag2 = False

                        for _pred_aug in _pred['arguments']:
                            if _pred_aug['role'] == _aug['role']:
                                if _pred_aug['text'] == _aug['text'] and _pred_aug['offset'] == _aug['offset']:
                                    tag1 = True
                                else:
                                    # 存在某一个被包含
                                    _pred_start = _pred_aug['offset']
                                    _pred_end = _pred_start + len(_pred_aug['text']) - 1
                                    _aug_start = _aug['offset']
                                    _aug_end = _aug_start + len(_aug['text']) - 1
                                    if _aug_start <= _pred_start <= _pred_end <= _aug_end:
                                        tag2 = True
                                    else:
                                        wrong = True

                        # 预测的 sub / obj 不含有 gt 中的label
                        if not tag1:
                            flag1 = False

                        if tag2 and not wrong:
                            flag2 = True

                    if flag1:
                        tmp_first_event['events'].append(_event)
                    elif flag2:
                        tmp_second_event['events'].append(_event)
                        tmp_second_event['pred_events'].append(_pred)
                    else:
                        tmp_third_event['events'].append(_event)

                else:
                    continue

        if len(tmp_first_event['events']):
            first_roles.append(tmp_first_event)

        if len(tmp_second_event['events']):
            second_roles.append(tmp_second_event)

        if len(tmp_third_event['events']):
            third_roles.append(tmp_third_event)

    print(f'First role examples nums: {len(first_roles)}')
    print(f'Second role examples nums: {len(second_roles)}')
    print(f'Third role examples nums: {len(third_roles)}')

    save_info(save_dir, first_roles, 'role1_first')
    save_info(save_dir, second_roles, 'role1_second')
    save_info(save_dir, third_roles, 'role1_third')



if __name__ == '__main__':
    convert_raw_data('../../data/final', save_data=True, save_dict=False)
    split_preliminary_trigger_data('../../data/final/raw_data', '../../data/final/preliminary_clean')
    split_preliminary_role_data('../../data/final/raw_data', '../../data/final/preliminary_clean')

    # # 划分 trigger_third_new 需要用复赛的数据集训练出一个 trigger 模型 (只给出划分好的 trigger third)
    # split_preliminary_trigger_third_data(bert_dir='../../../bert/torch_roberta_wwm',
    #                                      base_dir='../../data/final/preliminary_clean',
    #                                      ckpt_path='')

