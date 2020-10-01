#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import copy
import json
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
import collections
from convert_raw_data import *


# In[4]:


def convert_raw_data(data_dir, save_data=False, save_dict=False,use_clean=False):
    """
    1、5 折交叉验证构造带标签数据的 distant trigger （长度大于一，并且出现次数大于一）
    2、将复赛数据 8.5 : 1.5 划分训练 / 验证数据；
    3、构建先验知识词典 ---- (tense_prob, polarity_prob)；
    4、构造 trigger 词典 （所有长度等于2的 trigger，去重后构造一个 trigger dict）
    """
    random.seed(123)

    raw_dir = os.path.join(data_dir, 'raw_data')
    mid_dir = os.path.join(data_dir, 'mid_data')
    if not os.path.exists(mid_dir):
        os.mkdir(mid_dir)


    test_examples = load_examples(os.path.join(raw_dir, 'sentences.json'))
    if not use_clean:
        stack_examples = load_examples(os.path.join(raw_dir, 'raw_stack.json'))
        preliminary_examples = load_examples(os.path.join(raw_dir, 'raw_preliminary.json'))
    else:
        stack_examples = load_examples(os.path.join(raw_dir, 'raw_stack_clean.json'))
        preliminary_examples = load_examples(os.path.join(raw_dir, 'raw_preliminary_clean.json')) 

    kf = KFold(10)

    triggers = {}

    nums = 0

    # 5折交叉构造 distant trigger
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
                if _t in tmp_sent :
                    #and now_triggers[_t] > 1
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
            if _t in tmp_sent :
                #and triggers[_t] > 1
                candidate_triggers.append(_t)

        _ex['distant_triggers'] = candidate_triggers
        if len(candidate_triggers) > nums:
            nums = len(candidate_triggers)

    print(nums)
    nums = 0

    # 构造 test 的 distant trigger
    for _ex in test_examples:
        tmp_sent = _ex['sentence']
        candidate_triggers = []
        for _t in triggers.keys():
            if _t in tmp_sent and triggers[_t] > 1:
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

    train, dev = train_test_split(stack_examples, shuffle=True, random_state=123, test_size=0.15)

    if save_data:
        save_info(raw_dir, stack_examples, 'stack')
        save_info(raw_dir, train, 'train')
        save_info(raw_dir, dev, 'dev')
        save_info(raw_dir, test_examples, 'test')
        save_info(raw_dir, preliminary_examples, 'preliminary_stack')

    if save_dict:
        save_info(mid_dir, tense2id, 'tense2id')
        save_info(mid_dir, polarity2id, 'polarity2id')
        save_info(mid_dir, triggers_dict, 'triggers_dict')
if __name__ == '__main__':
    stack = []
    with open(os.path.join("../../data/preliminary/raw_data/", 'stack.json'), encoding='utf-8') as f:
        for line in f.readlines():
            stack.append(json.loads(line.strip()))
    new_stack=[]
    for sample in tqdm(stack):
        new_sample={'sentence':sample['text'],'events':[],'distant_triggers':sample['distant_trigger']}
        for event in sample['labels']:
            new_event={'trigger':{'text':event['trigger'][0],'length':len(event['trigger'][0]),'offset':event['trigger'][1]},                       'tense': '过去','polarity': '肯定',                       'arguments':[]}
            for role in event.keys():
                if role=='trigger':
                    continue
                if isinstance(event[role],list):
                    role_name=role if role!='location' else 'loc'
                    if role_name=='subject':
                        role_name='object'
                    elif role_name=='object':
                        role_name='subject'
                    new_event['arguments'].append({'role':role_name, 'text': event[role][0], 'offset': event[role][1], 'length':len(event[role][0])})
            new_sample['events'].append(new_event)
        new_stack.append(new_sample)
    save_info("../../data/final/raw_data/",new_stack,'raw_preliminary')
    raw_dir="../../data/final/raw_data/"
    raw_pre_examples = load_examples(os.path.join(raw_dir, 'raw_preliminary.json'))
    #生成复赛样式的初赛数据

    raw_pre_examples,nums=clean_data(raw_pre_examples)
    print(nums)
    #对初赛数据进行清洗
    raw_examples = load_examples(os.path.join(raw_dir, 'raw_stack.json'))
    raw_examples,nums=clean_data(raw_examples)
    print(nums)
    save_info("../../data/final/raw_data/",raw_pre_examples,'raw_preliminary_clean')
    save_info("../../data/final/raw_data/",raw_examples,'raw_stack_clean')

    convert_raw_data('../../data/final', save_data=True, save_dict=True,use_clean=True)
    #初赛数据的样本从复赛数据中读取distant triggers
