# coding=utf-8
"""
@author: Oscar
@license: (C) Copyright 2019-2022, ZJU.
@contact: 499616042@qq.com
@software: pycharm
@file: ensemble_predict.py
@time: 2020/9/18 20:04
"""
import os
import copy
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from src_final.preprocess.processor import fine_grade_tokenize
from src_final.utils.functions_utils import get_model_path_list, prepare_info
from src_final.utils.model_utils import AttributionClassifier

logger = logging.getLogger(__name__)

# 全局常量
ERNIE_BERT_DIR = '../bert/torch_ernie_1'

MID_DATA_DIR = './data/final/mid_data'
STACK_DIR = './out/stack'
SUBMIT_DIR = './submit'


def base_attribution_predict(examples, model, device, tokenizer, desc):

    polarity_logits, tense_logits = None, None

    for _ex in tqdm(examples, desc=f'Ensemble attribution in model {desc}'):
        text = _ex['sentence']

        tokens = fine_grade_tokenize(text, tokenizer)

        encode_dict = tokenizer.encode_plus(text=tokens,
                                            max_length=512,
                                            pad_to_max_length=False,
                                            is_pretokenized=True,
                                            return_token_type_ids=True,
                                            return_attention_mask=True,
                                            return_tensors='pt')

        base_inputs = {'token_ids': encode_dict['input_ids'],
                       'attention_masks': encode_dict['attention_mask'],
                       'token_type_ids': encode_dict['token_type_ids']}

        for _event in _ex['events']:
            trigger = _event['trigger']
            trigger_start = trigger['offset'] + 1
            trigger_end = trigger['offset'] + len(trigger['text'])
            trigger_loc = torch.tensor([[trigger_start, trigger_end]]).long()

            window_size = 20

            pooling_masks_range = range(max(1, trigger_start - window_size),
                                        min(min(1 + len(text), 511), trigger_end + window_size))

            pooling_masks = [0] * (2 + len(text))
            for i in pooling_masks_range:
                pooling_masks[i] = 1
            for i in range(trigger_start, trigger_end + 1):
                pooling_masks[i] = 0

            pooling_masks = torch.tensor([pooling_masks]).float()

            model_inputs = copy.deepcopy(base_inputs)
            model_inputs['trigger_index'] = trigger_loc
            model_inputs['pooling_masks'] = pooling_masks

            for key in model_inputs.keys():
                model_inputs[key] = model_inputs[key].to(device)

            tmp_polarity, tmp_tense = model(**model_inputs)

            tmp_polarity = tmp_polarity[0].cpu().numpy().reshape([1, -1])
            tmp_tense = tmp_tense[0].cpu().numpy().reshape([1, -1])

            if polarity_logits is None:
                polarity_logits = tmp_polarity
                tense_logits = tmp_tense
            else:
                polarity_logits = np.append(polarity_logits, tmp_polarity, axis=0)
                tense_logits = np.append(tense_logits, tmp_tense, axis=0)

    return polarity_logits, tense_logits


def ensemble_attribution(version):
    """
    将 attribution 用百度 ERNIE 模型交叉验证
    """
    logger.info('Ensemble attribution')
    info_dict = prepare_info(task_type='attribution', mid_data_dir=MID_DATA_DIR)

    polarity2id = info_dict['polarity2id']
    tense2id = info_dict['tense2id']

    id2polarity = {polarity2id[key]: key for key in polarity2id.keys()}
    id2tense = {tense2id[key]: key for key in tense2id.keys()}

    # 需要进行 ensemble 的最优文件
    with open(os.path.join(SUBMIT_DIR, f'submit_{version}.json'), encoding='utf-8') as f:
        examples = json.load(f)

    ernie_tokenizer = BertTokenizer.from_pretrained(ERNIE_BERT_DIR)

    ernie_model_dir = os.path.join(STACK_DIR, 'attribution', 'ernie_pgd')

    ernie_models_path = get_model_path_list(ernie_model_dir)

    ernie_model = AttributionClassifier(bert_dir=ERNIE_BERT_DIR)

    device = torch.device('cuda:1')

    all_polarity_logits, all_tense_logits = None, None
    count = 0.

    with torch.no_grad():

        for idx, _model_path in enumerate(ernie_models_path):
            logger.info(f'Load ckpt from {_model_path}')
            ernie_model.load_state_dict(torch.load(_model_path, map_location=torch.device('cpu')))
            ernie_model.eval()
            ernie_model.to(device)

            ernie_polarity_logits, ernie_tense_logits = base_attribution_predict(examples, ernie_model, device,
                                                                                 ernie_tokenizer, f'ernie {idx}')

            if all_polarity_logits is None:
                all_polarity_logits = ernie_polarity_logits
                all_tense_logits = ernie_tense_logits
            else:
                all_polarity_logits += ernie_polarity_logits
                all_tense_logits += ernie_tense_logits

            count += 1

    all_polarity_logits /= float(count)
    all_tense_logits /= float(count)

    polarity = np.argmax(all_polarity_logits, axis=-1)
    tense = np.argmax(all_tense_logits, axis=-1)

    idx = 0

    for _ex in tqdm(examples, desc=f'modify attribution'):
        for _event in _ex['events']:
            tmp_polarity = id2polarity[polarity[idx]]
            tmp_tense = id2tense[tense[idx]]
            idx += 1

            _event['polarity'] = tmp_polarity
            _event['tense'] = tmp_tense

    with open(os.path.join(SUBMIT_DIR, f'submit_{version}_ensemble.json'), 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
