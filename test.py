# coding=utf-8
"""
@author: Oscar
@license: (C) Copyright 2019-2022, ZJU.
@contact: 499616042@qq.com
@software: pycharm
@file: test.py
@time: 2020/7/30 16:23
"""
import os
import copy
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from ensemble_predict import ensemble_attribution
from src_final.preprocess.convert_raw_data import clean_data
from src_final.preprocess.processor import fine_grade_tokenize, search_label_index, ROLE2_TO_ID
from src_final.utils.options import TestArgs
from src_final.utils.functions_utils import load_model_and_parallel
from src_final.utils.evaluator import pointer_trigger_decode, pointer_decode, crf_decode
from src_final.utils.model_utils import TriggerExtractor, Role2Extractor, Role1Extractor, AttributionClassifier


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def pipeline_predict(opt):
    """
    pipeline predict the submit results
    """
    if not os.path.exists(opt.submit_dir):
        os.mkdir(opt.submit_dir)

    submit = []

    with open(os.path.join(opt.raw_data_dir, 'test.json'), encoding='utf-8') as f:
        text_examples = json.load(f)

    tokenizer = BertTokenizer.from_pretrained(opt.bert_dir)

    trigger_model = TriggerExtractor(bert_dir=opt.bert_dir,
                                     use_distant_trigger=opt.use_distant_trigger)

    role1_model = Role1Extractor(bert_dir=opt.bert_dir,
                                 use_trigger_distance=opt.role1_use_trigger_distance)

    role2_model = Role2Extractor(bert_dir=opt.bert_dir,
                                 use_trigger_distance=opt.role2_use_trigger_distance)

    attribution_model = AttributionClassifier(bert_dir=opt.bert_dir)

    logger.info('Load models')
    trigger_model, device = load_model_and_parallel(trigger_model, opt.gpu_ids[0],
                                                    ckpt_path=os.path.join(opt.trigger_ckpt_dir, 'model.pt'))

    role1_model, _ = load_model_and_parallel(role1_model, opt.gpu_ids[0],
                                             ckpt_path=os.path.join(opt.role1_ckpt_dir, 'model.pt'))

    role2_model, _ = load_model_and_parallel(role2_model, opt.gpu_ids[0],
                                             ckpt_path=os.path.join(opt.role2_ckpt_dir, 'model.pt'))

    attribution_model, _ = load_model_and_parallel(attribution_model, opt.gpu_ids[0],
                                                   ckpt_path=os.path.join(opt.attribution_ckpt_dir, 'model.pt'))

    id2role = {ROLE2_TO_ID[key]: key for key in ROLE2_TO_ID.keys()}

    start_threshold = opt.role1_start_threshold
    end_threshold = opt.role1_end_threshold

    with open(os.path.join(opt.mid_data_dir, f'polarity2id.json'), encoding='utf-8') as f:
        polarity2id = json.load(f)
    with open(os.path.join(opt.mid_data_dir, f'tense2id.json'), encoding='utf-8') as f:
        tense2id = json.load(f)

    polarity2id = polarity2id['map']
    tense2id = tense2id['map']

    id2polarity = {polarity2id[key]: key for key in polarity2id.keys()}
    id2tense = {tense2id[key]: key for key in tense2id.keys()}

    counts = 0
    with torch.no_grad():
        trigger_model.eval()
        role1_model.eval()
        role2_model.eval()
        attribution_model.eval()

        for _ex in tqdm(text_examples, desc='decode test examples'):
            distant_triggers = _ex['distant_triggers']

            tmp_instance = {'sentence': _ex['sentence'],
                            'words': _ex['words']}

            tmp_text = _ex['sentence']
            tmp_text_tokens = fine_grade_tokenize(tmp_text, tokenizer)

            assert len(tmp_text) == len(tmp_text_tokens)

            trigger_encode_dict = tokenizer.encode_plus(text=tmp_text_tokens,
                                                        max_length=512,
                                                        pad_to_max_length=False,
                                                        is_pretokenized=True,
                                                        return_token_type_ids=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')

            tmp_base_inputs = {'token_ids': trigger_encode_dict['input_ids'],
                               'attention_masks': trigger_encode_dict['attention_mask'],
                               'token_type_ids': trigger_encode_dict['token_type_ids']}

            trigger_inputs = copy.deepcopy(tmp_base_inputs)

            # 构造 test 里的 distant trigger
            if opt.use_distant_trigger:
                distant_trigger_label = [0] * len(tmp_text)
                for _trigger in distant_triggers:
                    tmp_trigger_tokens = fine_grade_tokenize(_trigger, tokenizer)
                    tmp_index_list = search_label_index(tmp_text_tokens, tmp_trigger_tokens)

                    assert len(tmp_index_list)

                    for _index in tmp_index_list:
                        for i in range(_index[0], _index[1] + 1):
                            distant_trigger_label[i] = 1

                if len(distant_trigger_label) > 510:
                    distant_trigger_label = distant_trigger_label[:510]
                distant_trigger_label = [0] + distant_trigger_label + [0]

                distant_trigger_label = torch.tensor([distant_trigger_label]).long()
                trigger_inputs['distant_trigger'] = distant_trigger_label

            for key in trigger_inputs.keys():
                trigger_inputs[key] = trigger_inputs[key].to(device)

            tmp_trigger_pred = trigger_model(**trigger_inputs)[0][0]

            tmp_trigger_pred = tmp_trigger_pred.cpu().numpy()[1:1 + len(tmp_text)]

            tmp_triggers = pointer_trigger_decode(tmp_trigger_pred, tmp_text, distant_triggers,
                                                  start_threshold=opt.trigger_start_threshold,
                                                  end_threshold=opt.trigger_end_threshold,
                                                  one_trigger=True)

            if not len(tmp_triggers):
                print(_ex['sentence'])

            events = []

            for _trigger in tmp_triggers:
                tmp_event = {'trigger': {'text': _trigger[0],
                                         'length': len(_trigger[0]),
                                         'offset': int(_trigger[1])},
                             'arguments': []}

                if len(_trigger) > 2:
                    print(_trigger)

                role_inputs = copy.deepcopy(tmp_base_inputs)

                # TODO 此处 start end 与新一版本的模型不一致，此版本是正确的
                trigger_start = _trigger[1] + 1
                trigger_end = trigger_start + len(_trigger[0]) - 1

                for i in range(trigger_start, trigger_end + 1):
                    role_inputs['token_type_ids'][0][i] = 1

                tmp_trigger_label = torch.tensor([[trigger_start, trigger_end]]).long()

                role_inputs['trigger_index'] = tmp_trigger_label

                trigger_distance = [511] * (len(tmp_text) + 2)
                for i in range(len(tmp_text)):
                    if trigger_start <= i <= trigger_end:
                        trigger_distance[i] = 0
                        continue
                    elif i < trigger_start:
                        trigger_distance[i] = trigger_start - i
                    else:
                        trigger_distance[i] = i - trigger_end

                if opt.role1_use_trigger_distance or opt.role2_use_trigger_distance:
                    role_inputs['trigger_distance'] = torch.tensor([trigger_distance]).long()

                for key in role_inputs.keys():
                    role_inputs[key] = role_inputs[key].to(device)

                tmp_roles_pred = role1_model(**role_inputs)[0][0].cpu().numpy()

                tmp_roles_pred = tmp_roles_pred[1:1 + len(tmp_text)]

                pred_obj = pointer_decode(tmp_roles_pred[:, :2], tmp_text, start_threshold, end_threshold, True)

                pred_sub = pointer_decode(tmp_roles_pred[:, 2:], tmp_text, start_threshold, end_threshold, True)

                if len(pred_obj) > 1:
                    print(pred_obj)

                if len(pred_sub) > 1:
                    print(pred_sub)

                pred_aux_tokens = role2_model(**role_inputs)[0][0]
                pred_aux = crf_decode(pred_aux_tokens, tmp_text, id2role)

                for _obj in pred_obj:
                    tmp_event['arguments'].append({'role': 'object', 'text': _obj[0],
                                                   'offset': int(_obj[1]), 'length': len(_obj[0])})
                for _sub in pred_sub:
                    tmp_event['arguments'].append({'role': 'subject', 'text': _sub[0],
                                                   'offset': int(_sub[1]), 'length': len(_sub[0])})

                for _role_type in pred_aux.keys():
                    for _role in pred_aux[_role_type]:
                        tmp_event['arguments'].append({'role': _role_type, 'text': _role[0],
                                                       'offset': int(_role[1]), 'length': len(_role[0])})

                att_inputs = copy.deepcopy(tmp_base_inputs)

                att_inputs['trigger_index'] = tmp_trigger_label

                window_size = 20

                pooling_masks_range = range(max(1, trigger_start - window_size),
                                            min(min(1 + len(tmp_text), 511), trigger_end + window_size))

                pooling_masks = [0] * (2 + len(tmp_text))
                for i in pooling_masks_range:
                    pooling_masks[i] = 1
                for i in range(trigger_start, trigger_end + 1):
                    pooling_masks[i] = 0

                att_inputs['pooling_masks'] = torch.tensor([pooling_masks]).float()

                for key in att_inputs.keys():
                    att_inputs[key] = att_inputs[key].to(device)

                polarity_logits, tense_logits = attribution_model(**att_inputs)

                polarity_logits = polarity_logits[0].cpu().numpy()
                tense_logits = tense_logits[0].cpu().numpy()

                tense = id2tense[np.argmax(tense_logits)]
                polarity = id2polarity[np.argmax(polarity_logits)]

                tmp_event['polarity'] = polarity
                tmp_event['tense'] = tense

                events.append(tmp_event)

            tmp_instance['events'] = events
            submit.append(tmp_instance)

    submit, nums = clean_data(submit)

    print(f'Clean {nums} data')
    with open(os.path.join(opt.submit_dir, f'submit_{opt.version}.json'), 'w', encoding='utf-8') as f:
        json.dump(submit, f, ensure_ascii=False, indent=2)

    logger.info(f'{counts} blank examples')


if __name__ == '__main__':
    args = TestArgs().get_parser()

    if '_distant_trigger' in args.trigger_ckpt_dir:
        logger.info('Use distant trigger in trigger extractor')
        args.use_distant_trigger = True

    if '_distance' in args.role1_ckpt_dir:
        logger.info('Use trigger distance in sub & obj extractor')
        args.role1_use_trigger_distance = True

    if '_distance' in args.role2_ckpt_dir:
        logger.info('Use trigger distance in time & loc extractor')
        args.role2_use_trigger_distance = True

    logger.info(f'Submit version: {args.version}')

    # 获取单模型结果
    pipeline_predict(args)

#     # 获取 attribution 中十折交叉验证的结果
#     ensemble_attribution(args.version)
