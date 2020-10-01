# coding=utf-8
"""
@author: Oscar
@license: (C) Copyright 2019-2022, ZJU.
@contact: 499616042@qq.com
@software: pycharm
@file: evaluator.py
@time: 2020/9/2 15:19
"""
import torch
import logging
import numpy as np
from tqdm import tqdm
from src_final.preprocess.processor import ROLE2_TO_ID, search_label_index


logger = logging.getLogger(__name__)


def get_base_out(model, loader, device, task_type):
    """
    每一个任务的 forward 都一样，封装起来
    """
    model.eval()

    with torch.no_grad():
        for idx, _batch in enumerate(tqdm(loader, desc=f'Get {task_type} task predict logits')):

            for key in _batch.keys():
                _batch[key] = _batch[key].to(device)

            tmp_out = model(**_batch)

            yield tmp_out


def pointer_trigger_decode(logits, raw_text, distant_triggers, start_threshold=0.5, end_threshold=0.5,
                           one_trigger=True):
    candidate_entities = []

    start_ids = np.argwhere(logits[:, 0] > start_threshold)[:, 0]
    end_ids = np.argwhere(logits[:, 1] > end_threshold)[:, 0]

    # 选最短的
    for _start in start_ids:
        for _end in end_ids:
            # 限定 trigger 长度不能超过 3
            if _end >= _start and _end - _start <= 2:
                # (start, end, start_logits + end_logits)
                candidate_entities.append((raw_text[_start: _end + 1], _start, logits[_start][0] + logits[_end][1]))
                break

    if not len(candidate_entities):
        for _dis_trigger in distant_triggers:
            trigger_ids = search_label_index(raw_text, _dis_trigger)

            for idx in trigger_ids:
                if idx[1] >= len(logits):
                    continue
                candidate_entities.append((raw_text[idx[0]: idx[1] + 1], idx[0],
                                           logits[idx[0]][0] + logits[idx[1]][1]))

    entities = []

    if len(candidate_entities):
        candidate_entities = sorted(candidate_entities, key=lambda x: x[-1], reverse=True)

        if one_trigger:
            # 只解码一个，返回 logits 最大的 trigger
            entities.append(candidate_entities[0][:2])
        else:
            # 解码多个，返回多个 trigger + 对应的 logits
            for _ent in candidate_entities:
                entities.append(_ent[:2])
    else:
        # 最后还是没有解码出 trigger 时选取 logits 最大的作为 trigger
        start_ids = np.argmax(logits[:, 0])
        end_ids = np.argmax(logits[:, 1])

        if end_ids < start_ids:
            end_ids = start_ids + np.argmax(logits[start_ids:, 1])

        entities.append((raw_text[start_ids: end_ids + 1], int(start_ids)))

    return entities


def pointer_decode(logits, raw_text, start_threshold=0.5, end_threshold=0.5, force_decode=False):
    """
    :param logits:          sub / obj 最后输出的 logits，第一行为 start 第二行为 end
    :param raw_text:        原始文本
    :param start_threshold: logits start 位置大于阈值即可解码
    :param end_threshold:   logits end 位置大于阈值即可解码
    :param force_decode:    强制解码输出
    :return:
    [(entity, offset),...]
    """
    entities = []
    candidate_entities = []

    start_ids = np.argwhere(logits[:, 0] > start_threshold)[:, 0]
    end_ids = np.argwhere(logits[:, 1] > end_threshold)[:, 0]

    # 选最短的
    for _start in start_ids:
        for _end in end_ids:
            if _end >= _start:
                # (start, end, logits)
                candidate_entities.append((_start, _end, logits[_start][0] + logits[_end][1]))
                break

    # 找整个候选集，如果存在包含的实体对选 logits 最高的作为候选
    for x in candidate_entities:
        flag = True
        for y in candidate_entities:
            if x == y:
                continue

            text_x = raw_text[x[0]:x[1] + 1]
            text_y = raw_text[y[0]:y[1] + 1]

            if text_x in text_y or text_y in text_x:
                if y[2] > x[2]:
                    flag = False
                    break
        if flag:
            entities.append((raw_text[x[0]:x[1] + 1], int(x[0])))

    if force_decode and not len(entities):
        start_ids = np.argmax(logits[:, 0])
        end_ids = np.argmax(logits[:, 1])

        if end_ids < start_ids:
            end_ids = start_ids + np.argmax(logits[start_ids:, 1])

        entities.append((raw_text[start_ids: end_ids + 1], int(start_ids)))

    return entities


def crf_decode(decode_tokens, raw_text, id2label):
    """
    CRF 解码，用于解码 time loc 的提取
    """
    predict_entities = {}

    decode_tokens = decode_tokens[1:-1]  # 除去 CLS SEP token

    index_ = 0

    while index_ < len(decode_tokens):

        token_label = id2label[decode_tokens[index_]].split('-')

        if token_label[0].startswith('S'):
            token_type = token_label[1]
            tmp_ent = raw_text[index_]

            if token_type not in predict_entities:
                predict_entities[token_type] = [(tmp_ent, index_)]
            else:
                predict_entities[token_type].append((tmp_ent, int(index_)))

            index_ += 1

        elif token_label[0].startswith('B'):
            token_type = token_label[1]
            start_index = index_

            index_ += 1
            while index_ < len(decode_tokens):
                temp_token_label = id2label[decode_tokens[index_]].split('-')

                if temp_token_label[0].startswith('I') and token_type == temp_token_label[1]:
                    index_ += 1
                elif temp_token_label[0].startswith('E') and token_type == temp_token_label[1]:
                    end_index = index_
                    index_ += 1

                    tmp_ent = raw_text[start_index: end_index + 1]

                    if token_type not in predict_entities:
                        predict_entities[token_type] = [(tmp_ent, start_index)]
                    else:
                        predict_entities[token_type].append((tmp_ent, int(start_index)))

                    break
                else:
                    break

        else:
            index_ += 1

    return predict_entities


def calculate_metric(gt, predict):
    """
    计算 tp fp fn
    """
    tp, fp, fn = 0, 0, 0
    for entity_predict in predict:
        flag = 0
        for entity_gt in gt:
            if entity_predict[0] == entity_gt[0] and entity_predict[1] == entity_gt[1]:
                flag = 1
                tp += 1
                break
        if flag == 0:
            fp += 1

    fn = len(gt) - tp

    return np.array([tp, fp, fn])


def get_p_r_f(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return np.array([p, r, f1])


def trigger_evaluation(model, dev_info, device, **kwargs):
    """
    线下评估 trigger 模型
    """
    dev_loader, dev_callback_info = dev_info

    pred_logits = None

    for tmp_pred in get_base_out(model, dev_loader, device, 'role'):
        tmp_pred = tmp_pred[0].cpu().numpy()

        if pred_logits is None:
            pred_logits = tmp_pred
        else:
            pred_logits = np.append(pred_logits, tmp_pred, axis=0)

    assert len(pred_logits) == len(dev_callback_info)

    start_threshold = kwargs.pop('start_threshold')
    end_threshold = kwargs.pop('end_threshold')

    zero_pred = 0

    tp, fp, fn = 0, 0, 0

    for tmp_pred, tmp_callback in zip(pred_logits, dev_callback_info):
        text, gt_triggers, distant_triggers = tmp_callback
        tmp_pred = tmp_pred[1:1 + len(text)]

        pred_triggers = pointer_trigger_decode(tmp_pred, text, distant_triggers,
                                               start_threshold=start_threshold,
                                               end_threshold=end_threshold)

        if not len(pred_triggers):
            zero_pred += 1

        tmp_tp, tmp_fp, tmp_fn = calculate_metric(gt_triggers, pred_triggers)

        tp += tmp_tp
        fp += tmp_fp
        fn += tmp_fn

    p, r, f1 = get_p_r_f(tp, fp, fn)

    metric_str = f'In start threshold: {start_threshold}; end threshold: {end_threshold}\n'
    metric_str += f'[MIRCO] precision: {p:.4f}, recall: {r:.4f}, f1: {f1:.4f}\n'
    metric_str += f'Zero pred nums: {zero_pred}'

    return metric_str, f1


def role1_evaluation(model, dev_info, device, **kwargs):
    dev_loader, (dev_callback_info, type_weight) = dev_info

    pred_logits = None

    for tmp_pred in get_base_out(model, dev_loader, device, 'role'):
        tmp_pred = tmp_pred[0].cpu().numpy()

        if pred_logits is None:
            pred_logits = tmp_pred
        else:
            pred_logits = np.append(pred_logits, tmp_pred, axis=0)

    assert len(pred_logits) == len(dev_callback_info)

    start_threshold = kwargs.pop('start_threshold')
    end_threshold = kwargs.pop('end_threshold')

    role_metric = np.zeros([2, 3])

    mirco_metrics = np.zeros(3)

    role_types = ['object', 'subject']

    for tmp_pred, tmp_callback in zip(pred_logits, dev_callback_info):
        text, trigger, gt_roles = tmp_callback
        tmp_pred = tmp_pred[1:len(text) + 1]

        pred_obj = pointer_decode(tmp_pred[:, :2], text, start_threshold, end_threshold, True)
        pred_sub = pointer_decode(tmp_pred[:, 2:], text, start_threshold, end_threshold, True)

        tmp_metric = np.zeros([2, 3])

        pred_roles = {'subject': pred_sub, 'object': pred_obj}

        for idx, _role in enumerate(role_types):
            tmp_metric[idx] += calculate_metric(gt_roles[_role], pred_roles[_role])

        role_metric += tmp_metric

    metric_str = f'In start threshold: {start_threshold}; end threshold: {end_threshold}\n'

    for idx, _role in enumerate(role_types):
        temp_metric = get_p_r_f(role_metric[idx][0], role_metric[idx][1], role_metric[idx][2])

        mirco_metrics += temp_metric * type_weight[_role]

        metric_str += '[%s] precision: %.4f, recall: %.4f, f1: %.4f.\n' % \
                      (_role, temp_metric[0], temp_metric[1], temp_metric[2])

    metric_str += f'[MIRCO] precision: {mirco_metrics[0]:.4f}, ' \
                  f'recall: {mirco_metrics[1]:.4f}, f1: {mirco_metrics[2]:.4f}'

    return metric_str, mirco_metrics[2]


def role2_evaluation(model, dev_info, device):
    dev_loader, (dev_callback_info, type_weight) = dev_info

    pred_tokens = []

    for tmp_pred in get_base_out(model, dev_loader, device, 'role'):
        pred_tokens.extend(tmp_pred[0])

    assert len(pred_tokens) == len(dev_callback_info)

    id2role = {ROLE2_TO_ID[key]: key for key in ROLE2_TO_ID.keys()}

    role_metric = np.zeros([2, 3])

    mirco_metrics = np.zeros(3)

    metric_str = ''

    role_types = ['time', 'loc']

    for tmp_tokens, tmp_callback in zip(pred_tokens, dev_callback_info):

        text, trigger, gt_roles = tmp_callback

        tmp_metric = np.zeros([2, 3])

        pred_roles = crf_decode(tmp_tokens, text, id2role)

        for idx, _role in enumerate(role_types):
            if _role not in pred_roles:
                pred_roles[_role] = []

            tmp_metric[idx] += calculate_metric(gt_roles[_role], pred_roles[_role])

        role_metric += tmp_metric

    for idx, _role in enumerate(role_types):
        temp_metric = get_p_r_f(role_metric[idx][0], role_metric[idx][1], role_metric[idx][2])

        mirco_metrics += temp_metric * type_weight[_role]

        metric_str += '[%s] precision: %.4f, recall: %.4f, f1: %.4f.\n' % \
                      (_role, temp_metric[0], temp_metric[1], temp_metric[2])

    metric_str += f'[MIRCO] precision: {mirco_metrics[0]:.4f}, ' \
                  f'recall: {mirco_metrics[1]:.4f}, f1: {mirco_metrics[2]:.4f}'

    return metric_str, mirco_metrics[2]


def attribution_evaluation(model, dev_info, device, **kwargs):
    dev_loader, dev_callback_info = dev_info

    polarity_logits, tense_logits = None, None

    for tmp_pred in get_base_out(model, dev_loader, device, 'attribution'):
        tmp_polarity_logits, tmp_tense_logits = tmp_pred

        tmp_polarity_logits = tmp_polarity_logits.cpu().numpy()
        tmp_tense_logits = tmp_tense_logits.cpu().numpy()

        if tense_logits is None:
            polarity_logits = tmp_polarity_logits
            tense_logits = tmp_tense_logits
        else:
            polarity_logits = np.append(polarity_logits, tmp_polarity_logits, axis=0)
            tense_logits = np.append(tense_logits, tmp_tense_logits, axis=0)

    assert len(tense_logits) == len(dev_callback_info)

    polarity2id = kwargs.pop('polarity2id')
    tense2id = kwargs.pop('tense2id')

    id2polarity = {polarity2id[key]: key for key in polarity2id.keys()}
    id2tense = {tense2id[key]: key for key in tense2id.keys()}

    polarity_acc = 0.
    tense_acc = 0.
    counts = 0.

    for tmp_pred_tense, tmp_pred_polarity, tmp_callback in \
            zip(tense_logits, polarity_logits, dev_callback_info):
        text, trigger, gt_attributions = tmp_callback

        pred_polarity = id2polarity[np.argmax(tmp_pred_polarity)]

        pred_tense = id2tense[np.argmax(tmp_pred_tense)]

        if pred_tense == gt_attributions[0]:
            tense_acc += 1


        if pred_polarity == gt_attributions[1]:
            polarity_acc += 1

        counts += 1

    metric_str = ''

    polarity_acc /= counts
    tense_acc /= counts

    metric_str += f'[ACC] polarity: {polarity_acc:.4f}, tense: {tense_acc:.4f}'

    return metric_str, (polarity_acc + tense_acc) / 2
