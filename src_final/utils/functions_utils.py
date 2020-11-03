# coding=utf-8
"""
@author: Oscar
@license: (C) Copyright 2019-2022, ZJU.
@contact: 499616042@qq.com
@software: pycharm
@file: functions_utils.py
@time: 2020/9/3 11:14
"""
import os
import copy
import json
import torch
import random
import logging
import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    设置随机种子
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_info(task_type, mid_data_dir):
    """
    prepare a dict for training in different task
    """
    assert task_type in ['trigger', 'role1', 'role2', 'attribution']

    info_dict = {}

    if task_type == 'attribution':

        with open(os.path.join(mid_data_dir, f'polarity2id.json'), encoding='utf-8') as f:
            polarity2id = json.load(f)
        with open(os.path.join(mid_data_dir, f'tense2id.json'), encoding='utf-8') as f:
            tense2id = json.load(f)

        polarity2id = polarity2id['map']
        tense2id = tense2id['map']

        info_dict['polarity2id'] = polarity2id
        info_dict['tense2id'] = tense2id

    return info_dict


def prepare_para_dict(opt, info_dict):
    feature_para, dataset_para, model_para = {}, {}, {}
    task_type = opt.task_type

    if hasattr(opt, 'dropout_prob'):
        model_para['dropout_prob'] = opt.dropout_prob

    if task_type == 'trigger':
        dataset_para['use_distant_trigger'] = opt.use_distant_trigger
        model_para['use_distant_trigger'] = opt.use_distant_trigger

    elif task_type in ['role1', 'role2']:
        dataset_para['use_trigger_distance'] = opt.use_trigger_distance
        model_para['use_trigger_distance'] = opt.use_trigger_distance
    else:
        feature_para['polarity2id'] = info_dict['polarity2id']
        feature_para['tense2id'] = info_dict['tense2id']

    return feature_para, dataset_para, model_para


def load_model_and_parallel(model, gpu_ids, ckpt_path=None, strict=True):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    """

    gpu_ids = gpu_ids.split(',')

    # set to device to the first cuda
    device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])

    if ckpt_path is not None:
        logger.info(f'Load ckpt from {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=strict)

    model.to(device)

    if len(gpu_ids) > 1:
        logger.info(f'Use multi gpus in: {gpu_ids}')
        gpu_ids = [int(x) for x in gpu_ids]
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        logger.info(f'Use single gpu in: {gpu_ids}')

    return model, device


def get_model_path_list(base_dir):
    """
    从文件夹中获取 model.pt 的路径
    """
    model_lists = []

    for root, dirs, files in os.walk(base_dir):
        for _file in files:
            if 'model.pt' == _file:
                model_lists.append(os.path.join(root, _file))

    model_lists = sorted(model_lists,
                         key=lambda x: (x.split('/')[-3], int(x.split('/')[-2].split('-')[-1])))

    return model_lists


def swa(model, model_dir, swa_start=1):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    model_path_list = get_model_path_list(model_dir)

#     assert 1 <= swa_start < len(model_path_list) - 1, \
#         f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0'

    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list[swa_start:]:
            logger.info(f'Load model from {_ckpt}')
            model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1

    # use 100000 to represent swa to avoid clash
    swa_model_dir = os.path.join(model_dir, f'checkpoint-100000')
    if not os.path.exists(swa_model_dir):
        os.mkdir(swa_model_dir)

    logger.info(f'Save swa model in: {swa_model_dir}')

    swa_model_path = os.path.join(swa_model_dir, 'model.pt')

    torch.save(swa_model.state_dict(), swa_model_path)

    return swa_model
