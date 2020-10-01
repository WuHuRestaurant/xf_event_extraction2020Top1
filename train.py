# coding=utf-8
"""
@author: Oscar
@license: (C) Copyright 2019-2022, ZJU.
@contact: 499616042@qq.com
@software: pycharm
@file: train.py
@time: 2020/7/30 15:54
"""
import os
import logging
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from src_final.preprocess.processor import *
from src_final.utils.trainer import train
from src_final.utils.options import TrainArgs
from src_final.utils.model_utils import build_model
from src_final.utils.dataset_utils import build_dataset
from src_final.utils.evaluator import trigger_evaluation, role1_evaluation, role2_evaluation, attribution_evaluation
from src_final.utils.functions_utils import set_seed, get_model_path_list, load_model_and_parallel, \
    prepare_info, prepare_para_dict


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def train_base(opt, info_dict, train_examples, dev_info=None):
    feature_para, dataset_para, model_para = prepare_para_dict(opt, info_dict)

    train_features = convert_examples_to_features(opt.task_type, train_examples, opt.bert_dir,
                                                  opt.max_seq_len, **feature_para)

    logger.info(f'Build {len(train_features)} train features')

    train_dataset = build_dataset(opt.task_type, train_features, 'train', **dataset_para)

    model = build_model(opt.task_type, opt.bert_dir, **model_para)

    train(opt, model, train_dataset)

    if dev_info is not None:
        dev_examples, dev_callback_info = dev_info

        dev_features = convert_examples_to_features(opt.task_type, dev_examples, opt.bert_dir,
                                                    opt.max_seq_len, **feature_para)

        logger.info(f'Build {len(dev_features)} dev features')

        dev_dataset = build_dataset(opt.task_type, dev_features, 'dev', **dataset_para)

        dev_loader = DataLoader(dev_dataset, batch_size=opt.eval_batch_size,
                                shuffle=False, num_workers=8)

        dev_info = (dev_loader, dev_callback_info)

        model_path_list = get_model_path_list(opt.output_dir)

        metric_str = ''

        max_f1 = 0.
        max_f1_step = 0

        for idx, model_path in enumerate(model_path_list):

            tmp_step = model_path.split('/')[-2].split('-')[-1]

            model, device = load_model_and_parallel(model, opt.gpu_ids[0],
                                                    ckpt_path=model_path)

            if opt.task_type == 'trigger':

                tmp_metric_str, tmp_f1 = trigger_evaluation(model, dev_info, device,
                                                            start_threshold=opt.start_threshold,
                                                            end_threshold=opt.end_threshold)

            elif opt.task_type == 'role1':
                tmp_metric_str, tmp_f1 = role1_evaluation(model, dev_info, device,
                                                          start_threshold=opt.start_threshold,
                                                          end_threshold=opt.end_threshold)
            elif opt.task_type == 'role2':
                tmp_metric_str, tmp_f1 = role2_evaluation(model, dev_info, device)
            else:
                tmp_metric_str, tmp_f1 = attribution_evaluation(model, dev_info, device,
                                                                polarity2id=info_dict['polarity2id'],
                                                                tense2id=info_dict['tense2id'])

            logger.info(f'In step {tmp_step}: {tmp_metric_str}')

            metric_str += f'In step {tmp_step}: {tmp_metric_str}' + '\n\n'

            if tmp_f1 > max_f1:
                max_f1 = tmp_f1
                max_f1_step = tmp_step

        max_metric_str = f'Max f1 is: {max_f1}, in step {max_f1_step}'

        logger.info(max_metric_str)

        metric_str += max_metric_str + '\n'

        eval_save_path = os.path.join(opt.output_dir, 'eval_metric.txt')

        with open(eval_save_path, 'a', encoding='utf-8') as f1:
            f1.write(metric_str)


def training(opt):
    processors = {'trigger': TriggerProcessor,
                  'role1': RoleProcessor,
                  'role2': RoleProcessor,
                  'attribution': AttributionProcessor}

    processor = processors[opt.task_type]()

    info_dict = prepare_info(opt.task_type, opt.mid_data_dir)

    train_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, 'stack.json'))
    train_examples = processor.get_train_examples(train_raw_examples)

    if opt.enhance_data and opt.task_type in ['trigger', 'role1', 'role2']:
        # trigger & role1
        if opt.task_type in ['trigger', 'role1']:
            train_aux_raw_examples = processor.read_json(os.path.join(opt.aux_data_dir, f'{opt.task_type}_first.json'))
            train_examples += processor.get_train_examples(train_aux_raw_examples)

        # sub & obj 用第二部分数据进行增强
        if opt.task_type == 'role1':
            logger.info('Using second data to enhance subject and object')
            train_aux_raw_examples = processor.read_json(os.path.join(opt.aux_data_dir, f'{opt.task_type}_second.json'))
            train_examples += processor.get_train_examples(train_aux_raw_examples)
        # time & loc 用初赛全部数据进行增强
        elif opt.task_type == 'role2':
            train_aux_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, 'preliminary_stack.json'))
            train_examples += processor.get_train_examples(train_aux_raw_examples)
        # trigger 用更正的第三部分数据进行增强
        else:
            logger.info('Use third data to enhance trigger')
            train_aux_raw_examples = processor.read_json(os.path.join(opt.aux_data_dir,
                                                                      f'{opt.task_type}_third_new.json'))
            train_examples += processor.get_train_examples(train_aux_raw_examples)

    dev_info = None
    if opt.eval_model:
        dev_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, 'dev.json'))
        dev_info = processor.get_dev_examples(dev_raw_examples)

    train_base(opt, info_dict, train_examples, dev_info)


def stacking(opt):
    """
    10 折交叉验证
    """
    logger.info('Start to KFold stack attribution model')
    processor = AttributionProcessor()

    info_dict = prepare_info(opt.task_type, opt.mid_data_dir)

    kf = KFold(10, shuffle=True, random_state=789)

    stack_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, 'stack.json'))

    base_output_dir = opt.output_dir

    for i, (train_ids, dev_ids) in enumerate(kf.split(stack_raw_examples)):
        logger.info(f'Start to train the {i} fold')

        train_raw_examples = [stack_raw_examples[_idx] for _idx in train_ids]
        train_examples = processor.get_train_examples(train_raw_examples)

        dev_raw_examples = [stack_raw_examples[_idx] for _idx in dev_ids]
        dev_info = processor.get_dev_examples(dev_raw_examples)

        tmp_output_dir = os.path.join(base_output_dir, f'v{i}')

        opt.output_dir = tmp_output_dir

        train_base(opt, info_dict, train_examples, dev_info)


if __name__ == '__main__':
    args = TrainArgs().get_parser()

    assert args.mode in ['train', 'stack'], 'mode mismatch'
    assert args.task_type in ['trigger', 'role1', 'role2', 'attribution'], 'task mismatch'

    mode = 'stack' if args.mode == 'stack' else 'final'
    args.output_dir = os.path.join(args.output_dir, mode, args.task_type, args.bert_type)

    set_seed(seed=123)

    if args.task_type == 'trigger':
        if args.use_distant_trigger:
            logger.info('Use distant trigger in trigger detection')
            args.output_dir += '_distant_trigger'
    elif args.task_type in ['role1', 'role2']:
        if args.use_trigger_distance:
            logger.info('Use trigger distance in role detection')
            args.output_dir += '_distance'

    if args.attack_train != '':
        args.output_dir += f'_{args.attack_train}'

    if args.weight_decay:
        args.output_dir += '_wd'

    if args.enhance_data and args.task_type in ['trigger', 'role1', 'role2']:
        logger.info('Enhance data')
        args.output_dir += '_enhanced'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'stack':
        assert args.task_type in ['attribution'], 'Only support attribution task to stack'
        stacking(args)
    else:
        training(args)
