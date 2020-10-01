# coding=utf-8
"""
@author: Oscar
@license: (C) Copyright 2019-2022, ZJU.
@contact: 499616042@qq.com
@software: pycharm
@file: dev.py
@time: 2020/7/30 16:22
"""
import os
import logging
from torch.utils.data import DataLoader
from src_final.preprocess.processor import *
from src_final.utils.options import DevArgs
from src_final.utils.model_utils import build_model
from src_final.utils.dataset_utils import build_dataset
from src_final.utils.evaluator import trigger_evaluation, role1_evaluation, role2_evaluation, attribution_evaluation
from src_final.utils.functions_utils import get_model_path_list, load_model_and_parallel, \
    prepare_info, prepare_para_dict

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def evaluate(opt):
    processors = {'trigger': TriggerProcessor,
                  'role1': RoleProcessor,
                  'role2': RoleProcessor,
                  'attribution': AttributionProcessor}

    processor = processors[opt.task_type]()

    info_dict = prepare_info(opt.task_type, opt.mid_data_dir)

    feature_para, dataset_para, model_para = prepare_para_dict(opt, info_dict)

    dev_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, 'dev.json'))

    dev_examples, dev_callback_info = processor.get_dev_examples(dev_raw_examples)

    dev_features = convert_examples_to_features(opt.task_type, dev_examples, opt.bert_dir,
                                                opt.max_seq_len, **feature_para)

    logger.info(f'Build {len(dev_features)} dev features')

    dev_dataset = build_dataset(opt.task_type, dev_features,
                                mode='dev', **dataset_para)

    dev_loader = DataLoader(dev_dataset, batch_size=opt.eval_batch_size,
                            shuffle=False, num_workers=8)

    dev_info = (dev_loader, dev_callback_info)

    model = build_model(opt.task_type, opt.bert_dir, **model_para)

    model_path_list = get_model_path_list(opt.dev_dir)

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

        logger.info(f'In step {tmp_step}:\n{tmp_metric_str}')

        metric_str += f'In step {tmp_step}:\n{tmp_metric_str}\n\n'

        if tmp_f1 > max_f1:
            max_f1 = tmp_f1
            max_f1_step = tmp_step

    max_metric_str = f'Max f1 is: {max_f1}, in step {max_f1_step}\n'

    logger.info(max_metric_str)

    metric_str += max_metric_str + '\n'

    eval_save_path = os.path.join(opt.dev_dir, 'eval_metric.txt')

    with open(eval_save_path, 'a', encoding='utf-8') as f1:
        f1.write(metric_str)


if __name__ == '__main__':
    args = DevArgs().get_parser()

    dev_dir = args.dev_dir

    if '_distant_trigger' in dev_dir:
        args.use_distant_trigger = True

    if '_distance' in dev_dir:
        args.use_trigger_distance = True

    evaluate(args)
