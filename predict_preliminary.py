#!/usr/bin/env python
# coding: utf-8

# In[23]:


import os
import logging
from torch.utils.data import DataLoader
from src_final.preprocess.processor import *
from src_final.utils.options import DevArgs
from src_final.utils.model_utils import build_model
from src_final.utils.dataset_utils import build_dataset
from src_final.utils.evaluator import *
from src_final.preprocess.convert_raw_data import *
import pickle as pk
from src_final.utils.functions_utils import *
import json
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


# In[2]:


# In[18]:


def trigger_evaluation_v1(model, dev_info, device, **kwargs):
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
    instances = []

    for tmp_pred, tmp_callback in zip(pred_logits, dev_callback_info):
        text, gt_triggers, distant_triggers = tmp_callback
        tmp_pred = tmp_pred[1:1+len(text)]

        pred_triggers = pointer_trigger_decode(tmp_pred, text, distant_triggers,
                                             start_threshold=start_threshold,
                                             end_threshold=end_threshold)

        if not len(pred_triggers):
            zero_pred += 1

        tmp_tp, tmp_fp, tmp_fn = calculate_metric(gt_triggers, pred_triggers)

        tp += tmp_tp
        fp += tmp_fp
        fn += tmp_fn

        # 不转成 str json 老是有 int64的 bug
        pred_triggers = sorted([(x[0], str(x[1])) for x in pred_triggers])
        gt_triggers = sorted([(x[0], str(x[1])) for x in gt_triggers])

#         if pred_triggers != gt_triggers:
        if True:
            instances.append({'text': text,
                              'pred': pred_triggers,
                              'gt': gt_triggers,
                              'distant': distant_triggers})

    p, r, f1 = get_p_r_f(tp, fp, fn)

    metric_str = f'In start threshold: {start_threshold}; end threshold: {end_threshold}\n'
    metric_str += f'[MIRCO] precision: {p:.4f}, recall: {r:.4f}, f1: {f1:.4f}\n'
    metric_str += f'Zero pred nums: {zero_pred}'

    return metric_str, f1, instances

def role_evaluation(model, dev_info, device, **kwargs):
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

    # role_types = ['object', 'subject', 'time', 'loc']
    role_types = ['object', 'subject']

    instances = []

    for tmp_pred, tmp_callback in zip(pred_logits, dev_callback_info):
        # TODO
        # 普通的 role 抽取
        text,trigger, gt_roles = tmp_callback
        tmp_pred = tmp_pred[1:len(text)+1]

        # mrc role 抽取
        # text, text_start, trigger, gt_roles = tmp_callback
        # tmp_pred = tmp_pred[text_start:text_start+len(text)]

        pred_obj = pointer_decode(tmp_pred[:, :2], text, start_threshold, end_threshold, True)
        pred_sub = pointer_decode(tmp_pred[:, 2:], text, start_threshold, end_threshold, True)
        # pred_time = pointer_decode(tmp_pred[:, 4:6], text, start_threshold, end_threshold)
        # pred_loc = pointer_decode(tmp_pred[:, 6:], text, start_threshold, end_threshold)

        tmp_metric = np.zeros([2, 3])

        # pred_roles = {'subject': pred_sub,
        #               'object': pred_obj,
        #               'time': pred_time,
        #               'loc': pred_loc}

        pred_roles = {'subject': pred_sub, 'object': pred_obj}

        wrong_pred = []

        for idx, _role in enumerate(role_types):
            tmp_metric[idx] += calculate_metric(gt_roles[_role], pred_roles[_role])

#             if sorted(gt_roles[_role]) != sorted(pred_roles[_role]):
            if True:
                wrong_pred.append({
                    'role': _role,
                    'pred': pred_roles[_role],
                    'gt': gt_roles[_role]
                })

        role_metric += tmp_metric

        if len(wrong_pred):
            instances.append({'text': text,
                              'trigger': trigger,
                              'wrong_pred': wrong_pred})

    metric_str = f'In start threshold: {start_threshold}; end threshold: {end_threshold}\n'

    for idx, _role in enumerate(role_types):

        temp_metric = get_p_r_f(role_metric[idx][0], role_metric[idx][1], role_metric[idx][2])

        mirco_metrics += temp_metric * type_weight[_role]

        metric_str += '[%s] precision: %.4f, recall: %.4f, f1: %.4f.\n' %                (_role, temp_metric[0], temp_metric[1], temp_metric[2])


    metric_str += f'[MIRCO] precision: {mirco_metrics[0]:.4f}, '                   f'recall: {mirco_metrics[1]:.4f}, f1: {mirco_metrics[2]:.4f}'

    return metric_str, mirco_metrics[2], instances

def attribution_evaluation(model, dev_info, device, **kwargs):
    dev_loader, dev_callback_info = dev_info

    polarity_logits, tense_logits = None, None

    # tense_logits = None

    for tmp_pred in get_base_out(model, dev_loader, device, 'attribution'):
        tmp_polarity_logits, tmp_tense_logits = tmp_pred
        # tmp_tense_logits = tmp_pred[0]

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

    instances = []

    for tmp_pred_tense, tmp_pred_polarity, tmp_callback in             zip(tense_logits, polarity_logits, dev_callback_info):
        text, trigger, gt_attributions = tmp_callback

        pred_polarity = id2polarity[np.argmax(tmp_pred_polarity)]

        pred_tense = id2tense[np.argmax(tmp_pred_tense)]

        # if pred_tense == '将来':
        #     pred_polarity = '可能'
        # else:
        #     pred_polarity = '肯定'

        wrong_pred = []

        if pred_tense == gt_attributions[0]:
            tense_acc += 1
        else:
            wrong_pred.append({'attribution': 'tense',
                               'pred': pred_tense,
                               'gt': gt_attributions[0]})


        if pred_polarity == gt_attributions[1]:
            polarity_acc += 1
        if True:
            wrong_pred.append({'attribution': 'polarity',
                               'pred': pred_polarity,
                               'gt': gt_attributions[1]})

        counts += 1

#         if len(wrong_pred):
        if True:
            instances.append({'text': text,
                              'trigger': trigger,
                              'wrong_pred': wrong_pred})

    metric_str = ''

    polarity_acc /= counts
    tense_acc /= counts


    metric_str += f'[ACC] polarity: {polarity_acc:.4f}, tense: {tense_acc:.4f}'

    return metric_str, (polarity_acc+tense_acc)/2, instances,polarity_logits,tense_logits

def evaluate(opt,dev_file='dev.json',force_one=True,model_name=None,dev_raw_examples=None):
    processors = {'trigger': TriggerProcessor,
                  'role1': RoleProcessor,'role2': RoleProcessor,
                  'attribution': AttributionProcessor}

    processor = processors[opt.task_type]()

    triggers_dict = None
    polarity2id, tense2id = None, None
    polarity_prior, tense_prior = None, None

    if opt.task_type == 'trigger':
        with open(os.path.join(opt.mid_data_dir, f'triggers_dict.json'), encoding='utf-8') as f:
            triggers_dict = json.load(f)
    elif opt.task_type == 'attribution':
        with open(os.path.join(opt.mid_data_dir, f'polarity2id.json'), encoding='utf-8') as f:
            polarity2id = json.load(f)
        with open(os.path.join(opt.mid_data_dir, f'tense2id.json'), encoding='utf-8') as f:
            tense2id = json.load(f)

        if opt.use_polarity_prior:
            polarity_prior = polarity2id['prob']

        if opt.use_tense_prior:
            tense_prior = tense2id['prob']

        polarity2id = polarity2id['map']
        tense2id = tense2id['map']
    if dev_raw_examples is None:
        dev_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir,dev_file))
    dev_examples, dev_callback_info = processor.get_dev_examples(dev_raw_examples)

    dev_features = convert_examples_to_features(opt.task_type, dev_examples, opt.bert_dir,
                                                opt.max_seq_len, triggers_dict=triggers_dict,
                                                mask_prob=0., polarity2id=polarity2id,
                                                tense2id=tense2id)
    logger.info(f'Build {len(dev_features)} dev features')

    dev_dataset = build_dataset(opt.task_type, dev_features,
                                mode='dev',
                                use_distant_trigger=opt.use_distant_trigger,
                                use_trigger_distance=opt.use_trigger_distance,
                                polarity_prior=polarity_prior,
                                tense_prior=tense_prior)
    dev_loader = DataLoader(dev_dataset, batch_size=opt.eval_batch_size,
                            shuffle=False, num_workers=8)

    dev_info = (dev_loader, dev_callback_info)
    model = build_model(opt.task_type, opt.bert_dir,
                        use_distant_trigger=opt.use_distant_trigger,
                        use_trigger_distance=opt.use_trigger_distance)
    if model_name is None:
        model_path_list = get_model_path_list(opt.dev_dir)
    else:
        model_path_list=[os.path.join(model_name,'model.pt')]
    metric_str = ''

    max_f1 = 0.
    max_f1_step = 0

    for idx, model_path in enumerate(model_path_list):
        tmp_step = model_path.split('/')[-2].split('-')[-1]

        model, device = load_model_and_parallel(model, opt.gpu_ids[0],
                                                ckpt_path=model_path)

        if opt.task_type == 'trigger':
            # tmp_metric_str, tmp_f1, tmp_instances = trigger_evaluation(model, dev_info, device)

            tmp_metric_str, tmp_f1, tmp_instances = trigger_evaluation_v1(model, dev_info, device,
                                                                          start_threshold=opt.start_threshold,
                                                                          end_threshold=opt.end_threshold,force_one=force_one)
        elif opt.task_type == 'role1':
            tmp_metric_str, tmp_f1, tmp_instances = role_evaluation(model, dev_info, device,
                                                                       start_threshold=opt.start_threshold,
                                                                       end_threshold=opt.end_threshold)
        else:
            tmp_metric_str, tmp_f1, tmp_instances,polarity_logits,tense_logits = attribution_evaluation(model, dev_info, device,
                                                                           polarity2id=polarity2id,
                                                                           tense2id=tense2id)

        logger.info(f'In step {tmp_step}:\n{tmp_metric_str}')

        metric_str += f'In step {tmp_step}:\n{tmp_metric_str}\n\n'

        tmp_model_dir = os.path.split(model_path)[0]
        if opt.task_type == 'attribution':
            pk.dump([polarity_logits,tense_logits],open(os.path.join(tmp_model_dir, 'logits.json'),'wb'))
        with open(os.path.join(tmp_model_dir, 'instance.json'), 'w', encoding='utf-8') as f:
            json.dump(tmp_instances, f, ensure_ascii=False, indent=2)

        if tmp_f1 > max_f1:
            max_f1 = tmp_f1
            max_f1_step = tmp_step

    max_metric_str = f'Max f1 is: {max_f1}, in step {max_f1_step}\n'

    logger.info(max_metric_str)

    metric_str += max_metric_str + '\n'

    eval_save_path = os.path.join(opt.dev_dir, 'eval_metric.txt')

    with open(eval_save_path, 'a', encoding='utf-8') as f1:
        f1.write(metric_str)


# In[21]:


args = DevArgs().get_parser()

args.bert_type='roberta_wwm'
args.bert_dir="../bert/torch_roberta_wwm/"
args.raw_data_dir="./data/final/raw_data/"
args.mid_data_dir="./data/final/mid_data/"
args.task_type='trigger'
args.gpu_ids='1,3'
args.eval_batch_size=64
args.max_seq_len=320
args.start_threshold=0.6
args.end_threshold=0.45
# args.start_threshold=0.65
# args.end_threshold=0.6
args.dev_dir=args.dev_dir_trigger
# args.dev_dir="./out/final/role1/roberta_wwm_distance_pgd_enhanced/"
# args.dev_dir="./out/final/attribution/roberta_wwm_pgd/"
dev_dir=args.dev_dir


if '_distant_trigger' in dev_dir:
    args.use_distant_trigger = True

if '_distance' in dev_dir:
    args.use_trigger_distance = True

if '_polarity_prior' in dev_dir:
    args.use_polarity_prior = True

if '_tense_prior' in dev_dir:
    args.use_tense_prior = True
evaluate(args,'preliminary_stack.json',force_one=False)


# In[19]:


args.task_type='role1'
args.gpu_ids='1,3'
args.eval_batch_size=64
args.max_seq_len=512
args.start_threshold=0.65
args.end_threshold=0.6
args.dev_dir=args.dev_dir_role
dev_dir=args.dev_dir

if '_distant_trigger' in dev_dir:
    args.use_distant_trigger = True

if '_distance' in dev_dir:
    args.use_trigger_distance = True

if '_polarity_prior' in dev_dir:
    args.use_polarity_prior = True

if '_tense_prior' in dev_dir:
    args.use_tense_prior = True
info=evaluate(args,'preliminary_stack.json')

dev_trigger_dir=args.dev_dir_trigger
dev_role_dir=args.dev_dir_role

# In[26]:
trigger_check_file=list(sorted([e for e in os.listdir(dev_trigger_dir) if 'checkpoint-' in e],key=lambda x:int(x.split("-")[1])))[-1]
role_check_file=list(sorted([e for e in os.listdir(dev_role_dir) if 'checkpoint-' in e],key=lambda x:int(x.split("-")[1])))[-1]

raw_dir=args.raw_data_dir
new_stack = load_examples(os.path.join(raw_dir, 'preliminary_stack.json'))
trigger_pred_info=load_examples(os.path.join(dev_trigger_dir,trigger_check_file,'instance.json'))
for idx in tqdm(range(len(new_stack))):
    pred_triggers=trigger_pred_info[idx]['pred']
    new_stack[idx]['pred_triggers']=[{'text':e[0],'length':len(e[0]),'offset':int(e[1])} for e in pred_triggers]
save_info("./data/final/raw_data/",new_stack,'preliminary_data_pred_trigger')


# In[27]:



role_pred_info=load_examples(os.path.join(dev_role_dir,role_check_file,'instance.json'))
text_idx=0
for idx in tqdm(range(len(role_pred_info))):
    text=role_pred_info[idx]['text']
    while(text!=new_stack[text_idx]['sentence']):
        text_idx+=1
    assert text==new_stack[text_idx]['sentence']
    trigger=role_pred_info[idx]['trigger']
    new_spo=None
    for e in new_stack[text_idx]['events']:
        if e['trigger']['text']==trigger:
            new_spo=copy.deepcopy(e)
            new_spo['arguments'].clear()
            break
    if new_spo is not None:
        for e in role_pred_info[idx]['wrong_pred']:
            role_name=e['role']
            for pred in e['pred']:
                role={'role':role_name,'text':pred[0],'length':len(pred[0]),'offset':int(pred[1])}
                new_spo['arguments'].append(role)
        if 'pred_events' in new_stack[text_idx].keys():
            new_stack[text_idx]['pred_events'].append(new_spo)
        else:
            new_stack[text_idx]['pred_events']=[new_spo]  
save_info("./data/final/raw_data/",new_stack,'preliminary_data_pred_trigger_and_role')


