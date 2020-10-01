#!/usr/bin/env bash

export BERT_TYPE="roberta_wwm"  # roberta_wwm / ernie_1  / uer_large

export BERT_DIR="../bert/torch_$BERT_TYPE"
export RAW_DATA_DIR="./data/final/raw_data"
export MID_DATA_DIR="./data/final/mid_data"
export AUX_DATA_DIR="./data/final/preliminary_clean"

export TASK_TYPE="trigger"

export MODE="dev"
export GPU_IDS="0"

python dev.py \
--gpu_ids=$GPU_IDS \
--mode=$MODE \
--raw_data_dir=$RAW_DATA_DIR \
--mid_data_dir=$MID_DATA_DIR \
--aux_data_dir=$AUX_DATA_DIR \
--bert_dir=$BERT_DIR \
--bert_type=$BERT_TYPE \
--task_type=$TASK_TYPE \
--eval_batch_size=256 \
--max_seq_len=512 \
--start_threshold=0.5 \
--end_threshold=0.5 \
--dev_dir="./out/final/${TASK_TYPE}/roberta_wwm_distant_trigger_pgd_enhanced"


