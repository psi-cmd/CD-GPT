#!/bin/bash

EPOCHS=${1:-20}
MEMORY_FACTOR=${2:-0.8}
LEARNING_RATE=${3:-1e-4}
MODEL_SIZE=${4:-1b}
SEED=${5:-42}
MAX_LENGTH=${6:-50}
NUM_WORKERS=${7:-8}

python -u finetune_CDGPT.py \
  --epochs $EPOCHS \
  --batch_size 0 \
  --memory_factor $MEMORY_FACTOR \
  --learning_rate $LEARNING_RATE \
  --model_size $MODEL_SIZE \
  --seed $SEED \
  --max_length $MAX_LENGTH \
  --num_workers $NUM_WORKERS \
  --checkpoint_dir ./checkpoints

