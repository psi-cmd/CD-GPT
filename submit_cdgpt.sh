#!/bin/bash

########################
#  EXPORT ENVIRONMENT  #
########################
USERNAME=lliu466
NCORES=32
function rtlog(){
  curl "https://api.day.app/8cyWQUgKWTiAyZVRBMdBUY/CD-GPT/$1"
}

########################
#  PARSE ARGUMENTS     #
########################
# 需要的命令行参数：
# 1: EPOCHS (训练轮数)
# 2: MEMORY_FACTOR (显存利用率，0-1之间)
# 3: LR_LORA (LoRA学习率)
# 4: LR_ADAPTER (Adapter学习率) 
# 5: PEFT_RANK (LoRA秩)
# 6: MODEL_SIZE (模型大小，例如small/base/large)
# 7: SEED (随机种子)
# 8: NUM_WORKERS (数据加载线程数)

EPOCHS=${1:-20}
MEMORY_FACTOR=${2:-1}
LR_LORA=${3:-1e-4}
LR_ADAPTER=${4:-2e-5}
PEFT_RANK=${5:-16}
MODEL_SIZE=${6:-1b}
SEED=${7:-42}
NUM_WORKERS=${8:-8}

echo "Parameters:"
echo "EPOCHS: $EPOCHS"
echo "MEMORY_FACTOR: $MEMORY_FACTOR"
echo "LR_LORA: $LR_LORA"
echo "LR_ADAPTER: $LR_ADAPTER"
echo "PEFT_RANK: $PEFT_RANK"
echo "MODEL_SIZE: $MODEL_SIZE"
echo "SEED: $SEED"
echo "NUM_WORKERS: $NUM_WORKERS"

MODELNAME=cdgpt_${MODEL_SIZE}_mf${MEMORY_FACTOR}_lora${LR_LORA}_adp${LR_ADAPTER}_r${PEFT_RANK}_e${EPOCHS}_s${SEED}_${RANDOM}

rtlog "Training+start"

########################
#  PREPARE ENVIRONMENT #
########################
MEMDIR=/dev/shm/
cp /staging/${USERNAME}/cdgpt_env.tar.gz $MEMDIR/
mkdir -p $MEMDIR/env_extract
pigz -p $NCORES -dc $MEMDIR/cdgpt_env.tar.gz | tar xf - -C $MEMDIR/env_extract
ln -s $MEMDIR/env_extract/cdgpt ./conda_env

rtlog "Env+Decompressed"

export PATH=$PWD/conda_env/bin:$PATH
source conda_env/bin/activate
python --version

########################
#  PREPARE DATA        #
########################
# 直接使用/staging/${USERNAME}/data目录
echo "Using data from /staging/${USERNAME}/data"
date;
export DATA_DIR=/staging/${USERNAME}/data
ls $DATA_DIR | wc -l
date;
rtlog "Data+Ready"

########################
#  EXPORT PATH         #
########################
PROJ_HOME=.
SCRIPT_HOME=${PROJ_HOME}

########################
#  TRAIN CD-GPT        #
########################
echo "Training CD-GPT model: $MODELNAME"

date;
# 创建结果目录并建立链接
mkdir -p result/${MODELNAME}
# 直接链接到/staging/${USERNAME}下已有的目录
ln -s /staging/${USERNAME}/lightning_logs ./lightning_logs
ln -s /staging/${USERNAME}/checkpoints ./checkpoints

# 开始训练，使用命令行参数
python -u finetune_CDGPT.py \
  --epochs $EPOCHS \
  --batch_size 0 \
  --memory_factor $MEMORY_FACTOR \
  --lr_lora $LR_LORA \
  --lr_adapter $LR_ADAPTER \
  --peft_rank $PEFT_RANK \
  --model_size $MODEL_SIZE \
  --seed $SEED \
  --num_workers $NUM_WORKERS \
  --data_dir $DATA_DIR \
  --output_dir ./result/${MODELNAME} \
  --checkpoint_dir ./checkpoints > ./result/${MODELNAME}/${MODELNAME}.LOG

date;
rtlog "Training+Completed"

# 将输出文件移到结果目录
mv _condor_stderr _condor_stdout ./result/${MODELNAME}/

# 将结果复制回/staging
mkdir -p /staging/${USERNAME}/results/${MODELNAME}
cp -r result/${MODELNAME}/* /staging/${USERNAME}/results/${MODELNAME}/

# 睡眠一段时间确保所有输出都已写入
sleep 60

########################
#  CLEAN UP            #
########################
rm -r $MEMDIR/conda_env 