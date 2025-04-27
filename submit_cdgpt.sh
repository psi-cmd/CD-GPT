#!/bin/bash

########################
#  EXPORT ENVIRONMENT  #
########################
USERNAME=psi-cmd
NCORES=32
function rtlog(){
  curl "https://api.day.app/YOUR_APP_KEY/CD-GPT/$1"
}

########################
#  PARSE ARGUMENTS     #
########################
# 需要的命令行参数：
# 1: EPOCHS (训练轮数)
# 2: MEMORY_FACTOR (显存利用率，0-1之间)
# 3: LEARNING_RATE (学习率)
# 4: MODEL_SIZE (模型大小，例如small/base/large)
# 5: SEED (随机种子)

EPOCHS=${1:-20}
MEMORY_FACTOR=${2:-1}
LEARNING_RATE=${3:-1e-4}
MODEL_SIZE=${4:-1b}
SEED=${5:-42}
MAX_LENGTH=${6:-50}
NUM_WORKERS=${7:-8}

echo "Parameters:"
echo "EPOCHS: $EPOCHS"
echo "MEMORY_FACTOR: $MEMORY_FACTOR"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "MODEL_SIZE: $MODEL_SIZE"
echo "SEED: $SEED"
echo "MAX_LENGTH: $MAX_LENGTH"
echo "NUM_WORKERS: $NUM_WORKERS"

MODELNAME=cdgpt_${MODEL_SIZE}_mf${MEMORY_FACTOR}_lr${LEARNING_RATE}_e${EPOCHS}_s${SEED}_${RANDOM}

rtlog "Training+start"

########################
#  PREPARE ENVIRONMENT #
########################
MEMDIR=/dev/shm/
cp /staging/${USERNAME}/cdgpt_env.tar.gz $MEMDIR/
mkdir $MEMDIR/conda_env
ln -s $MEMDIR/conda_env ./conda_env
pigz -p $NCORES -dc $MEMDIR/cdgpt_env.tar.gz | tar xf - -C ./conda_env

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
  --learning_rate $LEARNING_RATE \
  --model_size $MODEL_SIZE \
  --seed $SEED \
  --max_length $MAX_LENGTH \
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