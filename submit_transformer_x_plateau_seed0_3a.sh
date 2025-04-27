#!/bin/bash

########################
#  EXPORT ENVIRONMENT  #
########################
USERNAME=lliu466
NCORES=32
function rtlog(){
  curl "https://api.day.app/8cyWQUgKWTiAyZVRBMdBUY/Molecular+Glue/$1"
}

bash -e rev.sh &

MEMDIR=/dev/shm/
rtlog "Training+start"
cp /staging/${USERNAME}/conda_env.tar.gz $MEMDIR/
mkdir $MEMDIR/conda_env
ln -s $MEMDIR/conda_env ./conda_env
pigz -p $NCORES -dc $MEMDIR/conda_env.tar.gz | tar xf - -C ./conda_env

rtlog "Env+Decompressed"

export PATH=$PWD/conda_env/bin:$PATH
source conda_env/bin/activate
python --version

# data
DATA_FILE_NAME="training.tar.gz"
echo "Extracting $DATA_FILE_NAME"
date;
cp /staging/${USERNAME}/$DATA_FILE_NAME ./
export VOXEL_DIR=$MEMDIR/voxels
pigz -p $NCORES -dc $DATA_FILE_NAME | tar xf - -C $MEMDIR
ls $VOXEL_DIR/*.pkl | wc -l
ls $VOXEL_DIR/*.pvar | wc -l
date;
rtlog "Data+Extracted"

#################
#  EXPORT PATH  #
#################
PROJ_HOME=.
SCRIPT_HOME=${PROJ_HOME}/src

EPOCHS=70
GRAPHCLAN=./ClanGraph_90_trial_df.pkl
MODELREPO=.
TASK=HeavyAtomsite


#######################
#  TRAIN TRANSFORMER  #
#######################
MODELTYPE=transformer
NBLOCKS=$1
echo $NBLOCKS
SEED=0
MODELNAME=3a_${RANDOM}_${TASK}_${MODELTYPE}_${NBLOCKS}_seed${SEED}

echo "Training models"

date;
ln -s /staging/${USERNAME}/output ${MODELREPO}/result
mkdir -p ${MODELREPO}/result/${MODELNAME}
ln -s /staging/${USERNAME}/output/lightning_logs ${MODELREPO}/lightning_logs
python -u ./src/models/train_lightning.py \
  --n_class=1 \
  --modeltype=${MODELTYPE} \
  --seed=${SEED} \
  --task=${TASK} \
  --modelname=${MODELNAME} \
  --n_blocks=${NBLOCKS} \
  --optimizer=AdamW \
  --scheduler=plateau \
  --lr=0.001 \
  --steps=${EPOCHS} \
  --n_structs=5000 \
  --n_resamples=1000 \
  --weight_decay=0 \
  --save_every=10 \
  --earlystop \
  --patience=${EPOCHS} \
  --graphclan=${GRAPHCLAN} \
  --datafolder=${VOXEL_DIR} \
  --outfolder=${MODELREPO}/result/${MODELNAME} \
  --verbose \
  --use_gpu > ${MODELREPO}/result/${MODELNAME}/${MODELNAME}.LOG
date;

mv _condor_stderr _condor_stdout ./result

sleep 600
##################
#  FETCH OUTPUT  #
##################
rm -r $MODELREPO/*

##############
#  CLEAN UP  #
##############
