args=("$@")

# Checkpoint, Data directories and Model name
CHKPT_DIR=${CHKPT_PATH}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH}
VAL_DATA_PATH=${VAL_DATA_PATH}
MODEL_NAME=${MODEL_NAME}

# Validate and Configure the training
python validate_and_configure.py $CHKPT_DIR --train_data_path ${TRAIN_DATA_PATH} --val_data_path ${VAL_DATA_PATH}

# Yolov3.conv.81 required for finetuning 
if [[ ! -f yolov3.conv.81 ]]
  then
   {
    wget ${args[0]}
   } ||
   {
    cp ${args[0]} yolov3.conv.81
   }
  fi

# Start training
CUDA_LAUNCH_BLOCKING=1 python train.py --cfg yolov3_custom.cfg --data custom.data  --epochs 50 --weights yolov3.conv.81 --multi-scale

cp -r runs ${CHKPT_DIR}/
cp -r weights/best.pt ${CHKPT_DIR}/${MODEL_NAME}
