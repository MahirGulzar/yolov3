args=("$@")

# Checkpoint directory
CHKPT_DIR=checkpoints_$(date +%Y-%m-%d:%H:%M:%S)

# Validate and Configure the training
python validate_and_configure.py $CHKPT_DIR --data_path ${DATA_PATH}

# Yolov3.conv.81 required for finetuning 
if [[ ! -f yolov3.conv.81 ]]
  then
    wget ${args[0]}
  fi

# Start training
value=$(<log_file.txt)
CUDA_LAUNCH_BLOCKING=1 python train.py --cfg yolov3_custom.cfg --data custom.data  --epochs 1 --weights yolov3.conv.81 --multi-scale

cp -r runs ${value}/
cp -r weights ${value}/