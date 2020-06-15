args=("$@")

# Checkpoint directory
CHKPT_DIR=/home/data/unified_yolo_checkpoints/checkpoints_$(date +%Y-%m-%d:%H:%M:%S)
mkdir -p $CHKPT_DIR

# Validate and Configure the training
python validate_and_configure.py

# Yolov3.conv.81 required for finetuning 
if [[ ! -f yolov3.conv.81 ]]
  then
    wget ${args[0]}
  fi

# Start training
CUDA_LAUNCH_BLOCKING=1 python train.py --cfg yolov3_custom.cfg --data custom.data --epochs 1 --weights yolov3.conv.81 --multi-scale --use_seg_depth --rect --notest

cp -r runs $CHKPT_DIR/
cp -r weights $CHKPT_DIR/
