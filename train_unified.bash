#!/usr/bin/env bash
set -e

# Checkpoint directory
CHKPT_DIR=/home/data/unified_yolo_checkpoints/checkpoints_$(date +%Y-%m-%d:%H:%M:%S)
mkdir -p /home/checkpoint_dir
data_path=$1
model_name=$2

# Validate and Configure the training
python validate_and_configure.py /home/checkpoint_dir --data_path $1 

shift

# in $@, there should be:
# --seg to train on segmentation
# --depth to train on depth
# --seg_classes to set how many classes segmentation has

# Start training
python train.py --cfg yolov3_custom.cfg --data custom.data --epochs 2 --weights yolov3.conv.81 --multi-scale --rect --notest $@

cp -r runs /home/checkpoint_dir/
cp weights/last.pt /home/checkpoint_dir/$model_name.pt

echo 'segment_images --segmenter yolo_generic --model_path yolov3_unified_semseg.pt --source_labelspace milrem $1 $2' > /home/checkpoint_dir/inference.sh
