#!/usr/bin/env bash
set -e

# Checkpoint directory
CHKPT_DIR=/home/data/unified_yolo_checkpoints/checkpoints_$(date +%Y-%m-%d:%H:%M:%S)
mkdir -p $CHKPT_DIR

# Validate and Configure the training
python validate_and_configure.py $CHKPT_DIR --data_path $1 

shift

# in $@, there should be:
# --seg to train on segmentation
# --depth to train on depth
# --seg_classes to set how many classes segmentation has

# Start training
python train.py --cfg yolov3_custom.cfg --data custom.data --epochs 40 --weights yolov3.conv.81 --multi-scale --rect --notest $@

cp -r runs $CHKPT_DIR/
cp weights/last.pt $CHKPT_DIR/yolov3_unified_semseg.pt

echo 'segment_images --segmenter yolo_generic --model_path yolov3_unified_semseg.pt --source_labelspace milrem $1 $2' > $CHKPT_DIR/inference.sh
