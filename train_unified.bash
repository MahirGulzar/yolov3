#!/usr/bin/env bash
set -e

# Checkpoint directory
mkdir -p $CHECKPOINT_DIR
data_path=$1
shift
model_name=$1
shift

# Validate and Configure the training
python validate_and_configure.py $CHECKPOINT_DIR --data_path $data_path


# in $@, there should be:
# --seg to train on segmentation
# --depth to train on depth
# --seg_classes to set how many classes segmentation has

# Start training
python train.py --cfg yolov3_custom.cfg --data custom.data --epochs 50 --weights yolov3.conv.81 --multi-scale --rect --notest $@

cp -r runs $CHECKPOINT_DIR/
cp weights/last.pt $CHECKPOINT_DIR/$model_name.pt

echo "segment_images --segmenter yolo_generic --model_path $model_name.pt --source_labelspace milrem \$1 \$2" > $CHECKPOINT_DIR/inference.sh
