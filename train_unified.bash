
# Checkpoint directory
CHKPT_DIR=/home/data/unified_yolo_checkpoints/checkpoints_$(date +%Y-%m-%d:%H:%M:%S)
mkdir -p $CHKPT_DIR

# Validate and Configure the training
python validate_and_configure.py

# Start training
python train.py --cfg yolov3_custom.cfg --data custom.data --epochs 1 --weights yolov3.conv.81 --multi-scale --use_seg_depth --rect --notest

cp -r runs $CHKPT_DIR/
cp weights/last.pt $CHKPT_DIR/yolov3_unified_semseg.pt
