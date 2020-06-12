args=("$@")

# Validate and Configure the training
python validate_and_configure.py

# Yolov3.conv.81 required for finetuning 
if [[ ! -f yolov3.conv.81 ]] then
    wget ${args[0]} 
fi

# Start training
CUDA_LAUNCH_BLOCKING=1 python train.py --cfg yolov3_custom.cfg --data custom.data  --epochs 100 --weights yolov3.conv.81 --multi-scale
