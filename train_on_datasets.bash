# Validate and Configure the training
python validate_and_configure.py

# Yolov3.conv.81 required for finetuning 
if [[ ! -f yolov3.conv.81 ]]
  then
    megadl 'https://mega.nz/#!qk9mgYwJ!5Ndff-Z9ZdKifZpY9Qu9WeHQVBeDiCwZbzSn_fJhjkc'
  fi

# Start training
python train.py --data custom.data --weights yolov3.conv.81  --cfg yolov3_custom.cfg --epochs 100 --multi-scale