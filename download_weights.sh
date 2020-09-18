args=("$@")

# Yolov3.conv.81 required for finetuning
if [[ ! -f yolov3.conv.81 ]]
  then
   {
    wget ${args[0]}
   } ||
   {
    cp ${args[0]} yolov3.conv.81
   }
   mkdir /home/weights
   cp yolov3.conv.81 /home/weights/yolov3.conv.81
  fi

