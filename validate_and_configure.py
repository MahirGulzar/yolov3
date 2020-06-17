import os
import glob
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = '/home/data/Train_Unified_Yolo'
SUB_PATHS = ['train', 'valid']
IMG_EXT_TYPES = ['png', 'jpg']
CLASSES = ['person', 'car']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chkpt_dir", type=str,
                        help="Checkpoint direcory for saving the weights")
    parser.add_argument("--data_path", type=str,
                        help='Main data folder containing different datasets in subfolders', default='/home/data/Train_Yolov3_Torch')                    
    args = parser.parse_args()

    DATA_PATH = args.data_path
    CHKPT_DIR = os.path.join(f'{DATA_PATH}_Checkpoints',args.chkpt_dir)

    os.makedirs(CHKPT_DIR,exist_ok=True)
    
    with open('log_file.txt','w+') as f:
        f.write(CHKPT_DIR)

    # Remove remnant files
    for f in glob.glob(os.path.join(CURRENT_DIR,'data','custom*')):
        os.remove(f)

    img_paths = []
    for ext in IMG_EXT_TYPES:
        img_paths.extend(glob.glob(os.path.join(
            DATA_PATH, '**', '*.%s' % ext), recursive=True))
    
    img_paths = [img for img in img_paths if 'depth' not in img and 'seg' not in img] #Exclude irrelevant images

    # Check if images exist
    if len(img_paths) == 0:
        raise ValueError(
            'No image files were found in the given path', os.path.join(DATA_PATH))

    # There must be a text file corresponding to each image
    for img_path in img_paths:
        dir_name = os.path.dirname(img_path)
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        label_file = os.path.join(dir_name, '%s.txt' % file_name)
        if not os.path.exists(label_file):
            raise ValueError(
                'No label file was found for the given image', img_path)

    # Populate train and test text files
    for ds in ['valid', 'train']:
        gt_file = os.path.join(CURRENT_DIR, "data", "custom_%s.txt" % ds)
        with open(gt_file, 'a+') as f:
            f.write('\n'.join(img_paths))

    # Create if data file does not exist
    # Override if necessary
    yolo_data_file = os.path.join(CURRENT_DIR, "data", "custom.data")
    yolo_names_file = os.path.join(CURRENT_DIR, "data", "custom.names")

    with open(yolo_data_file, 'w+') as f:
        f.write('classes=%d\n' % len(CLASSES))
        f.write('train=data/custom_train.txt\n')
        f.write('valid=data/custom_valid.txt\n')
        f.write('names=data/custom.names\n')
        f.write('backup=backup/\n')
        f.write('eval=coco')

    # Re/create the names file
    with open(yolo_names_file, 'w+') as f:
        f.write('\n'.join(CLASSES))

    # Check if the cfg file exists
    cfg_file = os.path.join(CURRENT_DIR, "cfg", "yolov3_custom.cfg")
    if not os.path.exists(cfg_file):
        raise ValueError('No cfg file was found for custom training', cfg_file)

    print("Configuration completed")