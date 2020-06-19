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
    parser.add_argument("--train_data_path", type=str,
                        help='Main data folder containing different datasets in subfolders', default='/home/data/Train_Darknet')
    parser.add_argument("--val_data_path", type=str,
                        help='Main data folder containing different datasets in subfolders', default='/home/data/Train_Darknet')
    args = parser.parse_args()

    TRAIN_DATA_PATH = args.train_data_path
    VAL_DATA_PATH = args.val_data_path
    PATHS = [TRAIN_DATA_PATH,VAL_DATA_PATH]
    CHKPT_DIR = os.path.abspath(args.chkpt_dir)
    
    os.makedirs(CHKPT_DIR,exist_ok=True)

    # Remove remnant files
    for f in glob.glob(os.path.join(CURRENT_DIR,'data','custom*')):
        os.remove(f)

    img_paths = [[],[]]

    for i,path in enumerate(PATHS):
        for ext in IMG_EXT_TYPES:
            img_paths[i].extend(glob.glob(os.path.join(
                path, '**', '*.%s' % ext), recursive=True))
        img_paths[i] = [img for img in img_paths[i] if 'depth' not in img and 'seg' not in img] #Exclude irrelevant images

        # Check if images exist
        if len(img_paths[i]) == 0:
            raise ValueError(
                'No image files were found in the given path', os.path.join(path))

        # There must be a text file corresponding to each image
        for img_path in img_paths[i]:
            dir_name = os.path.dirname(img_path)
            file_name = os.path.splitext(os.path.basename(img_path))[0]
            label_file = os.path.join(dir_name, '%s.txt' % file_name)
            if not os.path.exists(label_file):
                raise ValueError(
                    'No label file was found for the given image', img_path)

    # Populate train and test text files
    for ds, imgs in zip(['train', 'valid'], img_paths):
        gt_file = os.path.join(CURRENT_DIR, "data", "custom_%s.txt" % ds)
        with open(gt_file, 'a+') as f:
            f.write('\n'.join(imgs))

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