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
    parser.add_argument("--dataset", type=str,
                        help="Dataset name for cross-validation. Leave empty for training on all of the data", default='ALL')
    args = parser.parse_args()
    if args.dataset == 'ALL':
        dataset_folders = [os.path.join(DATA_PATH,ds_name) for ds_name in os.listdir(DATA_PATH) if 'check' not in ds_name] # exclude checkpoints folder
    else:
        dataset_folders = [os.path.join(DATA_PATH,args.dataset)]
    
    # Iterate through separate dataset folders
    for dataset in dataset_folders:
        # Iterate through training and test datasets
        for sub_path in SUB_PATHS:
            img_paths = []

            for ext in IMG_EXT_TYPES:
                img_paths.extend(glob.glob(os.path.join(dataset, sub_path, "*.%s") % ext))
            # Exclude segmentation and depth ground truths
            img_paths = sorted(
                img for img in img_paths if 'depth' not in img and 'seg' not in img)

            # Check if images exist
            if len(img_paths) == 0:
                raise ValueError(
                    'No image files were found in the given path', os.path.join(dataset,sub_path))

            # There must be a text file corresponding to each image
            for img_path in img_paths:
                dir_name = os.path.dirname(img_path)
                file_name = os.path.splitext(os.path.basename(img_path))[0]
                label_file = os.path.join(dir_name, '%s.txt' % file_name)
                if not os.path.exists(label_file):
                    raise ValueError(
                        'No label file was found for the given image', img_path)

            # Populate train and test text files
            gt_file = os.path.join(CURRENT_DIR, "data", "custom_%s.txt" % sub_path)
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