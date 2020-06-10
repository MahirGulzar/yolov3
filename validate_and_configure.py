import os
import glob

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = '/home/data/yolo'
SUB_PATHS = ['train', 'test']
IMG_EXT_TYPES = ['png', 'jpg']
CLASSES = ['person', 'car']

if __name__ == "__main__":
    # Iterate through training and test datasets
    for sub_path in SUB_PATHS:
        img_paths = []
        for ext in IMG_EXT_TYPES:
            img_paths.extend(glob.glob(os.path.join(DATA_PATH, sub_path, "*.%s")%ext))
        # Exclude segmentation and depth ground truths
        img_paths = sorted(img for img in img_paths if 'depth' not in img and 'seg' not in img)

        # Check if images exist
        if len(img_paths) == 0:
            raise ValueError('No image files were found in the given path', DATA_PATH)
        
        # There must be a text file corresponding to each image
        for img_path in img_paths:
            dir_name = os.path.dirname(img_path)
            file_name = os.path.splitext(os.path.basename(img_path))[0]
            label_file = os.path.join(dir_name, '%s.txt'%file_name)
            if not os.path.exists(label_file):
                raise ValueError('No label file was found for the given image', img_path)
        
        # Populate train and test text files
        gt_file = os.path.join(CURRENT_DIR,"data","custom_%s.txt"%sub_path)
        with open(gt_file, 'w+') as f:
            f.write('\n'.join(img_paths))
    
    # Create if data file does not exist
    # Override if necessary
    yolo_data_file = os.path.join(CURRENT_DIR,"data","custom.data")
    yolo_names_file = os.path.join(CURRENT_DIR,"data","custom.names")

    need_to_write = False
    if not os.path.exists(yolo_data_file):
        need_to_write = True
    else:
        with open(yolo_data_file,'r') as f:
            first_line = f.readline().strip()
            if first_line == '':
                need_to_write = True
            else:
                try:
                    class_number = int(first_line.split('=')[-1])
                    if class_number != len(CLASSES):
                        need_to_write = True
                except ValueError:
                    need_to_write = True

    if need_to_write:
        # Re/create the data file
        with open(yolo_data_file,'w+') as f:
            f.write('classes=%d\n'%len(CLASSES))
            f.write('train=data/custom_train.txt\n')
            f.write('valid=data/custom_valid.txt\n')
            f.write('names=data/custom.names\n')
            f.write('backup=backup/\n')
            f.write('eval=coco')
        
    # Re/create the names file
    with open(yolo_names_file, 'w+') as f:
        f.write('\n'.join(CLASSES))
    
    # Check if the cfg file exists
    cfg_file = os.path.join(CURRENT_DIR,"cfg","yolov3_custom.cfg")
    if not os.path.exists(cfg_file):
        raise ValueError('No cfg file was found for custom training', cfg_file)

    print("Configuration completed")