import os
import cv2
import sys
from pathlib import Path
import shutil
import cv2
import re
import random
import numpy as np
import argparse
from collections import defaultdict, Counter
root_path = '../src'
if not root_path in sys.path:
    sys.path.insert(0, root_path)

from utils.misc import load_file, save_file, listdir

ALL_DATASETS = [
    'Grating_A6',
    'Grating_SON1',
    'LASEROPTIK_SON1', 
    'THALES_SESO_10000',
    'THALES_SESO_A',
    'LASEROPTIK',
    'LASEROPTIK_10000'
    ]

DATA_SUBSETS = {'Grating': ['Grating_A6','Grating_SON1'],
                'Laseroptik': ['LASEROPTIK_SON1', 'LASEROPTIK', 'LASEROPTIK_10000'],
                'Thales': ['THALES_SESO_10000','THALES_SESO_A'],
               }

# First damaged shot index in each location
FIRST_DAMAGED_SHOTS = {
'Grating_A6':{
    (1, 1): (None, None),
    (1, 2): (6, 6),
    (2, 1): (6, 11),
    (2, 2): (6, 5),
    (3, 1): (6, 51), # it is not clear if this is the real damage
    (3, 2): (6, 62),
    (4, 1): (6, 10),
    (4, 2): (6, 2),
    (5, 1): (6, 4),
    (5, 2): (6, 10),
    },
'Grating_SON1': {
    (1, 1): (None, None),
    (1, 2): (1, 1),
    (1, 3): (None, None),
    (2, 1): (None, None),
    (2, 2): (None, None),
    (2, 3): (None, None),
    (3, 1): (None, None),
    (3, 2): (None, None),
    (3, 3): (None, None),
    (4, 1): (None, None),
    (4, 2): (None, None),
    (4, 3): (None, None),
    (5, 1): (1, 53),
    (5, 2): (1, 11),
    (5, 3): (3, 1),
    (6, 1): (1, 2),
    (6, 2): (1, 2),  
    (6, 3): (1, 4),
    },
'LASEROPTIK': {
    (1, 1): (6, 2),   # todo: (6,2)
    (1, 2): (6, 1),         # todo: +1 to shot no.
    (2, 1): (6, 3),         # todo: +1 to shot no.
    (2, 2): (6, 7),         # todo: +1 to shot no.
    (3, 1): (6, 1),         # todo: +1 to shot no.
    (3, 2): (6, 1),         # todo: +1 to shot no.
    (4, 1): (6, 3),         # todo: +1 to shot no.
    (4, 2): (6, 1),         # todo: +1 to shot no.
    (5, 1): (None, None),   # todo: only 1 burst - remove R5C1?
    (5, 2): (6,4)           # todo: +1 to shot no.
    },
'LASEROPTIK_SON1': {
    (1, 1): (None, None),
    (1, 2): (None, None),
    (1, 3): (None, None),
    (2, 1): (None, None),
    (2, 2): (None, None),
    (2, 3): (None, None),
    (3, 1): (None, None),
    (3, 2): (None, None),
    (3, 3): (None, None),
    (4, 1): (7, 9),        # todo: (7,9)
    (4, 2): (5, 82),
    (4, 3): (7, 6),
    (5, 1): (1, 70),
    (5, 2): (2, 4),        # todo: (2,4)
    (5, 3): (1, 92),
    (6, 1): (1, 6),
    (6, 2): (1, 2),        # todo: (1,2)
    (6, 3): (1, 6),
    },
'LASEROPTIK_10000': {
    (1, 1): (None, None),   # todo: remove bursts 104 and 105 from training?
    (1, 2): (None, None)    # todo: remove bursts 104 and 105 from training?
    },
'THALES_SESO_10000': {
    (1, 1): (None, None),
    (1, 2): (None, None),   # todo: remove burst 104 from training?
    },
'THALES_SESO_A': {
    (1, 1): (None, None),
    (1, 2): (None, None),
    (2, 1): (8, 3),
    (2, 2): (7, 7),
    (3, 1): (7, 72),
    (3, 2): (7, 45),    # R3C2: B7-S43 to B7-S44 -> why black spot gone? Maybe not a real damage
    (4, 1): (7, 11),
    (4, 2): (8, 3),
    (5, 1): (8, 3),
    (5, 2): (7, 67),
    },
}


def compute_overall_mean_std(data_dir):
    img_means, img_stds = [], []
    for root, dirs, files in os.walk(data_dir):
        for img_file in files:
            if img_file.lower().endswith(('.png')):
                img_path = os.path.join(root, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                img_means.append(np.mean(img))
                img_stds.append(np.std(img))
                
    overall_mean = np.mean(img_means)/255
    overall_std =  np.sqrt(np.mean([s**2 for s in img_stds]))/255
    print("Overall Mean:", overall_mean)
    print("Overall Std:", overall_std)

    return overall_mean, overall_std
    
def process_raw_images(raw_data_dir, dest_data_dir, dataset, crop_area=None):
    '''
    Normalize raw images in range 0-255 and save as png. 
    '''

    print("Converting to PNG images", dataset)
    shutil.rmtree(f"{dest_data_dir}/{dataset}", ignore_errors=True)
    
    for row in sorted(os.listdir(f"{raw_data_dir}/{dataset}")):
        for col in sorted(os.listdir(f"{raw_data_dir}/{dataset}/{row}")):
            
            dest_img_dir = f"{dest_data_dir}/{dataset}/{int(row[3]):02d}_{int(col[3]):02d}"
            os.makedirs(dest_img_dir, exist_ok=True)
            
            for burst in sorted(os.listdir(f"{raw_data_dir}/{dataset}/{row}/{col}")):

                for shot in sorted(os.listdir(f"{raw_data_dir}/{dataset}/{row}/{col}/{burst}")):
                    
                    img = cv2.imread(f"{raw_data_dir}/{dataset}/{row}/{col}/{burst}/{shot}", cv2.IMREAD_UNCHANGED)
                    img = (img-img.min())/(img.max()-img.min()) * 255
    
                    if crop_area is not None:
                        crop_t, crop_l, crop_h, crop_w = crop_area
                        img = img[crop_t:crop_h, crop_l:crop_w]
                    
                    cv2.imwrite(f"{dest_img_dir}/burst{int(burst[5:]):02d}_shot{int(shot.split('.')[0][4:]):03d}.png", img)
                    

def generate_labels(data_dir, dataset):

    if dataset not in FIRST_DAMAGED_SHOTS:
        print(f"Dataset {dataset} doesn't have damaged indexes yet")
        return

    print(f"Generating class labels  for {dataset}")
    count = 0
    for loc in sorted(os.listdir(f"{data_dir}/images/{dataset}")):
        
        if len(loc.split('.'))>1:
            continue
        
        row, col = [int(v) for v in loc.split('_')]

        # skip row1, col1 due to unstable images
        if (row==1) and (col==1):
            continue

        try:
            damaged_burst_idx, damaged_shot_idx = FIRST_DAMAGED_SHOTS[dataset][row, col]
        except:
            
            continue

        os.makedirs(f"{data_dir}/labels/{dataset}", exist_ok=True)
        
        img_files = listdir(f"{data_dir}/images/{dataset}/{loc}", file_type='png')
        with open(f"{data_dir}/labels/{dataset}/{loc}.txt", 'w+') as f:

            # NOTE! This assumes that, for each location, shots after the first damaged idx are also damaged. 
            # This may produce false class labels if the shots after the first damaged index are not actually damaged. 
            label = 0
            for idx, file in enumerate(img_files):
                
                img = f"{dataset}/{loc}/{file}"
                
                burst_idx, shot_idx = [int(v) for v in re.findall(r'\d+', file)]

                # set the label to 1 after the first damaged index 
                if damaged_burst_idx is not None:
                    if (burst_idx==damaged_burst_idx):
                        if (shot_idx==damaged_shot_idx):
                            label = 1
                        
                f.write(img + ',' + str(label) + '\n')
                count += 1
                
    print(f"{count} labels created!")

def load_labels(data_dir, datasets, num_classes=2):
    img_labels = []
    for dataset in datasets:
        for loc in os.listdir(f"{data_dir}/labels/{dataset}"):
            label_file = f"{data_dir}/labels/{dataset}/{loc}"
            if label_file.endswith('.txt'):
                for img_label in load_file(label_file):
                    img, label = img_label.split(',')
                    onehot_label = np.eye(num_classes)[int(label)]
                    img_labels.append((img, onehot_label))
    return img_labels
    
def train_val_split(data_dir, dataset, val_split=0.3, num_classes=2, balanced_class=False, shuffle=False):
    
    train_labels = []
    val_labels = []
    
    if os.path.exists(f"{data_dir}/labels/{dataset}"):
        print(f"Splitting labels for {dataset} ...")

        for loc in listdir(f"{data_dir}/labels/{dataset}", file_type='txt'):

            # Separate the labels into classes
            cls_dict = defaultdict(list)
            for img_label in load_file(f"{data_dir}/labels/{dataset}/{loc}"):
                img, label = img_label.split(',')
                onehot_label = np.eye(num_classes)[int(label)]
                cls_dict[int(label)].append((img, onehot_label))
            
            # undersample class with large samples
            if balanced_class:
                # print("Addressing class imbalance ")
                class_0_count, class_1_count = len(cls_dict[0]), len(cls_dict[1])
                # print(dataset, loc, class_0_count, class_1_count)
                np.random.seed(42)
                # NOTE! If there is only class in this location, it will be ignored
                if class_0_count > class_1_count:
                    cls_dict[0] = [cls_dict[0][idx] for idx in np.random.choice(class_0_count, size=class_1_count)]
                else:
                    cls_dict[1] = [cls_dict[1][idx] for idx in np.random.choice(class_1_count, size=class_0_count)]
            
            # Split each class labels into training and validation sets                           
            for c, class_labels in cls_dict.items():
                print(f"{dataset}/{loc}, {c=}, {len(class_labels)} images ")
                if shuffle:
                    random.seed(42)
                    random.shuffle(class_labels)
                train_split_indx = int(len(class_labels)*(1-val_split))
                train_labels.extend(class_labels[:train_split_indx])
                val_labels.extend(class_labels[train_split_indx:])

    else:
        print(f"{data_dir}/{dataset} does not exist!!")

    return train_labels, val_labels

def create_train_val_split(data_dir, datasets, val_split=0.3, num_classes=2, balanced_class=False, shuffle=False):
    all_train_labels = []
    all_val_labels = []
    for dataset in datasets:
        train_labels, val_labels = train_val_split(data_dir, dataset, val_split, num_classes, balanced_class, shuffle)
        all_train_labels.extend(train_labels)
        all_val_labels.extend(val_labels)
    return all_train_labels, all_val_labels


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Preprocesses data for supervised laser damage prediction')
    
    parser.add_argument("--raw_data-dir", type=str, default='/home/ubuntu/Projects/clf-laser-damage-prediction/data/FS_LIDT_SEPT22/Sample_NF', 
        help="Directory where the raw images are located")
    
    parser.add_argument("--dest_data-dir", type=str, default='/home/ubuntu/Projects/clf-laser-damage-prediction/data/near_field', 
        help="Directory to save the preprocessed images/labels")

    parser.add_argument("--datasets", type=str, nargs='+', default=None, 
        help="List of datasets to process e.g. ['THALES_SESO_A']")
    
    parser.add_argument("--process_raw_images", action='store_true', default=False,
        help="If to convert TIFFs to PNGs and save converted images with new file format e.g. 01_01/burst01_shot001.png")
    
    args = parser.parse_args()

    if args.datasets is None:
        args.datasets = ALL_DATASETS

    # generate labels
    for dataset in args.datasets:
        
        # Preprocess raw images at once
        if not os.path.exists(f"{args.dest_data_dir}/images/{dataset}") or args.process_raw_images:
            crop_area = (6, 96, 486, 576) # (top, left, bottom, right)
            process_raw_images(args.raw_data_dir, args.dest_data_dir + '/images' , dataset, crop_area)
        
        generate_labels(args.dest_data_dir, dataset)
        
        # create train and val labels
        train_labels, val_labels = train_val_split(args.dest_data_dir, dataset, val_split=0.3, balanced_class=True, shuffle=True)

        # save class labels to html
        # if len(train_labels)>0:
        #     all_labels_dict = [dict(img=img_label[0], label=img_label[1])  for img_label in train_labels + val_labels]
        #     html_content = generate_html(all_labels_dict, 
        #         data_dir=args.img_data_dir, 
        #         title=f'GT Labels - {dataset}'
        #     )
        #     save_file(f'../data/{dataset}_labels.html', html_content)
