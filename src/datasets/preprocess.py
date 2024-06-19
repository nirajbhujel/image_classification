import os
import sys
import re
import random
import numpy as np
from collections import defaultdict, Counter
root_path = '../src'
if not root_path in sys.path:
    sys.path.insert(0, root_path)

from utils.misc import load_file, save_file, listdir


FIRST_DAMAGED_SHOTS = {
'Grating_A6':{
    (1, 1): (None, None),
    (1, 2): (6, 5),
    (2, 1): (6, 11),
    (2, 2): (6, 5),
    (3, 1): (6, 51),
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
    (1, 1): (None, None),
    (1, 2): (6, 1),
    (2, 1): (6, 3),
    (2, 2): (6, 7),
    (3, 1): (6, 1),
    (3, 2): (6, 1),
    (4, 1): (6, 3),
    (4, 2): (6, 1),
    (5, 1): (None, None),
    (5, 2): (6,4)
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
    (4, 1): (7, 20),
    (4, 2): (5, 82),
    (4, 3): (7, 6),
    (5, 1): (1, 70),
    (5, 2): (2, 5),
    (5, 3): (1, 92),
    (6, 1): (1, 6),
    (6, 2): (1, 1),
    (6, 3): (1, 6),
    },
}

def create_labels(data_dir, dataset):

    if dataset not in FIRST_DAMAGED_SHOTS:
        return

    print(f"Creating class labels - {dataset=}")
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


def train_val_split(data_dir, dataset, val_split=0.3, balanced_class=False, shuffle=False):
    
    train_labels = []
    val_labels = []
    
    if os.path.exists(f"{data_dir}/labels/{dataset}"):
        print(f"Splitting {dataset} labels into train and validaton set")

        for loc_labels in listdir(f"{data_dir}/labels/{dataset}", file_type='txt'):

            # Separate the labels into classes
            cls_dict = defaultdict(list)
            for img_label in load_file(f"{data_dir}/labels/{dataset}/{loc_labels}"):
                img, label = img_label.split(',')
                cls_dict[int(label)].append((img, int(label)))
            
            # undersample class with large samples
            if balanced_class:
                # print("Addressing class imbalance ")
                class_0_count, class_1_count = len(cls_dict[0]), len(cls_dict[1])
                np.random.seed(42)
                if class_0_count > class_1_count:
                    cls_dict[0] = [cls_dict[0][idx] for idx in np.random.choice(class_0_count, size=class_1_count)]
                else:
                    cls_dict[1] = [cls_dict[1][idx] for idx in np.random.choice(class_1_count, size=class_0_count)]

            # Split each class labels into training and validation sets                           
            for c, class_labels in cls_dict.items():
                # print(f"{loc_labels}, {c=}, {len(class_labels)} images ")
                if shuffle:
                    random.seed(42)
                    random.shuffle(class_labels)
                train_split_indx = int(len(class_labels)*(1-val_split))
                train_labels.extend(class_labels[:train_split_indx])
                val_labels.extend(class_labels[train_split_indx:])

    else:
        print(f"{dataset} labels not available to split.")

    return train_labels, val_labels

def create_train_val_split(data_dir, datasets, val_split=0.3, balanced_class=False, shuffle=False):
    all_train_labels = []
    all_val_labels = []
    for dataset in datasets:
        train_labels, val_labels = train_val_split(data_dir, dataset, val_split, balanced_class, shuffle)
        all_train_labels.extend(train_labels)
        all_val_labels.extend(val_labels)
    return all_train_labels, all_val_labels


def generate_html(data, data_dir, title='Images'):
    '''
    data: Data to visualize. List of dictionaries {img: str, label: int, pred: int, prob: float}.
    data_dir: Absolute path of the image location to load images
    '''

    # Convert to nested dict structure
    dataset_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for item in data:
        dset, loc, filename = item['img'].split('/')
        burst = filename.split('_')[0]
        shot = filename.split('_')[1]
        dataset_dict[dset][loc][burst].append((shot, item))

    html = ['<html>', '<head>', f'<title>{title}</title>', '</head>', '<body>']

    for dset, locations in sorted(dataset_dict.items()):
        html.append(f'<h2>Dataset: {dset}</h2>')
        
        for location, bursts in sorted(locations.items()):
            html.append(f'<h3>{dset} - {location}</h3>')
            
            html.append('<table border="1">')
            
            for burst, shots in sorted(bursts.items()):

                html.append('<tr>')
                html.append(f'<td>{burst}</td>')
                
                html.append('<td>')

                # show single image in td
                for shot, item in shots:

                    color = 'black'
                    caption = f'c={item['label']}'
                    img_src = f'{data_dir}/{dset}/{location}/{burst}_{shot}'

                    # change color if prediction available
                    if 'pred' in item:
                        color = 'red' if item['pred']!=item['label'] else color
                        caption += f', y={item['pred']} p={item['prob']:.2f}'

                    # Construct the figure element with image and captions
                    html.append('<figure style="display: inline-block; margin: 10px; text-align: center; font-size: 10px;">')
                    # Image with colored border and title attributes to show information when mouse hover over image
                    html.append(f'<img src="{img_src}" alt="{burst}_{shot}" title="{img_src}" style="width:100px;height:100px; border: 2px solid {color};"> ')
                    # Image caption (image label)
                    html.append(f'<figcaption style=font-style">{caption}</figcaption>')
                    html.append('</figure>')
        
                html.append('</td>')
                html.append('</tr>')
            
            html.append('</table>')

    html.append('</body>')
    html.append('</html>')

    return '\n'.join(html)


if __name__=='__main__':
    
    data_dir = '../data/near_field'
    datasets = [
        'Grating_A6',
        'Grating_SON1',
        'LASEROPTIK_SON1', 
        'THALES_SESO_10000',
        'THALES_SESO_A',
        'LASEROPTIK',
        'LASEROPTIK_10000'
        ]


    for dataset in datasets:
        create_labels(data_dir, dataset)

        train_labels, val_labels = train_val_split(data_dir, dataset)

        if len(train_labels)>0:
            all_labels_dict = [dict(img=img_label[0], label=img_label[1])  for img_label in train_labels + val_labels]
            html_content = generate_html(all_labels_dict, 
                data_dir="/home/ubuntu/Projects/clf-laser-damage-prediction/data/near_field/images", 
                title=f'GT Labels - {dataset}')
            save_file(f'../data/{dataset}_labels.html', html_content)