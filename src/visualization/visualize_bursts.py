import os
import sys
import shutil

from glob import glob
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

def visualize_bursts(burst_dir, num_rows=10, num_cols=20, figwidth=20):
    
    bursts = sorted(os.listdir(burst_dir), key=lambda x: int(x[5:]))
    
    # num_rows = min(num_rows, len(bursts))
    num_rows = len(bursts)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figwidth, figwidth*num_rows/num_cols))
    if num_rows==1:
        axes = axes.reshape(1, -1)
    
    for i, burst in enumerate(bursts):
        
        shots = sorted(glob(f"{burst_dir}/{burst}/*.tiff"))
        
        # Calculate the step size
        col_step = max(1, len(shots)//num_cols)
        
        col_files = []
        for j,file in enumerate(shots[::col_step][:num_cols]): # Slice the shots to fit the number of columns
            img = plt.imread(file)
            axes[i][j].imshow(img[100:400, 200:500])
            col_files.append(os.path.basename(file))

        # set y-label to each row
        axes[i][0].set_ylabel(f"Burst{i+1}", fontsize=8)

    # set x-label to each col
    for j, cfile in enumerate(col_files):
        axes[-1][j].set_xlabel(cfile.split(".")[0], fontsize=8)

    # Hide X and Y axes tick marks
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{'_'.join(burst_dir.split('/')[3:])}")
    plt.tight_layout()
    
    return fig, axes
    # plt.show()

if __name__=='__main__':
    
    # subplot burst shots for each location in a single 
    root_dir = "../../data/FS_LIDT_SEPT22/Sample_NF"

    for dir_name in sorted(os.listdir(root_dir)):

        print(f"Plotting {dir_name=}")

        data_dir = f"{root_dir}/{dir_name}"

        plot_dir = f"../../data/plots/{dir_name}"
        os.makedirs(plot_dir, exist_ok=True)


        for row_id in tqdm(sorted(os.listdir(data_dir))):

            if not os.path.isdir(f"{data_dir}/{row_id}"):
                continue

            for col_id in sorted(os.listdir(f"{data_dir}/{row_id}")):

                fig, axes = visualize_bursts(f"{data_dir}/{row_id}/{col_id}")

                fig.savefig(f"{plot_dir}/{row_id}_{col_id}.png", dpi=500)
                plt.close('all')
            