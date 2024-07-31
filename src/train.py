"""
Created on Wed June  6 20:26:47 2024
@author: Niraj Bhujel (niraj.bhujel@stfc.ac.uk)
"""

import os
import sys
import numpy as np

import time
import json
import yaml
import pickle
import math
import shutil
import hydra
import logging
import traceback
import itertools

import datetime
from omegaconf import DictConfig, OmegaConf

import matplotlib
matplotlib.use('agg')

import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic=True

from utils.misc import create_new_dir, copy_src
from utils.utils import set_random_seed, model_parameters
from trainer import Trainer
from datasets.preprocess import DATA_SUBSETS, ALL_DATASETS

def ddp_setup(rank: int, world_size: int, backend='nccl'):
    
    # set master address if not set
    if os.getenv("MASTER_ADDR", None) is None:
        os.environ["MASTER_ADDR"] = '127.0.0.1' #"localhost" 
        os.environ["MASTER_PORT"] = '29500' # "12355"

    # torch.distributed.init_process_group(backend="nccl", init_method='env://', rank=rank, world_size=world_size)
    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=36000))
    torch.cuda.set_device(rank)
    
    # Try to use all memory allocated for caching - this is to prevent cuda out of memory which is caused by another training process
    torch.cuda.set_per_process_memory_fraction(0.99, rank)
    print(f"DDP Initialized - Master :{os.environ['MASTER_ADDR']}, Port:{os.environ['MASTER_PORT']}, rank={rank}, world_size={world_size}")


def run(local_rank, world_size, cfg):
    
    print(f"Running {local_rank}/{world_size} ......")

    if cfg.optim.lr_adjust:
        if cfg.optim.lr_adjust_rule=="sqrt_wrt_1024":
            cfg.optim.lr = round(cfg.optim.lr * math.sqrt(cfg.train.batch_size * world_size /1024), 6)
        else:
            cfg.optim.lr = cfg.optim.lr * world_size * cfg.train.batch_size/16

    if cfg.train.ddp:
        ddp_setup(local_rank, world_size)
        print(f"DDP initialized for rank {local_rank}/{world_size}: ", torch.distributed.is_initialized())
    
    exp_name = "_".join([cfg.exp.name,
                         # "_".join(cfg.data.test_datasets),
                         cfg.net.type,
                         f"lr{cfg.optim.lr:.4f}",
                         f"e{cfg.train.epochs}",
                         f"b{cfg.train.batch_size*world_size}",
                        ])

    log_dir = f"../logs/session{cfg.exp.session:03d}/{exp_name}/{"_".join(cfg.data.test_datasets)}"
    
    # Create log directories and backup src code
    if not cfg.train.debug:
        cfg.exp.log_dir = create_new_dir(log_dir)
        cfg.exp.ckpt_dir = create_new_dir(f"{log_dir}/checkpoints")
        cfg.exp.summary_dir = create_new_dir(f"{log_dir}/summary")

        # save config
        with open(cfg.exp.log_dir + '/cfg.yaml', "w") as f:
            yaml.dump(OmegaConf.to_yaml(cfg), f)

        # Save the command
        with open(cfg.exp.log_dir + "/command.txt", 'w') as f:
            f.write(' '.join(sys.argv))
        
        # Backup src scripts
        copy_src("../src", f"{cfg.exp.log_dir}/src/")
    
    # Create trainer
    trainer = Trainer(cfg, local_rank, world_size)
    
    if local_rank==0:        
        trainer.logger.info(f"Running Experiment: {exp_name}")
        trainer.logger.info(trainer.network)
        trainer.logger.info(f"Training parameters: {model_parameters(trainer.network)} (backbone:{model_parameters(trainer.network.net)},  heads: {model_parameters(trainer.network.head)})")
        

    trainer.train()
                
    if cfg.train.ddp:
        destroy_process_group()

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    
    # Print current config
    print(OmegaConf.to_yaml(cfg))

    ## Test each surface
    # for test_dset_name in cfg.data.subsets:
    #     cfg.data.test_datasets = DATA_SUBSETS[test_dset_name]
    #     cfg.data.train_datasets = list(itertools.chain(*[DATA_SUBSETS[k] for k in cfg.data.subsets if k!=test_dset_name]))

    # Test each dataset
    for dset in ALL_DATASETS:
        cfg.data.test_datasets = [dset]
        cfg.data.train_datasets = [d for d in ALL_DATASETS if d!=dset]
        
        # Set random seed
        set_random_seed(cfg.exp.seed)

        if cfg.train.ddp:

            # Spawn ddp processes either use (1) torch distributed run or (2) use mp.spawn
            if cfg.train.dist_run:
                local_rank = int(os.environ["LOCAL_RANK"]) # os.environ is set during distributed run
                world_size=int(os.environ['WORLD_SIZE'])
                run(local_rank, world_size, cfg)

            else:
                world_size = torch.cuda.device_count()
                mp.spawn(run, args=(world_size, cfg), nprocs=world_size)

        else:        
            run(0, 1, cfg)

if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Exiting from training early because of KeyboardInterrupt')
        sys.exit()
    except Exception as e:
        print(e)
        traceback.print_exc()

