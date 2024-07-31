"""
Created on Wed June  6 10:57:22 2024
@author: Niraj Bhujel (niraj.bhujel@stfc.ac.uk)
"""

import os
import sys
import math
import json
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.dataset import LaserImageDataset, get_transforms, create_datasets
from models.network import Network
from models.loss import BCE_Loss

from utils import *
from utils import logger
from visualization.vis_utils import create_figure

class Trainer:

    def __init__(self, cfg, rank=0, world_size=1, **kwargs):

        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size

        self.log_dir = cfg.exp.log_dir
        self.ckpt_dir = cfg.exp.ckpt_dir
        self.debug = cfg.train.debug
        
        self.logger = logger.create_logger(cfg.exp.log_dir + "/train_log.txt")
        
        if not self.debug:
            self.tb_logger = SummaryWriter(log_dir=cfg.exp.summary_dir)
            
        if cfg.train.ddp:
            self.device = torch.device("cuda:{}".format(rank))
        else:
            self.device = torch.device(f"cuda:{cfg.train.gpu}") if torch.cuda.is_available() else torch.device("cpu")    
        
        print(f"GPU: {self.device}/{world_size}, CPUs: {os.cpu_count()}")
        
        self.train_dataset, self.val_dataset, self.test_dataset = create_datasets(cfg)
            
        self.train_loader = DataLoader(self.train_dataset,
                                        batch_size=cfg.train.batch_size,
                                        shuffle=False if cfg.train.ddp else True,
                                        num_workers=cfg.train.num_workers,
                                        worker_init_fn=seed_worker, # for reproducibility in multi-process
                                        generator=torch_generator(cfg.exp.seed),
                                        pin_memory=False, # pin_memory is slightly faster but consume high cpu resource
                                        sampler=DistributedSampler(self.train_dataset) if cfg.train.ddp else None,
                                        persistent_workers=True if cfg.train.num_workers>0 else False,
                                        drop_last=False
                                        )
        
        self.val_loader = DataLoader(self.val_dataset,
                                        batch_size=cfg.train.batch_size,
                                        shuffle=False,
                                        num_workers=cfg.train.num_workers,
                                        persistent_workers=True if cfg.train.num_workers>0 else False,
                                        drop_last=True
                                        )
        
        self.test_loader = DataLoader(self.test_dataset,
                                        batch_size=cfg.train.batch_size,
                                        shuffle=False,
                                        num_workers=cfg.train.num_workers,
                                        persistent_workers=True if cfg.train.num_workers>0 else False,
                                        drop_last=True
                                        )

        self.logger.info(f"Train samples: {len(self.train_dataset)}, BatchSize: {self.train_loader.batch_size}, Batches: {len(self.train_loader)}")
        self.logger.info(f"Val samples: {len(self.val_dataset)}, BatchSize: {self.val_loader.batch_size}, Batches: {len(self.val_loader)}")
        self.logger.info(f"Test samples: {len(self.test_dataset)}, BatchSize: {self.test_loader.batch_size}, Batches: {len(self.test_loader)}")
        
        if not (len(self.train_loader)>0 or len(self.test_loader)>0):
            raise
            
        
        # setup network
        self.network = Network(cfg)
        self.network = self.network.to(self.device)
        self.network_ddp = self.network
        self.logger.info(f"Training parameters: {self.network.num_params}")

        if not cfg.train.debug:
            self.tb_logger.add_scalar("parameters", self.network.num_params)
            
        # Setup MTL
        self.mtl_wrapper = MultiTaskLossWrapper(cfg.loss)

        # Setup Optimizers
        net_params_group = get_params_groups(self.network.net) # (list of dict)
        net_params_group.append({'params': self.network.head.parameters(), 'lr': 2*cfg.optim.lr})
            
        self.optimizer = optim.AdamW(net_params_group, lr=cfg.optim.lr)
        
        # Schedulers
        if cfg.optim.lr_scheduler=='reduceonplateau':
            self.lr_scheduler=ReduceLROnPlateau(self.optimizer, 
                                            mode='min', 
                                            factor=cfg.optim.lr_reduce_factor, 
                                            patience=cfg.optim.lr_patience, 
                                            verbose=True)

        elif cfg.optim.lr_scheduler=='cosine':
            self.lr_scheduler=CosineScheduler(self.optimizer, 
                                            base_lr=cfg.optim.lr, 
                                            target_lr=cfg.optim.lr_min, 
                                            epochs=cfg.train.epochs+1, 
                                            steps_per_epoch=len(self.train_loader), 
                                            warmup_epochs=cfg.optim.warmup_epochs)
        # Weight scheduler
        if cfg.optim.weight_decay>0:
            self.wt_scheduler = cosine_scheduler(cfg.weight_decay, 
                                                cfg.weight_decay_end, 
                                                cfg.epochs, 
                                                len(self.train_loader))

        
        # Early Stopping
        self.early_stopping = EarlyStopping(patience=cfg.train.early_stop_patience,
                                            metric_name=cfg.train.early_stop_metric,
                                            trace_func=self.logger.info,
                                            )
        
        # Model Checkpointing 
        self.model_checkpointer = ModelCheckpoint(self.network, 
                                            self.ckpt_dir,
                                            ckpt_name='best_acc',
                                            monitor=cfg.train.eval_metric,
                                            mode='max',
                                            trace_func=self.logger.info,
                                            debug=self.debug
                                            )

        self.bce_loss = BCE_Loss()

        if cfg.task=='classification':
            self.metric_logger = MetricLogger(name='binary_accuracy')

        self.log_hist_freq = cfg.train.eval_interval * len(self.train_loader) #if cfg.single_batch else len(self.train_loader) # save time by logging every eval_interval

        self.global_step = 0
        

    def train(self, ):
        
        cfg = self.cfg
        train_hist = defaultdict(list)
        last_epoch = 0
        start = time.time()
        
        if cfg.train.ddp:
            if has_batchnorms(self.network):
                self.network = nn.SyncBatchNorm.convert_sync_batchnorm(self.network)

            self.network = DDP(self.network, device_ids=[self.rank])
            self.network_ddp = self.network.module
            
                
        if cfg.train.network_in is not None:
            self.logger.info(f"Loading pretrained network from {cfg.train.network_in}")
            try:
                self.network_ddp.load_state_dict(torch.load(cfg.train.network_in), strict=False)
                print("Model loaded sucessfully!!")
            except Exception as e:
                print(e)
                pass
            
        if cfg.train.resume_training:

            try:
                last_epoch = self.load_ckpt(self.device, ckpt_dir=self.ckpt_dir)
                with open(self.log_dir + '/train_hist.pkl', 'rb') as f:
                    train_hist = pickle.load(f)

                self.logger.info(f"Resumed training from epoch {last_epoch}\n")
            except Exception as e:
                print(e)

        self.logger.info(f"Training Started (device: {self.device}, rank: {self.rank})")
        
        for epoch in range(last_epoch, cfg.train.epochs):

            self.curr_epoch = epoch

            if self.rank==0:
                self.logger.info(f"Epoch {epoch}/{cfg.train.epochs}")

            epoch_start = time.time()

            if cfg.train.ddp and (not cfg.train.single_batch):
                self.train_loader.sampler.set_epoch(epoch)

            hist = self.train_epoch(epoch)
            
            if self.rank==0:

                # evaluate if eval condition reached
                eval_condition = ((epoch+1)%cfg.train.eval_interval==0) & (epoch>cfg.optim.warmup_epochs)
                if eval_condition:
                    self.logger.info("Evaluating")
                    
                    hist = self.evaluate_epoch(hist=hist, phase='val', dataloader=self.val_loader, plot=True, log_outputs=not(self.debug))

                    hist = self.evaluate_epoch(hist=hist, phase='test', dataloader=self.test_loader, plot=True, log_outputs=not(self.debug))

                    self.model_checkpointer(np.mean(hist[self.model_checkpointer.monitor]))

                    self.early_stopping(np.mean(hist[self.early_stopping.metric_name]))
                    
                    
                # average metrics over iterations
                for k, v in hist.items():
                    train_hist[k].append(np.mean(v))

                train_hist['etc'].append(time.time() - epoch_start)
                train_hist['lr'].append(float(f"{self.optimizer.param_groups[0]['lr']:.6f}"))

                log_text = ', '.join([f"{k}:{str(v[-1]):.9s}" for k, v in sorted(train_hist.items())])
                free_mem, total_mem = torch.cuda.mem_get_info(self.device)
                log_text += f", used/total:{(total_mem-free_mem)/1024**2:.0f}/{total_mem/1024**2:.0f} Mib"
                self.logger.info(f"epoch: {epoch}, step:{self.global_step}, {log_text}")
                
                # Check for nan values
                for k, v in hist.items():
                    if np.any(np.isnan(v)):
                        self.logger.exception(f"{k} has nan values")
                        raise Exception(f"{k} has nan values!! {v}")
                        
                if not self.debug:
                    for k, v in train_hist.items():
                        self.tb_logger.add_scalar(k, v[-1], epoch)

                    for task_name in self.mtl_wrapper.learnable_tasks:
                        self.tb_logger.add_scalar(f'loss_wts/learnable_{task_name}', self.mtl_wrapper.get_weight(task_name).cpu(), epoch)

                    if eval_condition:
                        self.save_ckpt(epoch, self.ckpt_dir)
                        with open(self.log_dir + '/train_hist.pkl', 'wb') as f:
                            pickle.dump(train_hist, f)

                if self.early_stopping.early_stop:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Compute final metric 
        self.metric_logger.compute()
        
        # Log hyperparam at the end of training
        if (not self.debug) and self.rank==0:

            self.log_hparams(train_hist, epoch)

            with open(self.log_dir + '/train_metric_results.json', 'w') as f:
                json.dump(self.metric_logger.results, f)
                
            with open(self.log_dir + '/train_hist.txt', 'w') as f:
                json.dump(json.dump({k: str(v) for k, v in train_hist.items()}, f), f)
        
            row_header = ['exp']
            row_data = [cfg.exp.name]
            for k, v in train_hist.items():
                row_header.append(k)
                row_data.append(v[-1] if v[-1]//1==0 else np.round(v[-1], 2))
                
            save_to_csv(f"../logs/log_results_session{cfg.exp.session}.csv", row_data, header=row_header)
            
            self.logger.info(f"DONE in {time.time()-start}!!")
            self.tb_logger.flush()
            self.tb_logger.close

    def log_hparams(self, train_hist, final_epoch):
        metric_dict = {f"hparams/{k}":val_list[-1] for k, val_list in train_hist.items() if ('train' in k) or ('val' in k)}
        hparams_dict = dictconf_to_dict(self.cfg)
        hparams_dict["hparams/parameters"] = self.network.num_params
        hparams_dict["hparams/final_epoch"] = final_epoch
        self.tb_logger.add_hparams(hparams_dict, metric_dict=metric_dict, run_name='hparams')

    def train_epoch(self, epoch, **kwargs):

        cfg = self.cfg

        # Container to store epoch history
        hist = defaultdict(list)
        
        pbar = tqdm(total=len(self.train_loader), position=0, leave=True)
        for iter, batch in enumerate(self.train_loader):

            self.optimizer.zero_grad(set_to_none=True)

            # set weight decay
            if cfg.optim.weight_decay>0:
                # only the first group is regularized
                self.optimizer.param_groups[0]["weight_decay"] = self.wt_scheduler[self.global_step]

            hist, loss = self.train_step(batch, hist)
            
            pbar.update(1)
            postfix = ''
            for k, v in hist.items():
                if "train" in k:
                    # if len(v)>0:
                    postfix += f", {k.split('/')[1]}: {v[-1]:8.3f}"
            pbar.set_postfix_str(postfix)
            
            if loss.isnan():
                self.logger.info("loss is nan !!")
                raise Exception

            loss.backward()

            if cfg.optim.clip_grad>0:
                norms = []
                # clip only backbone
                for name, p in self.network_ddp.net.named_parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        # norms.append(param_norm.item())
                        clip_coef = cfg.optim.clip_grad / (param_norm + 1e-6)
                        if clip_coef < 1:
                            p.grad.data.mul_(clip_coef)

            self.optimizer.step() # update all model parameters

            # set learning rate after optimizer step (otherwise the first value of the lr schedule will be skipped in Pytorch>1.1)
            if isinstance(self.lr_scheduler, CosineScheduler):
                self.lr_scheduler.step()
                
            if cfg.train.reset_head_steps is not None:
                if self.global_step == cfg.train.reset_head_steps:
                    print("RESETTING HEADS")
                    self.head.reset_parameters()
            
            self.global_step += 1

            torch.cuda.synchronize()
            # break

        pbar.close()

        if isinstance(self.lr_scheduler, ReduceLROnPlateau) and self.curr_epoch>cfg.optim.warmup_epochs:
            self.lr_scheduler.step(np.mean(hist['train/loss']))

        return hist

    def train_step(self, batch, hist, **kwargs):
        cfg = self.cfg

        self.network.train()

        imgs, labels, _ = batch

        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        preds = self.network(imgs)

        loss = 0

        bce_loss = self.bce_loss(labels, preds)
        hist['train/bce_loss'].append(bce_loss.detach().item())
        loss += bce_loss

        hist['train/loss'].append(loss.detach().item())

        if (self.global_step%self.log_hist_freq==0) and (self.rank==0) and (not self.debug):
            self.log_output_image(dict(images=imgs, labels=labels, preds=preds), cfg.train.n_images, phase='train')

        return hist, loss

    
    def evaluate_epoch(self, dataloader, phase='val', hist=None, plot=False, log_outputs=False):
        self.network.eval()

        cfg = self.cfg

        if hist is None:
            hist = defaultdict(list)

        self.metric_logger.reset()

        pbar = tqdm(total=len(dataloader), position=0)
        with torch.no_grad():
            for iter, batch in enumerate(dataloader):
                pbar.update(1)

                imgs, labels, _ = batch

                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                preds = self.network(imgs)
                preds = F.softmax(preds, dim=-1)

                acc = self.metric_logger(preds, labels)
                hist[f"{phase}/acc"].append(acc.detach().item())

                string = f"mean_accuracy:{np.mean(hist[cfg.train.eval_metric]):.3f}"
                pbar.set_postfix_str(string)

        if (self.global_step%self.log_hist_freq==0) and (not self.debug):
            self.log_output_image(dict(images=imgs, labels=labels, preds=preds), cfg.train.n_images, phase='val')

        return hist

    def log_outputs(self, outputs, phase='train', add_histogram=True, add_mesh=True):
        pass

            
    def log_output_image(self, outputs, n_images=5, phase="train", dpi=100, fontsize=10):

        B, C, H, W = outputs['images'].shape # (492, 656)
        # print(B, C, H, W)

        n_images = min(n_images, B)

        num_cols = 5

        num_rows = n_images//num_cols
        
        fig, axes = create_figure(num_rows, num_cols, subplot_size=(2, W/H*2))

        axes = axes.flatten()
        for i in np.arange(n_images):
            axes[i].imshow(prep_for_plot(outputs['images'][i], unnormalize=True))
            axes[i].set_title(f"label:{outputs['labels'][i].argmax().item()}, pred:{outputs['preds'][i].argmax().item()}", fontsize=fontsize)
            # add color boundary around the image
            for spine in axes[i].spines.values():
                spine.set_edgecolor('green' if outputs['labels'][i].argmax().item()==outputs['preds'][i].argmax().item() else 'red')
                spine.set_linewidth(2) 

        add_plot(self.tb_logger, f"{phase}/output_images", self.global_step)

    def save_ckpt(self, epoch, ckpt_dir, ckpt_name='model_states'):
        state = {
            'last_epoch': epoch,
            'global_step': self.global_step,
            'model_state': self.network_ddp.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state':self.lr_scheduler.state_dict(),
            'mtl_wrapper': self.mtl_wrapper.state_dict(),
            }
        torch.save(state, os.path.join(ckpt_dir, ckpt_name + '.pth'))

    def load_ckpt(self, map_location, ckpt_dir, ckpt_name='model_states'):
        checkpoint = torch.load(os.path.join(ckpt_dir, ckpt_name + '.pth'), map_location=map_location)

        self.global_step = checkpoint['global_step']
        self.network_ddp.load_state_dict(checkpoint['model_state'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.mtl_wrapper.load_state_dict(checkpoint['mtl_wrapper'])

        return checkpoint['last_epoch']

