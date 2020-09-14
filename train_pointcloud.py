import os
import time
import argparse
import numpy as np
import copy
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data.dataloader

# utils
from utils import Logger, dict_coll
from utils import clean_state_dict, coll, get_instance
from trainer.pc_trainer import PCTrainer

# model
import pointnet3 as module_loss
import pointnet3 as module_arch

# dataloader
import pointcloud_data_loaders.shapenet_dataloader as module_pc_data
# import pointcloud_data_loaders.pc_data_loader as module_pc_data

#
from parse_config import ConfigParser


def main(config, resume):
    logger = config.get_logger('train')
    seeds = [int(x) for x in config._args.seeds.split(",")]
    torch.backends.cudnn.benchmark = True
    
    # print information in logger
    logger.info("Launching experiment with config:")
    logger.info(config)

    # what are the seeds?
    if len(seeds) > 1:
        run_metrics = []

    for seed in seeds:
        tic = time.time()
        
        # use manual seed
        if True:
            logger.info(f"Setting experiment random seed to {seed}")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        ## instantialize the model and print its info
        model = get_instance(module_arch, 'arch', config)
        logger.info(model)

        loader_kwargs = {}
        coll_func = config.get("collate_fn", "dict_flatten")
        if coll_func == "flatten":
            loader_kwargs["collate_fn"] = coll
        elif coll_func == "dict_flatten":
            loader_kwargs["collate_fn"] = dict_coll
        else:
            raise ValueError("collate function type {} unrecognised".format(coll_func))

        dataset = get_instance(module_pc_data, 'dataset', config, split='train', has_warper=True)
        if config["disable_workers"]:
            num_workers = 0
        else:
            num_workers = 4
        data_loader = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            **loader_kwargs,
        )
        val_dataset = get_instance(module_pc_data, 'dataset', config, split='test')
        valid_data_loader = DataLoader(val_dataset, batch_size=1, **loader_kwargs)

        # get function handles of loss and metrics
        loss = getattr(module_loss, config['loss'])
        loss_args = config['loss_args']
        loss =loss(**loss_args)

        metrics = [getattr(module_metric, met) for met in config['metrics']]

        ## model parameter statisitcs
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        biases = [x.bias for x in model.modules() if isinstance(x, nn.Conv2d) or isinstance(x, nn.Conv1d) or isinstance(x, nn.Linear)]
        weights = [x.weight for x in model.modules() if isinstance(x, nn.Conv2d) or isinstance(x, nn.Conv1d) or isinstance(x, nn.Linear)]
        trainbiases = [x for x in trainable_params if sum([(x is b) for b in biases])]
        trainweights = [x for x in trainable_params if sum([(x is w) for w in weights])]
        trainparams = [x for x in trainable_params if not sum([(x is b) for b in biases]) and not sum([(x is w) for w in weights])]
        print(
            len(trainparams), 'Parameters', 
            len(trainbiases), 'Biases', 
            len(trainweights), 'Weights', 
            len(trainable_params), 'Total Params'
            )

        ## set different lr to weight and bias
        bias_lr = config.get('bias_lr', None)
        other_lr = config.get('other_lr', None)
        if bias_lr is not None and other_lr is not None:
            print("bias_lr is not None and other_lr is not None")
            optimizer = get_instance(
                torch.optim, 'optimizer', config, 
                [
                     # using the default learning rate
                    {"params": trainweights}, 
                    # using different learning rates
                    {"params": trainbiases, "lr": bias_lr},
                    {"params": trainparams, "lr": other_lr}
                ]
            )
        elif bias_lr is not None:
            optimizer = get_instance(
                torch.optim, 'optimizer', config, 
                [
                     # using the default learning rate
                    {"params": trainweights+trainparams}, 
                    # using different learning rates
                    {"params": trainbiases, "lr": bias_lr},
                ]
            )
        else:
            optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)


        ## scheduler
        lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config,
                                    optimizer)


        ## log dir                            
        print("config.log_dir: ", config.log_dir)
        print("config.result_dir: ", config.result_dir)

        ## pointcloud trainer             
        trainer = PCTrainer(
            model=model,
            loss=loss,
            metrics=metrics,                        # metrics
            resume=resume,                          # resume path
            config=config,                          # config structure
            optimizer=optimizer,
            data_loader=data_loader,
            lr_scheduler=lr_scheduler,
            mini_train=config._args.mini_train,
            valid_data_loader=valid_data_loader,  # no need for validation this time
        )
        trainer.train()
        duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - tic))
        logger.info(f"Training took {duration}")

        # ## evaluation
        # epoch = config["trainer"]["epochs"]
        # config._args.resume = config.save_dir / f"checkpoint-epoch{epoch}.pth"
        # config["mini_eval"] = config._args.mini_train
        # evaluation(config, logger=logger)
        # logger.info(f"Log written to {config.log_path}")

# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-f', '--folded_correlation',
                        help='whether to use folded correlation (reduce mem)')
    parser.add_argument('-p', '--profile', action="store_true",
                        help='whether to print out profiling information')
    parser.add_argument('-b', '--batch_size', default=None, type=int,
                        help='the size of each minibatch')
    parser.add_argument('-g', '--n_gpu', default=None, type=int,
                        help='if given, override the numb')
    parser.add_argument('--seeds', default="0", help='random seeds')
    parser.add_argument('--mini_train', action="store_true")
    parser.add_argument('--train_single_epoch', action="store_true")
    parser.add_argument('--disable_workers', action="store_true")
    parser.add_argument('--check_bn_working', action="store_true")
    parser.add_argument('--vis', action="store_true")
    parser.add_argument('--comment', help="comments to the session", type=str)
    parser.add_argument('--sub_dir', default=None, type=str,
        help="add sub_dir to specify a folder under dataset root")
    config = ConfigParser(parser)

    # We allow a small number of cmd-line overrides for fast dev
    args = config._args
    if args.folded_correlation is not None:
        config["loss_args"]["fold_corr"] = args.folded_correlation
    if config._args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if config._args.n_gpu is not None:
        config["n_gpu"] = args.n_gpu
    config["profile"] = args.profile
    config["vis"] = args.vis
    config["disable_workers"] = args.disable_workers
    config["trainer"]["check_bn_working"] = args.check_bn_working
    config["device"] = args.device
    config["comment"]=args.comment
    if args.sub_dir is not None:
        config["dataset"]["args"]["root"] = os.path.join(config["dataset"]["args"]["root"], args.sub_dir)
        print(config["dataset"]["args"]["root"])


    if args.train_single_epoch:
        print("Restring training to a single epoch....")
        config["trainer"]["epochs"] = 1
        config["trainer"]["save_period"] = 1

    main(config, args.resume)
