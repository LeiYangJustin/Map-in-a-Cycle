import os
import time
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as RotTool

from parse_config import ConfigParser
from utils import clean_state_dict, get_instance
from utils.util import dict_coll

## model
import pointnet3 as module_arch
## dataset
import pointcloud_data_loaders as module_pc_data
## alignement
from pointcloud_utils.alignment_check import AlignmentCheck 

## evaluation and correspondence
from pointnet3 import DVE_loss 


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

validation_task_type = [ 
    'correspondence', 
    'registration']

class AllInOneChecker():
    def __init__(self,
        task_type,  model, logger, dataset, num_classes, save_dir, temperature, loader_params, 
        device, early_stop_iter=10, mini_eval=False, _debug=False):
        assert task_type in validation_task_type
        self.task_type = task_type
        self.model = model
        self.model.eval()
        self.logger = logger
        self.dataset = dataset
        self.num_classes = num_classes
        self.save_dir = save_dir
        self.temperature = temperature
        self.early_stop_iter = early_stop_iter
        self._debug = _debug
        self.mini_eval = mini_eval
        self.device = device
        self.iteration = 0
        self.dataloader_params = loader_params

    def initialize_dataloader(self, loader_params):        
        loader_kwargs = {}
        if loader_params["coll_func"] == "flatten":
            loader_kwargs["collate_fn"] = coll
        elif loader_params["coll_func"] == "dict_flatten":
            loader_kwargs["collate_fn"] = dict_coll
        else:
            raise ValueError("collate function type {} unrecognised".format(coll_func))
        
        if loader_params["disable_workers"]:
            num_workers = 0
        else:
            num_workers = 4

        return DataLoader(
                self.dataset,
                batch_size=loader_params["batch_size"],
                num_workers=num_workers,
                #drop_last=True,
                shuffle=False,
                pin_memory=True,
                **loader_kwargs,
            )

    def check_alignment(
        self, translation_range, rotation_angle_range, num_samples, seed=None, early_stop_iter=None, mini_eval=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        if early_stop_iter is None:
            early_stop_iter = self.early_stop_iter
        if mini_eval is None:
            mini_eval = self.mini_eval
        self.logger.info(f"check alignment iteration: {self.iteration}")

        with torch.no_grad():
            avg_rot_mse_err = AverageMeter()
            avg_rot_rmse_err = AverageMeter()
            avg_rot_mae_err = AverageMeter()
            
            avg_tra_mse_err = AverageMeter()
            avg_tra_rmse_err = AverageMeter()
            avg_tra_mae_err = AverageMeter()
            
            
            checker = AlignmentCheck(
                temperature=0.05, 
                translation_range=translation_range, 
                rotation_angle_range=rotation_angle_range, 
                registration=True)

            ## evaluation
            est_eulers, est_rotations, est_translations = [], [], []
            for i, batch in enumerate(self.data_loader):
                ## process data
                data, meta = batch["data"], batch["meta"]
                data = data.to(self.device)
                B, b, N, C = data.shape; assert b==1
                data = data.squeeze(1)
                if C == 6:
                    has_normal = True

                ## generated randomly posed data
                data_dst = checker.transform_data(data, has_normal=has_normal, random_perm=False)
                
                if num_samples is not None:
                    ## random viewpoint to mimic the partial oclussion
                    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
                    data = checker.subsample(data, partial_vp=random_p1, has_normal=has_normal, num_samples=num_samples)
                    data_dst = checker.subsample(data_dst, partial_vp=random_p1, has_normal=has_normal, num_samples=num_samples)
                else:
                    print('num_samples is None; no subsampling performed')

                data_src = data
                feat_src = self.model(data_src) ## B, N, D
                feat_dst = self.model(data_dst)

                for k in range(3):
                    feat = self.model(data) ## B, N, D
                    est_eulers, est_rotations, est_translations, est_shape = \
                        checker(data, feat, data_dst, feat_dst, has_normal=has_normal)
                    del feat
                    data = est_shape.unsqueeze(0)
                    
                ## compute the final transformation
                R, est_translations = checker._align(data_src[0,:,:3], data[0,:,:3])
                est_eulers = RotTool.from_matrix(R.cpu()).as_euler('zyx', degrees=True)

                ## compute error
                eulers_ab = np.concatenate(checker.angle_list, axis=0)
                translations_ab = np.asarray(torch.cat(checker.translation_list,dim=0).cpu())
            
                # rotations_ab_pred = np.concatenate(rotations, axis=0)
                eulers_ab_pred = est_eulers
                translations_ab_pred = np.asarray(est_translations.cpu())
                print(f"{eulers_ab} ---- {eulers_ab_pred}")
                print(f"{translations_ab} ---- {translations_ab_pred}")

                r_ab_mse = np.mean((eulers_ab-eulers_ab_pred)**2)
                r_ab_rmse = np.sqrt(r_ab_mse)
                r_ab_mae = np.mean(np.abs(eulers_ab-eulers_ab_pred))
                avg_rot_mse_err.update(r_ab_mse, B)
                avg_rot_mae_err.update(r_ab_mae, B)
                avg_rot_rmse_err.update(r_ab_rmse, B)

                self.logger.info(
                    f"avg rotation err at iter {i}: "
                    f"MSE {avg_rot_mse_err.avg}; "
                    f"RMSE {avg_rot_rmse_err.avg}; "
                    f"MAE {avg_rot_mae_err.avg}")

                t_ab_mse = np.mean((translations_ab-translations_ab_pred)**2)
                t_ab_rmse = np.sqrt(t_ab_mse)
                t_ab_mae = np.mean(np.abs(translations_ab-translations_ab_pred))
                avg_tra_mse_err.update(t_ab_mse, B)
                avg_tra_mae_err.update(t_ab_mae, B)
                avg_tra_rmse_err.update(t_ab_rmse, B)

                self.logger.info(
                    f"avg translation err at iter {i}: "
                    f"MSE {avg_tra_mse_err.avg}; "
                    f"RMSE {avg_tra_rmse_err.avg}; "
                    f"MAE {avg_tra_mae_err.avg}")
        
        self.logger.info(
            f"Final average: "
            f"ROT_MSE {avg_rot_mse_err.avg}; "
            f"ROT_RMSE {avg_rot_rmse_err.avg}; "
            f"ROT_MAE {avg_rot_mae_err.avg}; "
            f"TRA_MSE {avg_tra_mse_err.avg}; "
            f"TRA_RMSE {avg_tra_rmse_err.avg}; "
            f"TRA_MAE {avg_tra_mae_err.avg}")

    def check_correspondence(self, dve_loss_args, seed=None, early_stop_iter=None, mini_eval=None):
        if seed is not None:
            torch.manual_seed(seed)
        if early_stop_iter is None:
            early_stop_iter = self.early_stop_iter
        if mini_eval is None:
            mini_eval = self.mini_eval
        self.logger.info(f"check correspondence iteration: {self.iteration}")

        with torch.no_grad():
            ## visualize the src_data
            avg_total_loss = AverageMeter()
            avg_cycle_loss = AverageMeter()
            avg_perm_loss = AverageMeter()
            avg_correct_match = AverageMeter()

            corr_checker = DVE_loss(**dve_loss_args, temperature=self.temperature)

            ## evaluation
            for i, batch in enumerate(self.data_loader):
                if (mini_eval and i == early_stop_iter):
                    print(f"mini_eval: early stop at {i}")
                    break
                
                ## process data
                tgt_data, tgt_meta = batch["data"], batch["meta"]
                tgt_data = tgt_data.to(self.device)
                B, b, N, C = tgt_data.shape
                print(tgt_data.shape)
                tgt_feat = self.model(tgt_data) ## B, N, D
                if self._debug:
                    print("tgt_data shape: ", tgt_data.shape)
                    print("tgt_feat shape: ", tgt_feat.shape)
                
                output_loss, output_info = corr_checker(tgt_feat, tgt_meta, epoch=self.iteration)

                avg_total_loss.update(output_loss['total_loss'], B)
                avg_cycle_loss.update(output_loss['cycle_loss'], B)
                avg_perm_loss.update(output_loss['perm_loss'], B)
                ##
                correct_match = output_info['correct_match']
                avg_correct_match.update(correct_match, B)

                self.logger.info(f"batch indices: {batch['meta']['index']}")
                self.logger.info(
                    f"{i} batch averaged rate: {correct_match:.6f} --- all-time averaged rate {avg_correct_match.avg:.6f}"
                    )

            self.logger.info(
                f"total_loss: {avg_total_loss.avg:.6f}\n"
                f"cycle_loss: {avg_cycle_loss.avg:.6f}\n"
                )
            self.logger.info(
                f"all-time averaged rate {avg_correct_match.avg:.6f}"
                )

    def check_iteration(self, params, max_iter=10):

        self.data_loader = self.initialize_dataloader(self.dataloader_params)
        print("data loader initialized")
        for i in range(max_iter):
            self.iteration = i
            if self.task_type == "correspondence":
                self.check_correspondence(dve_loss_args=params)            
            elif self.task_type == "registration":
                translation_range = params['translation_range']
                rotation_angle_range = params['rotation_angle_range']
                num_samples = params.get('num_samples', None)
                self.check_alignment(
                    translation_range=translation_range,
                    rotation_angle_range=rotation_angle_range,
                    num_samples=num_samples,
                    seed=i)
            else:
                raise NotImplementedError

def evaluation(config, logger=None):
    ##########################################################################################
    device = torch.device('cuda:0' if config["n_gpu"] > 0 else 'cpu')
    if logger is None:
        logger = config.get_logger('test')
    logger.info("Running evaluation with configuration:")
    logger.info(config)
    
    task_type = config["tester"]["task_type"]
    assert task_type in validation_task_type
    logger.info(f"The task is {task_type}")

    print("mini_eval: ", config.get("mini_eval", False))
    print("result_dir: ", config["tester"]["result_dir"])

    ##########################################################################################
    # dataset
    if task_type == "correspondence":
        print("correspondence")
        dataset = get_instance(module_pc_data, 'dataset', config, split='test', has_warper=True)
    elif task_type == "registration":
        print("registration")
        dataset = get_instance(module_pc_data, 'dataset', config, split='test', has_warper=False, augmentation=True)
    else:
        raise NotImplementedError
    
    ##########################################################################################
    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()
    ckpt_path = config._args.resume
    logger.info(f"Loading checkpoint: {ckpt_path} ...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # model and eval
    state_dict = checkpoint['state_dict']
    model.load_state_dict(clean_state_dict(state_dict))
    model = model.to(device)
    model.eval()

    ##########################################################################################
    ## initialize evaluation
    loader_params = {}
    loader_params["coll_func"] = config.get("collate_fn", "dict_flatten")
    loader_params["disable_workers"] = config["disable_workers"]
    loader_params["batch_size"]=config["batch_size"]

    mini_eval = config.get("mini_eval", False)
    temperature = config["temperature"]
    num_classes = config['dataset']['num_lbl_classes']
    # num_points = config["dataset"]["args"]["num_points"]
    
    save_dir = os.path.join(config["tester"]["result_dir"], config["tester"]["task_type"])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    eval_checker = AllInOneChecker(
        task_type, model, logger, dataset, num_classes,
        save_dir, temperature, loader_params, device,
        early_stop_iter=10, _debug=False, mini_eval=mini_eval)

    if task_type == "correspondence":
        eval_checker.check_iteration(params=config['dve_loss_args'], max_iter=1)
    elif task_type == "registration":
        eval_checker.check_iteration(params=config['align_args'], max_iter=3)
    else:
        raise NotImplementedError


#####################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--config', help="config file path")
    parser.add_argument('--resume', help='path to ckpt for evaluation')
    parser.add_argument('--device', help='indices of GPUs to enable')
    parser.add_argument('--mini_eval', action="store_true")
    parser.add_argument('--n_gpu', default=0, type=int, help='indices of GPUs to enable')
    parser.add_argument('--task_type', default=None, type=str,
                        help='one of the following: 1) correspondence; 2) registration; 3) tsne')  
    parser.add_argument('--disable_workers', action="store_true")

    eval_config = ConfigParser(parser)
    msg = "For evaluation, a model checkpoint must be specified via the --resume flag"
    assert eval_config._args.resume, msg

    # command-line args
    eval_config["disable_workers"] = eval_config._args.disable_workers
    eval_config["mini_eval"] = eval_config._args.mini_eval
    eval_config["n_gpu"] = eval_config._args.n_gpu
    if eval_config._args.task_type is not None:
        eval_config["tester"]["task_type"] = eval_config._args.task_type

    assert eval_config["tester"].get("task_type", False), "You must specify a task to be evaluated"
    print(eval_config["tester"]["task_type"])
    print(eval_config["tester"].get("num_references", "N/A"))
    print(eval_config["tester"].get("num_clusters", "N/A"))
    evaluation(eval_config)