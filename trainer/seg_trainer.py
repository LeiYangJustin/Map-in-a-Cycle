import numpy as np
import torch
import time
import tqdm
import datetime
# from torchvision.utils import make_grid
from pkg_resources import parse_version
# from base import BaseTrainer
from base import BasePCTrainer
from torch.nn.modules.batchnorm import _BatchNorm
# from model.metric import runningIOU
from pointcloud_utils.pointcloud_vis import *
from pointcloud_utils.iou_metric import *
from pointcloud_utils.iou_metric import PointCloudIOU
import os
from utils import clean_state_dict
from pointcloud_utils import pointcloud_vis

def my_timer(prev_tic, name, timings):
    timings[name] = time.time() - prev_tic
    tic = time.time()
    return timings, tic

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


class FSTrainer(BasePCTrainer):
    """
    Trainer class

    Note:
        Inherited from BasePCTrainer.
    """

    def __init__(
            self,
            model,
            pretrain,
            metrics,
            optimizer,
            resume,
            config,
            iou_eval,
            data_loader,
            valid_data_loader=None,
            lr_scheduler=None,
            visualizations=None,
            mini_train=False,
            check_bn_working=False,
            **kwargs,
    ):
        super().__init__(model, None, metrics, optimizer, resume, config)
        self.pretrain = pretrain
        self.config = config
        self.iou_eval = iou_eval
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = max(1, int(len(self.data_loader) / 5.))
        
        ############################################################################
        # load state dict
        ckpt_path = config["pretrain"]["resume"]
        self.logger.info(f"Loading checkpoint: {ckpt_path} ...")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            self.pretrain = torch.nn.DataParallel(self.pretrain)
        self.pretrain.load_state_dict(clean_state_dict(state_dict))
        if config['n_gpu'] > 1:
            self.pretrain = self.pretrain.module
        self.pretrain = self.pretrain.to(self.device)
        self.pretrain.eval()
        ############################################################################
        
        
        self.mini_train = mini_train
        self.check_bn_working = check_bn_working

        assert self.lr_scheduler.optimizer is self.optimizer
        assert self.start_epoch >= 1

        # Our epoch 1 is step -1 (but we can't explicitly call
        # step(-1) because that would update the lr based on a negative epoch)
        # NB stateful schedulers eg based on loss won't be restored properly
        if self.start_epoch != 1:
            self.lr_scheduler.step(self.start_epoch - 2)

        # only perform last epoch check for older PyTorch, current versions
        # immediately set the `last_epoch` attribute to 0.
        if parse_version(torch.__version__) <= parse_version("1.0.0"):
            assert self.lr_scheduler.last_epoch == self.start_epoch - 2


    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(
                output,
                target,
                self.data_loader.dataset,
                self.config
            )
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics
    
    def _visualize_examples(self):
        """ visualization
        batch_vis_idx: batch idx to be visualized; output
        """
        batch_vis_idx = None
        if self.config.get('visualize', False):
            num_vis = self.config['visualize']['num_examples']
            batch_vis_idx = np.random.choice(len(self.data_loader.dataset)-3, num_vis, replace=False)
            batch_vis_idx = np.concatenate([batch_vis_idx+3, [0, 1, 2]]) # avoid replacement
            print("batch_vis_idx: ", batch_vis_idx)        
        return batch_vis_idx


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """

        self.model.train()
        train_tic = time.time()
        avg_loss = AverageMeter()

        mean_shape_IOU_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()

        total_metrics = [AverageMeter() for a in range(len(self.metrics))]
        seen_tic = time.time()
        seen = 0
        profile = self.config["profile"]
        total_batches = len(self.data_loader)

        if profile:
            batch_tic = time.time()
        
        for batch_idx, batch in enumerate(self.data_loader):
            data, meta = batch["data"], batch["meta"]
            gtlbls = meta["label"]
            shape_index = meta['index']
            data = data.to(self.device)
            seen_batch = data.shape[0]
            seen += seen_batch
            if profile:
                timings = {}
                timings, tic =my_timer(batch_tic, "data transfer", timings)
            
            """
            first do an pretrain forward pass with no grad
            """
            with torch.no_grad():
                feats = self.pretrain(data)
                if profile: timings, tic =my_timer(tic, "pretrain fwd", timings)

            """
            set grad to zero before optim iter
            """
            self.optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):

                # fewshot model forward pass
                assert not feats.requires_grad 
                input_data = torch.cat([data.squeeze(1), feats], dim=-1)
                output, preds = self.model(input_data)
                if profile: timings, tic =my_timer(tic, "fewshot fwd", timings)
                
                # cross entropy
                B, N, C = output.shape
                output = output.reshape(-1, C)
                gtlbls = gtlbls.type(torch.LongTensor).to(self.device)
                gtlbls = gtlbls.reshape(-1) ## to make labels start from 0
                loss = self.model.segmentation_loss(output, gtlbls)  
                if profile: timings, tic =my_timer(tic, "loss fwd", timings)
                
                print()
                print(f"======= TRAIN =======\n epoch {epoch} segmentation loss (ce): {loss.item()}")
                
                ## backward pass
                loss.backward()
                if profile: timings, tic =my_timer(tic, "loss-back", timings)

                ## optimize
                self.optimizer.step()
                if profile: timings, tic =my_timer(tic, "optim-step", timings)

                ##########################################################################
                ## IOU evaluation
                # print("preds.shape: ", preds.shape)
                # print("gtlbls.shape: ", gtlbls.shape)
                # print("data.size(0)", data.size(0))
                # gtlbls = gtlbls.reshape(preds.shape)
                # for b in range(B):
                #     mean_shape_IoU, area_intersection, area_union = self.iou_eval.get_mean_IoU(preds[b], gtlbls[b])
                #     mean_shape_IOU_meter.update(mean_shape_IoU.item(), data.size(0))
                #     intersection_meter.update(area_intersection, data.size(0))
                #     union_meter.update(area_union, data.size(0))

                gtlbls = gtlbls.reshape(-1)
                preds = preds.reshape(-1)
                mean_shape_IoU, area_intersection, area_union = self.iou_eval.get_mean_IoU(preds, gtlbls)
                mean_shape_IOU_meter.update(mean_shape_IoU.item())
                intersection_meter.update(area_intersection)
                union_meter.update(area_union)
                
                if batch_idx % 30 == 0:
                    print(f"epoch {epoch} step {batch_idx}")
                print(
                    f"shape_IOU {mean_shape_IoU:.4f} "
                    f"mean shape_IOU {mean_shape_IOU_meter.avg:.4f} "
                    f"mean category {(intersection_meter.avg/union_meter.avg).mean():.4f}")
                if profile: timings, tic =my_timer(tic, "IOU-step", timings)
                ##########################################################################
            
            avg_loss.update(loss.item(), data.size(0))

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                toc = time.time() - seen_tic
                rate = max(seen / toc, 1E-5)
                tic = time.time()
                msg = "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} "
                msg += "Hz: {:.2f}, ETA: {}"
                batches_left = total_batches - batch_idx
                remaining = batches_left * self.data_loader.batch_size / rate
                eta_str = str(datetime.timedelta(seconds=remaining))
                self.logger.info(
                    msg.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        len(self.data_loader.dataset),
                        100.0 * batch_idx / len(self.data_loader),
                        loss.item(),
                        rate,
                        eta_str
                    )
                )
                seen_tic = time.time()
                seen = 0
                if profile:
                    timings["vis"] = time.time() - tic

            del data
            del loss
            del output
            torch.cuda.empty_cache()

            if profile:
                timings, batch_tic =my_timer(batch_tic, "minibatch", timings)
                
                print("==============")
                for key in timings:
                    ratio = 100 * timings[key] / timings["minibatch"]
                    msg = "{:.3f} ({:.2f}%) >>> {}"
                    print(msg.format(timings[key], ratio, key))
                print("==============")

            if self.mini_train and batch_idx > 3:
                self.logger.info("Mini training: exiting epoch early...")
                break

        log = {
            'loss': avg_loss.avg, 
            'mean shape iou:': mean_shape_IOU_meter.avg, 
            "mean category": (intersection_meter.avg/union_meter.avg).mean()
            }

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch - 1)
        return log


    ########################################################################
    #################### validate the segmentation net ########################
    def _segmentation_valid_epoch(self, epoch):
        self.model.eval()
        train_tic = time.time()

        avg_loss = AverageMeter()
        val_mean_shape_IOU_meter = AverageMeter()
        val_intersection_meter = AverageMeter()
        val_union_meter = AverageMeter()

        total_metrics = [AverageMeter() for a in range(len(self.metrics))]
        seen_tic = time.time()
        seen = 0
        profile = self.config["profile"]
        total_batches = len(self.valid_data_loader)

        if profile:
            batch_tic = time.time()

        print(f"val data loader length: {len(self.valid_data_loader)}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):

                if (self.config["evaluation"].get("mini_eval", False) and batch_idx == 15):
                    print("Early stop as mini_eval is used")
                    break

                data, meta = batch["data"], batch["meta"]
                gtlbls = meta["label"]
                shape_index = meta['index']
                gtlbls = gtlbls.to(self.device)
                data = data.to(self.device)
                seen_batch = data.shape[0]
                seen += seen_batch
                if profile:
                    timings = {}
                    timings, tic =my_timer(batch_tic, "data transfer", timings)

                self.model.eval()
                feats = self.pretrain(data)
                if profile: timings, tic =my_timer(tic, "pretrain fwd", timings)

                # fewshot model forward pass
                # print("data shape", data.shape)
                # print("feats shape", feats.shape)
                input_data = torch.cat([data.squeeze(1), feats], dim=-1)
                output, preds = self.model(input_data)
                if profile: timings, tic =my_timer(tic, "fewshot fwd", timings)
                
                # cross entropy
                B, N, C = output.shape
                output = output.reshape(-1, C)
                gtlbls = gtlbls.type(torch.LongTensor).to(self.device)
                gtlbls = gtlbls.reshape(-1) ## to make labels start from 0
                loss = self.model.segmentation_loss(output, gtlbls)  
                if profile: timings, tic =my_timer(tic, "loss fwd", timings)
                print()
                print("======= VAL =======\nsegmentation loss (ce): ", loss.item())

                ##########################################################################
                ## IOU evaluation
                # print("preds.shape: ", preds.shape)
                # print("gtlbls.shape: ", gtlbls.shape)
                # print("data.size(0)", data.size(0))
                # gtlbls = gtlbls.reshape(preds.shape)
                # for b in range(B):
                #     mean_shape_IoU, area_intersection, area_union = self.iou_eval.get_mean_IoU(preds[b], gtlbls[b])
                #     val_mean_shape_IOU_meter.update(mean_shape_IoU.item(), data.size(0))
                #     val_intersection_meter.update(area_intersection, data.size(0))
                #     val_union_meter.update(area_union, data.size(0))
                gtlbls = gtlbls.reshape(-1)
                preds = preds.reshape(-1)
                mean_shape_IoU, area_intersection, area_union = self.iou_eval.get_mean_IoU(preds, gtlbls)
                val_mean_shape_IOU_meter.update(mean_shape_IoU.item())
                val_intersection_meter.update(area_intersection)
                val_union_meter.update(area_union)
                
                if batch_idx % 30 == 0:
                    print(f"epoch {epoch} step {batch_idx}")
                print(
                    f"shape_IOU {mean_shape_IoU:.4f} "
                    f"mean shape_IOU {val_mean_shape_IOU_meter.avg:.4f} "
                    f"mean category {(val_intersection_meter.avg/val_union_meter.avg).mean():.4f}")
                ##########################################################################

            avg_loss.update(loss.item(), data.size(0))

            # if self.verbosity >= 2 and batch_idx % self.log_step == 0:
            if self.verbosity >= 2 and batch_idx % 30 == 0:
                toc = time.time() - seen_tic
                rate = max(seen / toc, 1E-5)
                tic = time.time()
                msg = "Validation Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} "
                msg += "Hz: {:.2f}, ETA: {}"
                batches_left = total_batches - batch_idx
                remaining = batches_left * self.valid_data_loader.batch_size / rate
                eta_str = str(datetime.timedelta(seconds=remaining))
                self.logger.info(
                    msg.format(
                        epoch,
                        batch_idx * self.valid_data_loader.batch_size,
                        len(self.valid_data_loader.dataset),
                        100.0 * batch_idx / len(self.valid_data_loader),
                        loss.item(),
                        rate,
                        eta_str
                    )
                )
                seen_tic = time.time()
                seen = 0
                if profile:
                    timings["vis"] = time.time() - tic

            # Do some aggressive reference clearning to ensure that we don't
            # hang onto memory while fetching the next minibatch (for safety, disabling
            # this for now)
            del data
            del loss
            del output
            torch.cuda.empty_cache()

        log = {
            'eval loss': avg_loss.avg, 
            'eval mean shape iou:': val_mean_shape_IOU_meter.avg,
            "eval mean category": (val_intersection_meter.avg/val_union_meter.avg).mean()
            }
        
        duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - train_tic))
        print(f"training epoch took {duration}")

        return log
