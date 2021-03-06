import numpy as np
import torch
import time
import tqdm
import datetime
from torchvision.utils import make_grid
from pkg_resources import parse_version
# from base import BaseTrainer
from base import BasePCTrainer
from torch.nn.modules.batchnorm import _BatchNorm
# from model.metric import runningIOU
from pointcloud_utils.pointcloud_vis import *
from pointcloud_utils.iou_metric import *
from pointcloud_utils.iou_metric import PointCloudIOU
import os

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


class PCTrainerAll(BasePCTrainer):
    """
    Trainer class

    Note:
        Inherited from BasePCTrainer.
    """

    def __init__(
            self,
            model,
            loss,
            metrics,
            optimizer,
            resume,
            config,
            data_loader_list,
            valid_data_loader=None,
            lr_scheduler=None,
            visualizations=None,
            mini_train=False,
            check_bn_working=False,
            sparse_loss=None,
            sampler_net=None,
            **kwargs,
    ):
        super().__init__(model, loss, metrics, optimizer, resume, config)
        self.config = config
        self.data_loader_list = data_loader_list
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        # self.log_step = max(1, int(len(self.data_loader) / 5.))
        self.log_step = max(1, int(20 / 5.))
        self.visualizations = visualizations if visualizations is not None else []

        ##
        self.sparse_loss = sparse_loss
        if self.sparse_loss is not None:
            """ init sampler net """
            assert sampler_net is not None, "sparse loss with no sampler net"
            self.sampler_net = sampler_net.cuda()
            self.use_sparse_constraint = True
        else:
            self.use_sparse_constraint = False

        self.mini_train = mini_train
        self.check_bn_working = check_bn_working
        self.loss_args = config.get('loss_args', {})

        assert self.lr_scheduler.optimizer is self.optimizer
        assert self.start_epoch >= 1

        if self.start_epoch != 1:
            # Our epoch 1 is step -1 (but we can't explicitly call
            # step(-1) because that would update the lr based on a negative epoch)
            # NB stateful schedulers eg based on loss won't be restored properly
            self.lr_scheduler.step(self.start_epoch - 2)

        # only perform last epoch check for older PyTorch, current versions
        # immediately set the `last_epoch` attribute to 0.
        if parse_version(torch.__version__) <= parse_version("1.0.0"):
            assert self.lr_scheduler.last_epoch == self.start_epoch - 2

        print('Loss args', self.loss_args)

        # Handle segmentation metrics separately, since the accumulation cannot be
        # done directly via an AverageMeter
        self.log_miou = self.config["trainer"].get("log_miou", False)


    # def _eval_metrics(self, output, target):
    #     acc_metrics = np.zeros(len(self.metrics))
    #     for i, metric in enumerate(self.metrics):
    #         acc_metrics[i] += metric(
    #             output,
    #             target,
    #             self.data_loader.dataset,
    #             self.config
    #         )
    #         self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
    #     return acc_metrics

    # def printer(self, msg):
    #     print("{:.3f} >>> {}".format(time.time() - self.tic, msg))

    # def _visualize_examples(self):
    #     """ visualization
    #     batch_vis_idx: batch idx to be visualized; output
    #     """
    #     batch_vis_idx = None
    #     if self.config.get('visualize', False):
    #         num_vis = self.config['visualize']['num_examples']
    #         batch_vis_idx = np.random.choice(len(self.data_loader.dataset)-3, num_vis, replace=False)
    #         batch_vis_idx = np.concatenate([batch_vis_idx+3, [0, 1, 2]]) # avoid replacement
    #         print("batch_vis_idx: ", batch_vis_idx)        
    #     return batch_vis_idx


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
        avg_total_loss = AverageMeter()
        avg_perm_loss = AverageMeter()
        avg_qap_loss = AverageMeter()
        avg_normal_loss = AverageMeter()
        avg_sharp_loss = AverageMeter()

        avg_matchsize = AverageMeter()
        avg_diff_via_recon = AverageMeter()
        avg_ch_dist = AverageMeter()

        seen_tic = time.time()
        profile = self.config["profile"]

        batches_left = 0
        for i in range(len(self.data_loader_list)):
            batches_left+=len(self.data_loader_list[i])

        print("total_batches: ", batches_left)
        
        # """ visualization """
        # pc_visualize = False
        # fname_vis = None
        # batch_vis_idx = self._visualize_examples()
        # # if self.config.get('visualize', False):
        # #     num_vis = self.config['visualize']['num_examples']
        # #     batch_vis_idx = np.random.choice(len(self.data_loader.dataset)-3, num_vis, replace=False)
        # #     batch_vis_idx = np.concatenate([batch_vis_idx+3, [0, 1, 2]]) # avoid replacement
        # #     print("batch_vis_idx: ", batch_vis_idx)
        # #     # batch_vis_idx = np.array([0, 1, 2])

        if profile:
            batch_tic = time.time()

        if True:
            loader_indices = torch.randperm(len(self.data_loader_list))


        # for loader_id, data_loader in enumerate(self.data_loader_list):
        for iter, loader_id in enumerate(loader_indices):

            print()
            print("loader_id ", loader_id)
            print()

            data_loader = self.data_loader_list[loader_id]
            for batch_idx, batch in enumerate(data_loader):
                data, meta = batch["data"], batch["meta"]
                data_indices = meta["index"]
                print(f"index: {data_indices}")
                data = data.to(self.device)
                seen_batch = data.shape[0]
                # print("data.shape in trainer: ", data.shape)
                
                # if batch_vis_idx is not None and np.sum(batch_vis_idx == batch_idx) == 1:
                #     pc_visualize = True
                #     # print("batch_vis_idx: ", batch_vis_idx)
                #     # print("batch_idx: ", batch_idx)
                #     # print(np.sum(batch_vis_idx == batch_idx))

                if profile:
                    timings = {}
                    timings["data transfer"] = time.time() - batch_tic
                    tic = time.time()
                
                """
                set grad to zero before optim iter
                """
                self.optimizer.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    output = self.model(data)
                    if profile: timings, tic =my_timer(tic, "fwd", timings)
                    
                    ## compute loss
                    # if pc_visualize is True:
                    #     parent_dir = self.config.result_dir
                    #     fname_vis = "vis_example_"+str(epoch)+"_"+str(batch_idx)+"_"
                    #     fname_vis = os.path.join(parent_dir, fname_vis)

                    output_loss, output_info = self.loss(output, meta, epoch, fname_vis=None, dictionary=None)
                    ## print loss related values to console
                    for name, iter_loss in output_loss.items():
                        print(name, iter_loss)
                    for name, iter_info in output_info.items():
                        print(name, iter_info)
                    total_loss = output_loss['total_loss']
                    loss=output_loss['cycle_loss']
                    pc_visualize = False
                    fname_vis = None
                    if profile: timings, tic =my_timer(tic, "loss-fwd", timings)
                    ## backward pass
                    total_loss.backward()
                    if profile: timings, tic =my_timer(tic, "loss-back", timings)
                    ## optimize
                    self.optimizer.step()
                    if profile: timings, tic =my_timer(tic, "optim-step", timings)
                
                print(f"loader id {loader_id} batch_idx: {batch_idx}")
                avg_loss.update(loss.item(), data.size(0))
            
                if True:
                    avg_total_loss.update(total_loss.item(), data.size(0))
                    avg_qap_loss.update(output_loss['qap_loss'].item(), data.size(0))
                    avg_perm_loss.update(output_loss['perm_loss'].item(), data.size(0))
                    avg_normal_loss.update(output_loss['normal_loss'].item(), data.size(0))
                    avg_matchsize.update(output_info['correct_match'], data.size(0))
                    avg_diff_via_recon.update(output_info['diff_via_recon'].item(), data.size(0))

                if self.mini_train and batch_idx > 3:
                    self.logger.info("skip...")
                    break

            if self.verbosity >= 2:
                toc = time.time() - seen_tic
                rate = max(len(data_loader) / toc, 1E-5)
                tic = time.time()
                msg = "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} "
                msg += "Hz: {:.2f}, ETA: {}"
                batches_left -= len(data_loader)
                remaining = batches_left * data_loader.batch_size / rate
                eta_str = str(datetime.timedelta(seconds=remaining))
                self.logger.info(
                    msg.format(
                        epoch,
                        len(data_loader)*data_loader.batch_size,
                        len(data_loader.dataset),
                        100.0 * batches_left*data_loader.batch_size / len(data_loader.dataset),
                        loss.item(),
                        rate,
                        eta_str
                    )
                )
                seen_tic = time.time()

            # Do some aggressive reference clearning to ensure that we don't
            # hang onto memory while fetching the next minibatch (for safety, disabling
            # this for now)
            del data
            del loss
            del output
            del total_loss
            output_loss.clear()
            output_info.clear()
            torch.cuda.empty_cache()
        
            if profile:
                timings["minibatch"] = time.time() - batch_tic
                batch_tic = time.time()

                print("==============")
                for key in timings:
                    ratio = 100 * timings[key] / timings["minibatch"]
                    msg = "{:.3f} ({:.2f}%) >>> {}"
                    print(msg.format(timings[key], ratio, key))
                print("==============")

            log = {'loss': avg_loss.avg}
            log = {**log,
                'loader_id': loader_id,
                'total_loss': avg_total_loss.avg, 
                'perm_loss': avg_perm_loss.avg, 
                'qap_loss': avg_qap_loss.avg, 
                'normal_loss': avg_normal_loss.avg, 
                'match_size': avg_matchsize.avg, 
                'diff_via_recon': avg_diff_via_recon.avg,
            }

            if self.mini_train and iter > 3:
                self.logger.info("Mini training: exiting epoch early...")
                break
        
        self.writer.set_step(epoch, 'train_epoch')
        self.writer.add_scalar('loss', log['loss'])

        duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - train_tic))
        print(f"training epoch took {duration}")

        # ## validation
        # val_tic = time.time()
        # if self.do_validation:
        #     val_log = self._valid_epoch(epoch)
        #     log = {**log, **val_log}
        # duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - val_tic))
        # print(f"validation epoch took {duration}")

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch - 1)
        return log

    def _segmentation_valid_epoch(self, epoch):

        self.logger.info(f"Running segmentation validation for epoch {epoch}")
        # input()
        self.model.eval()
        mean_shape_IOU_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()


        n_cls = self.config["dataset"]["num_lbl_classes"]
        n_pts = self.config["dataset"]["args"]["num_points"]
        print("n_cls: ", n_cls)
        iou_eval = PointCloudIOU(n_cls)

        seed = 0
        torch.manual_seed(seed)
        with torch.no_grad():
            """
            select K reference models as a batch
            compute their features
            process to desired shape
            """
            # valdiation time
            tic = time.time()

            K = self.config["evaluation"].get("num_refs", 4)
            temperature = self.config["evaluation"].get("temperature")
            early_stop_num = 15

            rand_indices = torch.randperm(self.valid_data_loader.dataset.size())[0:K]
            self.logger.info(f"auxiliary shape indices: {rand_indices}")
            aux_models = self.valid_data_loader.dataset.get_data_with_indices(rand_indices)
            aux_data, aux_meta = aux_models["data"], aux_models["meta"]
            aux_lbls = aux_meta["label"]
            # compute features
            aux_data = aux_data.to(self.device)
            aux_feat = self.model(aux_data) # model outputs descriptors(desc)
            Ba, N, C = aux_feat.shape
            # print("aux_data shape: ", aux_data.shape)
            # print("aux_feat shape: ", aux_feat.shape)
            # print("aux_lbls shape: ", aux_lbls.shape)
            # print("aux_lbls: ", aux_lbls)

            # write_obj_with_labels("check_model.obj", aux_data[0], aux_lbls[0])

            # process fas to desired shape
            # print("after reshape, ")
            aux_data = aux_data.view(Ba*N, -1)
            aux_feat = aux_feat.contiguous().view(Ba*N, -1)
            aux_lbls = aux_lbls.view(Ba*N, -1)
            # print("aux_data shape: ", aux_data.shape)
            # print("aux_feat shape: ", aux_feat.shape)
            # print("aux_lbls shape: ", aux_lbls.shape)
            # print("aux_lbls max: ", aux_lbls.max())
            # print("aux_lbls min: ", aux_lbls.min())

            # # make data_aux
            def _compute_vis_offset(num_instance, layout="cyclic", distance=1.0, extra_offset=[0.0, 0.0, 0.0]):
                distance = float(distance)
                if layout is "cyclic":
                    rads = np.linspace(0.0, 2.0*np.pi, Ba, endpoint=False)
                    # print("rads.shape: ", rads.shape)
                    multiplier = np.zeros((Ba, 3))
                    cos_rads, sin_rads = np.cos(rads), np.sin(rads)
                    multiplier[:, 1]=sin_rads
                    multiplier[:, 2]=cos_rads
                else:
                    raise "no other layout type is supported for now"
                aux_offsets = torch.from_numpy(multiplier)*distance
                aux_offsets = aux_offsets.type(torch.FloatTensor)
                aux_offsets -= torch.tensor(extra_offset)
                return aux_offsets

            aux_offsets = _compute_vis_offset(num_instance=Ba, layout="cyclic", distance=2.5, extra_offset=[0.5, 0.0, 0.0])
            xyzl_as = aux_data.cpu()
            xyzl_as = xyzl_as.view(Ba, N, -1)
            xyzl_as[:,:,:3] = xyzl_as[:,:,:3] + aux_offsets[:, None, :]
            xyzl_as = xyzl_as.view(Ba*N, -1)
            vis_data_aux = {"xyz": xyzl_as, "gt_lbls": aux_lbls}
            # vis_data_aux = {"xyz": xyzl_as}

            """
            evaluation
            """
            mean_shape_IoU_list = []
            # mean_IoU_dataset = torch.tensor([0.0])
            # cnt = 0
            # vis_idx = rand_indices = torch.randperm(dataset.size())[100:200]

            print("load validation data: ", time.time() - tic)
            tic = time.time()

            for i, batch in enumerate(self.valid_data_loader):
            # for i, batch in enumerate(tqdm(self.valid_data_loader, ncols=80)):
                
                if (self.config["evaluation"].get("mini_eval", False) and i == early_stop_num):
                    print("Early stop as mini_eval is used")
                    break

                data, meta = batch["data"], batch["meta"]
                gt_lbls = meta["label"]
                # print("load data: ", time.time() - tic)
                # tic = time.time()
                # print("data.shape ", data.shape)
                data = data.to(self.device)
                feat = self.model(data) # model outputs descriptors(desc)
                
                assert feat.shape[0] == 1
                feat = feat.squeeze(0)
                gt_lbls = gt_lbls.squeeze(0)
                xyz = data.squeeze(1).squeeze(0).cpu()
                # print("xyz shape: ", xyz.shape)
                # print("feat shape: ", feat.shape)
                # print("gt_lbls shape: ", gt_lbls.shape)
                
                ## pack to correspondence
                pred_lbls, perm_1a = compute_correspondence_from_a_to_1_no_batch(
                    feat, aux_feat, aux_lbls, n_cls, temperature=temperature)
                # mean_IoU_shape, pred_lbls = hard_segmentation_eval(iou_eval, gt_lbls, aux_lbls, perm_1a)
                # pred_lbls = aux_lbls[perm_1a]
                # print("pred_lbls shape: ", pred_lbls.shape)

                mean_shape_IoU, area_intersection, area_union = iou_eval.get_mean_IoU(pred_lbls, gt_lbls)
                mean_shape_IoU_list.append(mean_shape_IoU)
                mean_shape_IOU_meter.update(mean_shape_IoU.item(), data.size(0))
                intersection_meter.update(area_intersection, data.size(0))
                union_meter.update(area_union, data.size(0))

                if mean_shape_IOU_meter.count % 50 == 0:
                    print(f"after {mean_shape_IOU_meter.count} shapes: shape mean_IoU = {mean_shape_IOU_meter.avg}")
                    print(f"after {intersection_meter.count} shapes: category mean_IoU = {(intersection_meter.avg/union_meter.avg).mean()}")

                if mean_shape_IOU_meter.count % 10 == 0:
                    vis_data_query = {"xyz": xyz, "gt_lbls": gt_lbls, "pred_lbls": pred_lbls}
                    parent_dir = os.path.join(self.config.result_dir, "segmentation")
                    if not os.path.exists(parent_dir):
                        os.mkdir(parent_dir)
                    fname_vis = f"rnd_seed{seed}-shape{i}.obj"
                    fname_vis = os.path.join(parent_dir, fname_vis)

                    idx = torch.arange(0, perm_1a.shape[0])
                    pair = torch.stack((perm_1a.cpu(),idx), dim=0)
                    pair = pair.permute(1,0)
                    write_paired_pointclouds_obj(fname_vis, vis_data_aux, pair, vis_data_query)

            print(f"check validation data {early_stop_num}: {time.time() - tic}")
            print(f"Segmentation Validation:\n --shape mean_IoU = {mean_shape_IOU_meter.avg} ")
            print(f"--category mean_IoU = {(intersection_meter.avg/union_meter.avg).mean()} with {intersection_meter.count} shapes")
            val_log = {"shape mean_IoUe": mean_shape_IOU_meter.avg}
            val_log = {"category mean_IoU": (intersection_meter.avg/union_meter.avg).mean()}
        return val_log

    def _correspondence_valid_epoch(self, epoch):
        self.logger.info(f"Running correspondence validation for epoch {epoch}")
        self.model.eval()
        match_rate_meter = AverageMeter()
        n_pts = self.config["dataset"]["args"]["num_points"]
        seed = 0
        torch.manual_seed(seed)
        early_stop_num = 100
        with torch.no_grad():
            tic = time.time()
            for i, batch in enumerate(self.valid_data_loader):   
                if (self.config["evaluation"].get("mini_eval", False) and i == early_stop_num):
                    print("Early stop as mini_eval is used")
                    break
                ## load
                data, meta = batch["data"], batch["meta"]
                gt_lbls = meta["label"]
                data = data.to(self.device)
                feat = self.model(data) # model outputs descriptors(desc)

                ## compute loss
                output_loss, output_info = self.loss(feat, meta, epoch, fname_vis=fname_vis)
                for name, iter_loss in output_loss.items():
                    print(name, iter_loss)
                for name, iter_info in output_info.items():
                    print(name, iter_info)
                total_loss = output_loss['total_loss']
                loss=output_loss['cycle_loss']
                val_log = {**output_loss, **output_info}
        return val_log

    