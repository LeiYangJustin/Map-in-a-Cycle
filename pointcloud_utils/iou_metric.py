import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# import pointnet3.sinkhorn_approximate as sinkFunc
import pointnet3.arch.helper_functions as hf

## make permutation matrix
def _get_partial_perm(corrNK):
    assert len(corrNK.shape)==2
    N,K = corrNK.shape
    # print("corrNK.shape ", corrNK.shape)
    assert N <= K
    maxID = torch.argmax(corrNK, dim=1).view(-1)
    # topk_val, topk_id = torch.topk(corrNK, k=1, dim=1)
    # print(f"topk_val {topk_val}, topk_id {topk_id}")
    # print(f"max topk_val {topk_val.max()} min topk_val {topk_val.min()}")
    # print("maxID shape:", maxID.shape)
    return maxID

def compute_correspondence_from_a_to_1_no_batch(f1, fas, labels, num_classes, 
    temperature, use_hard_correspondence=True, use_sinkhorn=False):
    """
    f1: per-point features of the query point set
    fas: per-point features of the collection of the point sets with known labels
    temperature: a hyperparameter for softmax normalization; 
                 this is required to be the same as set for training
    """

    # permute to CxN from NxC
    N, C = f1.shape
    f1 = f1.permute(1,0)
    fas = fas.permute(1,0)

    # print("f1.shape: ", f1.shape)
    # print("fas.shape: ", fas.shape)

    corr_1a = torch.matmul(f1.t(), fas) / temperature 
    smcorr_1a = F.softmax(corr_1a, dim=1) # checked
    
    # if use_sinkhorn:
    #     assert smcorr_1a.shape[0] == smcorr_1a.shape[1]
    #     smcorr_1a_sink, _ = sinkFunc.gumbel_sinkhorn(
    #         corr_1a, temp=0.3, n_iters=30)
    #     if smcorr_1a_sink.shape[0]==1:
    #         smcorr_1a_sink = smcorr_1a_sink.squeeze(0) 
    #     else:
    #         smcorr_1a_sink = torch.mean(smcorr_1a_sink, dim=0)
    #     smcorr_1a = smcorr_1a_sink

    perm_1a = _get_partial_perm(smcorr_1a) # 1d indices
    if use_hard_correspondence:
        pred_lbls = labels[perm_1a]
        return pred_lbls, perm_1a
    else:
        # return smcorr_1a
        # print("labels.shape", labels.shape)
        indicator = torch.ones(labels.shape)
        # print("f1.shape", f1.shape)
        pred_lbls = torch.zeros((N, 1)).type(torch.LongTensor)
        score_pred = torch.zeros((N, 1))-1.0
        for i in range(num_classes): 
            score_i = torch.matmul(smcorr_1a, indicator*(labels==i))
            # print(score_i.shape)
            idx = score_i > score_pred
            score_pred[idx] = score_i[idx]
            pred_lbls[idx] = i
        return pred_lbls, perm_1a


def vote_label_topk(Q, Rs, R_lbls, temperature, num_classes, k=3, weight=None):
    """
    Q is the query model; size (1, N, D)
    Rs are a set of reference models; size (B, N, D)
    R_lbls are a set of labels corresponding to the reference models; size (B, N, 1)
    """
    device = Q.device
    if len(Rs.shape)==2:
        Rs = Rs.unsqueeze(0)

    B, N, D = Rs.shape
    if len(R_lbls.shape)==2 or len(R_lbls.shape)==1:
        R_lbls = R_lbls.view(B, N, 1)

    ## correlation
    Q_repeat = Q.repeat(B, 1, 1)
    score = F.softmax(Q_repeat.bmm(Rs.transpose(2, 1))/temperature, dim=2)
    top_k_score, top_k_index = torch.topk(score, k=k, dim=2)

    if weight is not None:
        top_k_score = top_k_score*weight[:,None,None]
    top_k_labels = hf.index_points(R_lbls, top_k_index.reshape(B, -1))
    top_k_labels = top_k_labels.reshape(B, N, k)

    ## vote
    vote_score = top_k_score.permute(1, 0, 2).reshape(N, -1)
    vote_labels = top_k_labels.permute(1, 0, 2).reshape(N, -1)

    Q_lbls = torch.zeros(N, num_classes)
    O = torch.zeros(vote_score.shape).to(device)
    for cls_lbl in range(num_classes):
        cls_score = torch.where(vote_labels == cls_lbl, vote_score, O)
        Q_lbls[:, cls_lbl] = cls_score.sum(dim=-1)
    Q_lbls = F.softmax(Q_lbls, dim=-1) ## per part probability
    return Q_lbls

class PointCloudIOU():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        # self.has_minus_one = has_minus_one
        # assert method in ["mean", "weighted"], "method should be a str type var accepting only 'mean' or 'weighted'"
        # self.method = method
        print(f"num_classes: {self.num_classes}")
        # self.fixed_weight = 1.0/self.n_classes

    def get_mean_IoU(self, pred_lbls, gt_lbls):
        gt_lbls= gt_lbls.detach().cpu().numpy() 
        pred_lbls= pred_lbls.detach().cpu().numpy()
        # mean_iou =torch.tensor([0.0])
        # for part_id in range(self.n_classes):
        #     # print(part_id)
        #     part_iou = self._get_part_mIoU(part_id, pred_lbls, gt_lbls)
        #     print(f"part_iou {part_iou}")
        #     mean_iou += (part_iou * self.fixed_weight)
        # print("pred_lbls.shape ", pred_lbls.shape)
        # print("gt_lbls.shape ", gt_lbls.shape)
        area_intersection, area_union = self.intersectionAndUnion(pred_lbls, gt_lbls)
        mean_iou = ((area_intersection+1e-10)/(area_union+1e-10)).mean()
        # print(f"I {area_intersection} / U {area_union} = {area_intersection/area_union} mean-> {mean_iou}" )
        return mean_iou, area_intersection, area_union
        
    # https://github.com/CSAILVision/semantic-segmentation-pytorch/utils.py
    def intersectionAndUnion(self, pred_lbls, gt_lbls):

        assert pred_lbls.shape == gt_lbls.shape
        imPred = np.asarray(pred_lbls).copy()
        imLab = np.asarray(gt_lbls).copy()
        numClass = self.num_classes

        imPred += 1
        imLab += 1
        # Remove classes from unlabeled pixels in gt image.
        # We should not penalize detections in unlabeled portions of the image.
        imPred = imPred * (imLab > 0)

        # Compute area intersection:
        intersection = imPred * (imPred == imLab)
        (area_intersection, _) = np.histogram(
            intersection, bins=numClass, range=(1, numClass))

        # Compute area union:
        (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
        (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
        area_union = area_pred + area_lab - area_intersection
        # print("area_pred: ", area_pred)
        # print("area_lab: ", area_lab)
        # print("area_intersection: ", area_intersection)
        # print("area_union: ", area_union)
        return (area_intersection, area_union)
