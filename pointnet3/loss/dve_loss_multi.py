import torch.nn.functional as F
import torch.nn as nn
import time
import torch
from utils import tps
from PIL import Image
import numpy as np
from pointcloud_utils import pointcloud_vis
import pointnet3.sinkhorn_approximate as sinkFunc
from pointnet3 import Local_Match_Consensus_loss_qap as LMC
# from pointnet3 import chamfer_distance_without_batch

def save_data_as_image(filename, data):
    min_val = np.amin(data)
    max_val = np.amax(data)
    # RESCALING THE DATA to 0 255
    img = (data - min_val) / (max_val-min_val) * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img, 'P')
    img.save(filename)


def compute_entropy(A, dim, reduction=None, is_prob=True):
    assert(len(A.shape)==2) # A is a 2D correlation matrix
    assert reduction in [None, "mean", "sum"]

    # to probability
    if not is_prob:
        A = F.softmax(A, dim=dim)

    # compute entropy
    ent = -1.0 * torch.sum(A * torch.log(A+1e-12), dim=dim)
    
    # reduced
    if reduction is None:
        return ent
    elif reduction == "mean":
        return ent.sum()/A.shape[dim]
    elif reduction == "sum":
        return ent.sum()


def feature_reconstruction(f1, f2, temperature):
    N = f1.shape[1]
    # 1 - compute cosine similarity between f1 and f2
    # corr = f1.permute(0,2,1).bmm(f2)/temperature
    corr = torch.matmul(f1.t(), f2) / temperature # f1 in CxN, f2 in CxM, corr in NxM 
    # 2 - making smcorr a weight matrix (smcorr[i, :].sum() = 1)
    # dim=0 -> f1, dim=1 -> f2
    smcorr = F.softmax(corr, dim=-1); del corr
    # 3 - reconstruction
    # f1_via_f2 = f2.bmm(smcorr.permute(0,2,1)); del smcorr
    f1_via_f2 = torch.matmul(f2, smcorr.t()); del smcorr
    # 4 - cosine similarity between f1 and its reconstruction
    # corr = f1_via_f2.permute(0,2,1).bmm(f1)
    corr = torch.matmul(f1_via_f2.t(), f1)
    # log_smcorr = F.log_softmax(corr, dim=1); del corr
    smcorr = F.softmax(corr, dim=-1); del corr
    # get diagonal of corr
    d = torch.diagonal(smcorr)
    d = d.sum()/N*-1.0
    return f1_via_f2, d


class DVE_loss_multi(nn.Module):
    def __init__(
        self,
        pow=0.5, fold_corr=False, normalize_vectors=True, 
        temperature=1.0,
        recursive=False,
        # fname_vis=None, train=False,
        sink_tau=[0.7,0.7], sink_iters=[20,20],
        lambda_lc=0.01, lambda_qap=0.01,
        local_args=None
        ):
        super(DVE_loss_multi, self).__init__()
        
        self.fold_corr = fold_corr
        self.normalize_vectors=normalize_vectors
        self.temperature = temperature
        self.sink_tau = sink_tau
        self.sink_iters = sink_iters

        self.lambda_lc = lambda_lc
        self.lambda_qap = lambda_qap

        print(f"temperature {temperature} sink_tau {sink_tau}, sink_iters {sink_iters}")
        
        """
        dist_thres=0.2, 
        _debug=False, 
        mode="distance", 
        loss_mode="L2"
        """
        if local_args is not None:
            print("local args: ", local_args)
            self.qap_loss = LMC(**local_args)
        else:
            self.qap_loss = LMC()


    ## multi auxiliary forward
    def forward(self, feats, meta, epoch, step=0, fname_vis=None, dictionary=None):
        print("--------dense_correlation_loss_combined_sink--------")
        device = feats.device
        X1 = meta['pc0'].to(device)
        # X1 = meta['pc1'].to(device)
        # X2 = meta['pc2'].to(device)
        feats1 = feats[0::2]
        feats2 = feats[1::2] # deformed
        B, N, C = feats1.shape
        
        
        num_vis = 1 if B > 3 else B
        if fname_vis is not None:
            vis_idx = np.random.choice(B, num_vis, replace=False)
        
        
        # parameters
        loss = 0.
        correct_match = 0.
        diff_via_recon = 0.
        f_recon = 0.0
        Lc = 0.0
        perm_to_I = 0.0
        kl_loss = 0.0
        ch_dist = 0.0
        
        concentrated_entropy=0.0
        divergent_entropy=0.0
        sink_concentrated_entropy=0.0
        sink_divergent_entropy=0.0

        M = 3
        assert B > 5

        for b in range(B):
            # C X N
            f1 = feats1[b].permute(1,0)  # source to B, C, N
            # f1 = f1.unsqueeze(0) # [1, C, N]
            f2 = feats2[b].permute(1,0)  # target
            # f2 = f2.unsqueeze(0)

            ##
            # fa = feats1[(b+1):(b+1+M)%B].permute(0,2,1) # auxiliary, size B, N, C
            fa_list = [feats1[(b+1+m)%B] for m in range(M)]
            fa = torch.cat(fa_list, dim=0).permute(1,0)
            # print("fa.shape ", fa.shape)
            ##
            
            # scale = 5.0
            # if self.normalize_vectors:
            #     f1 = F.normalize(f1, p=2, dim=0) * scale
            #     f2 = F.normalize(f2, p=2, dim=0) * scale
            #     fa = F.normalize(fa, p=2, dim=0) * scale
            
            # ## rotational equivariance constraint
            # f_diff = torch.sum((f1-f2)*(f1-f2), dim=0)
            # f_recon += torch.mean(f_diff)
            
            # ## f1 && fa correlation
            corr_1a = torch.matmul(f1.t(), fa)/self.temperature # f1 in 3xN, f2 in 3xM, corr in NxM 
            # print("corr_1a.shape ", corr_1a.shape)
            smcorr_1a = F.softmax(corr_1a, dim=-1)

            # ## f1 reconstructed by fa
            # f1_via_fa_t = smcorr_1a.bmm(fa.permute(0,2,1))
            # # print("f1_via_fa_t.shape ", f1_via_fa_t.shape)
            # corr_1a2 = f1_via_fa_t.bmm(f2.repeat(M, 1, 1))/self.temperature ## [M, N, N]
            # corr_1a2, _ = corr_1a2.max(dim=0) ## new from MxNxN -> 1xNxN
            # # print("corr_1a2.shape ", corr_1a.shape)
            f1_via_fa_t = torch.matmul(smcorr_1a, fa.t())
            corr_1a2 = torch.matmul(f1_via_fa_t, f2)/self.temperature
            smcorr_1a2 = F.softmax(corr_1a2, dim=-1)
            # print("smcorr_1a2 ", smcorr_1a2)
            

            with torch.no_grad():
                diff = X1[b, :, None, :] - X1[b, None, :, :]
                diff = (diff * diff).sum(2).sqrt()
                diff = diff.pow(0.5) # make distance more sharper
                # smcorr_1a2_sink, _ = sinkFunc.gumbel_sinkhorn(
                #     smcorr_1a2, temp=self.sink_tau[1], n_iters=self.sink_iters[1], normalized=True)
                smcorr_1a2_sink, _ = sinkFunc.gumbel_sinkhorn(
                    corr_1a2, temp=self.sink_tau[1], n_iters=self.sink_iters[1]); del corr_1a2
                if smcorr_1a2_sink.shape[0]==1:
                    smcorr_1a2_sink = smcorr_1a2_sink.squeeze(0) 
                else:
                    smcorr_1a2_sink = torch.mean(smcorr_1a2_sink, dim=0)
            # print(smcorr_1a2_sink)

            # ## additional loss
            # # kl_div_score = LMC.Local_Match_Consensus_loss(X1[b], X1[(b+1)%B], smcorr_1a)
            # kl_div_score = self.qap_loss(X1[b], X1[(b+1)%B], smcorr_1a)
            # kl_loss += 10.0*kl_div_score
            # # print(f"QAP loss: {kl_div_score}")
            
            # dve loss smcorr_1a2
            L = diff * smcorr_1a2 ## (N x N) (N x N)
            a = L.sum()/N
            weight=torch.log(1.0/a.detach())

            # constraint to permutation
            ## for reference
            perm_to_I += 3.0*F.l1_loss(torch.eye(N).cuda(), smcorr_1a2_sink, reduction='sum')/N

            Lc_b = F.l1_loss(smcorr_1a2_sink, smcorr_1a2, reduction='sum')/N
            Lc += 3.0*Lc_b
            # Lc_b_1a = F.l1_loss(smcorr_1a_sink, smcorr_1a, reduction='sum')/N; del smcorr_1a
            # Lc += 3.0*Lc_b_1a

            ## w/o dve loss: this is to learn rotational invariance
            # corr_12 = f1.permute(0,2,1).bmm(f2)/self.temperature
            # corr_12 = corr_12.squeeze(0)
            corr_12 = torch.matmul(f1.t(), f2)/self.temperature
            smcorr_12 = F.softmax(corr_12, dim=-1); del corr_12
            L12 = diff * smcorr_12
            L += 0.5*L12

            ## chamfer distance loss
            if dictionary is not None:
                weight_inds = F.softmax(torch.matmul(dictionary, f1)/self.temperature, dim=-1)
                soft_induced_pts = torch.matmul(weight_inds, X1[b])
                # print(X1[b].unsqueeze(0).shape)
                # print(soft_induced_pts.shape)
                ch_dist_l2r = chamfer_distance_without_batch(soft_induced_pts, X1[b].unsqueeze(0), debug=False)
                ch_dist_r2l = chamfer_distance_without_batch(X1[b].unsqueeze(0), soft_induced_pts, debug=False)
                ch_dist += (ch_dist_r2l + ch_dist_l2r)
                # print("both_side chamfer dist: ", ch_dist)


            ## final loss
            loss += (L.sum()/N)
            # print(f"Loss: {L.sum()/N}, smcorr1a2 to perm: {Lc_b}")
            print(f"Loss: {L.sum()/N}, Loss 12: {L12.sum()/N}, smcorr1a2 to perm: {Lc_b}")

            ## record & check
            with torch.no_grad():
                ## f1 f1 correlation
                _, d = feature_reconstruction(f1, f1, self.temperature)
                # print(f"recon: {d}")
                diff_via_recon += d

                # ## f1 fa correlation
                # corr_1a2, max_idx_corr_1a2
                # corr2 = torch.matmul(f1_via_fa_t, f2)/self.temperature
                # smcorr2 = F.softmax(corr_1a2, dim=1)
                max_idx = torch.argmax(smcorr_1a2, dim=1)
                count = 0.0
                tol_count = 0.0
                for i, max_id in enumerate(max_idx):
                    # if diff[max_id, i] < 0.27: tol_count+=1
                    if max_id == i: count += 1
                correct_match += count

                if fname_vis is not None and np.sum(vis_idx==b) == 1:
                    # matrices as images
        
                    # txt_fname = fname_vis+str(b) + "diff.png"
                    # npdata = diff.cpu().detach().numpy()
                    # save_data_as_image(txt_fname, npdata)
                    
                    txt_fname = fname_vis+str(b) + "smcorr_1a2.png"
                    npdata = smcorr_1a2.cpu().detach().numpy()
                    save_data_as_image(txt_fname, npdata)
                    
                    txt_fname = fname_vis+str(b) + "smcorr_1a2_sink.png"
                    npdata = smcorr_1a2_sink.cpu().numpy()
                    save_data_as_image(txt_fname, npdata)

                    # ## 3D model
                    # filename = fname_vis+str(b)+".obj"
                    # pair = np.stack((np.array(range(max_idx.shape[0])), max_idx.cpu()), axis=1)
                    # data1 = {"xyz": X1[b].cpu()}
                    # data2 = {"xyz": X2[b].cpu()}
                    # pointcloud_vis.write_paired_pointclouds_obj(filename, data1, pair, data2)
                    # del data1, data2
                    print("saved files")

                del diff

        print("--------LOSS with DVE: {}--------".format(loss/B))
        total_loss = (loss + self.lambda_lc*Lc + self.lambda_qap*kl_loss + 3.0*ch_dist)

        print("epoch", epoch)
        # print("loss ", loss/B)
        # print("Lc ", Lc/B)
        # print("kl_loss ", kl_loss/B)
        # print("total_loss ", total_loss/B)
        # print(f"concentrated_entropy {concentrated_entropy}")
        # print(f"divergent_entropy {divergent_entropy}")
        # print(f"sink_concentrated_entropy {sink_concentrated_entropy}")
        # print(f"sink_divergent_entropy {sink_divergent_entropy}")
        print(f"smcorr_to_I {perm_to_I/B}")

        output_loss = {
            'total_loss': total_loss/B,
            'cycle_loss': loss/B, 
            'perm_loss': Lc/B, 
            'qap_loss': kl_loss/B, 
            'ch_dist': ch_dist/B
        }
        output_info = {
            'correct_match': correct_match/B, 
            'diff_via_recon': diff_via_recon/B
        }
        return output_loss, output_info
