import torch.nn.functional as F
import torch.nn as nn
import time
import torch
# from utils import tps
from PIL import Image
import numpy as np
from pointcloud_utils import pointcloud_vis
import pointnet3.sinkhorn_approximate as sinkFunc
from pointnet3 import Local_Match_Consensus_loss_qap as LMC
# from pointnet3 import chamfer_distance_without_batch


def estimate_mem(x):
    if x.dtype == torch.float32:
        nbytes = 4
    elif x.dtype == torch.float16:
        nbytes = 2
    elif x.dtype == torch.int8:
        nbytes = 1
    else:
        import ipdb; ipdb.set_trace()
    return torch.numel(x) * nbytes / (1024) ** 3


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
    corr = torch.matmul(f1.t(), f2) / temperature # f1 in CxN, f2 in CxM, corr in NxM 
    # 2 - making smcorr a weight matrix (smcorr[i, :].sum() = 1)
    # dim=0 -> f1, dim=1 -> f2
    smcorr = F.softmax(corr, dim=1); del corr
    # 3 - reconstruction
    # f1_via_f2_t = torch.matmul(smcorr, f2.t()); del smcorr
    f1_via_f2 = torch.matmul(f2, smcorr.t()); del smcorr
    # 4 - cosine similarity between f1 and its reconstruction
    corr = torch.matmul(f1_via_f2.t(), f1)
    # log_smcorr = F.log_softmax(corr, dim=1); del corr
    smcorr = F.softmax(corr, dim=1); del corr
    # get diagonal of corr
    d = torch.diagonal(smcorr)
    d = d.sum()/N*-1.0
    return f1_via_f2, d


def compute_mask(X, Y):
    """
    X, Y: xyz size N x C
    """
    D = X[:, None, :] - Y[None, :, :]
    D = (D * D).sum(2).sqrt()
    nX, nY = D.shape
    # print(D.shape)
    # D = D.pow(0.5) # make distance more sharper
    temperature = 0.005
    minX, _ = D.min(dim=1)
    # print("minX max: ", minX.max())
    # print("minX sum: ", torch.sum(minX))
    # minX = minX/torch.sum(minX)
    # print("minX normalized max: ", minX.max())

    minX = F.softmax(minX/temperature,dim=0)
    # print("minX softmax max: ", minX.max())
    # print("minX softmax sum: ", torch.sum(minX))

    minY, _ = D.min(dim=0)
    minY = F.softmax(minY/temperature,dim=0)
    # print("minY softmax max: ", minY.max())
    # print("minY softmax sum: ", torch.sum(minY))

    minX = minX.view(-1,1)
    minY = minY.view(1,-1)
    mask_XY_comp = minX * minY
    mask_XY = 1.0 - mask_XY_comp
    # print("mask_XY_comp max: ", mask_XY_comp.max())
    # print("mask_XY: ", mask_XY)
    # input()
    # print(mask_XY.max())
    return mask_XY, mask_XY_comp


class DVE_loss_2edge(nn.Module):
    def __init__(
        self,
        pow=0.5, fold_corr=False, normalize_vectors=True, 
        temperature=1.0,
        # fname_vis=None, train=False,
        sink_tau=[0.3,0.3], sink_iters=[30,30],
        lambda_lc=0.0, lambda_qap=0.0, lambda_ln=0.0,
        local_args=None
        ):
        super(DVE_loss_2edge, self).__init__()
        
        self.fold_corr = fold_corr
        self.normalize_vectors=normalize_vectors
        self.temperature = temperature
        self.sink_tau = sink_tau
        self.sink_iters = sink_iters
        self.lambda_lc = lambda_lc
        self.lambda_qap = lambda_qap
        self.lambda_ln = lambda_ln
        
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

    def forward(self, feats, meta, epoch, step=0, fname_vis=None, dictionary=None):
        print("--------dense_correlation_loss_combined_sink--------")
        device = feats.device
        X1 = meta['pc0'].to(device)
        # X1 = meta['pc1'].to(device)
        # X2 = meta['pc2'].to(device)
        if X1.shape[2] > 3:
            # print(X1.shape)
            N1 = X1[:,:,3:]
            X1 = X1[:,:,:3]
            has_normal = True

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
        normal_loss =0.0
        
        concentrated_entropy=0.0
        divergent_entropy=0.0
        sink_concentrated_entropy=0.0
        sink_divergent_entropy=0.0

        for b in range(B):
            # C X N
            f1 = feats1[b].permute(1,0)  # source to B, C, N
            
            f2 = f1
            #f2 = feats2[b].permute(1,0)  # target
            
            fa = feats1[(b+1)%B].permute(1,0) # auxiliary

            """
            shape diff can be large yet mask_comp max is small. 
            this is because shape can be different in the whole sense.
            when this is the case, the mask_comp max is small
            because this shape difference makes no difference in point level
            mask_comp should find the outliers
            """
            # print("shape diff ", ch_dist_pts) 
            ## rotational equivariance constraint
            f_diff = torch.sum((f1-f2)*(f1-f2), dim=0)
            f_recon += torch.mean(f_diff)
            
            ## f1 && fa correlation
            corr_1a = torch.matmul(f1.t(), fa)/self.temperature # f1 in 3xN, f2 in 3xM, corr in NxM 
            smcorr_1a = F.softmax(corr_1a, dim=1)

            ## f1 reconstructed by fa
            f1_via_fa_t = torch.matmul(smcorr_1a, fa.t())
            corr_1a2 = torch.matmul(f1_via_fa_t, f2)/self.temperature
            smcorr_1a2 = F.softmax(corr_1a2, dim=1)

            # print(smcorr_1a2)
            with torch.no_grad():
                smcorr_1a2_sink, _ = sinkFunc.gumbel_sinkhorn(
                    corr_1a2, temp=self.sink_tau[1], n_iters=self.sink_iters[1]); del corr_1a2
                smcorr_1a_sink, _ = sinkFunc.gumbel_sinkhorn(
                    smcorr_1a, temp=self.sink_tau[1], n_iters=self.sink_iters[1], normalized=True); del corr_1a

                if smcorr_1a2_sink.shape[0]==1:
                    smcorr_1a2_sink = smcorr_1a2_sink.squeeze(0) 
                else:
                    smcorr_1a2_sink = torch.mean(smcorr_1a2_sink, dim=0)
                diff = X1[b, :, None, :] - X1[b, None, :, :]
                # print("diff[:,:,6] ", diff[:,:,6])
                dist = (diff * diff).sum(2).sqrt()
                dist = dist.pow(0.5) # make distance more sharper
                # print("dist: ", dist)
                if has_normal:
                    # print("N1.shape: ", N1.shape)
                    # n = N1[b, 0, :]
                    # print("normal vector ", n)
                    # print(n.shape)
                    # print(torch.matmul(n,n.t()))
                    normal_dist = 1.0 - torch.abs(torch.matmul(N1[b], N1[b].t()))
                    # print("normal_dist ", normal_dist)
                    # input()
                    # normal_dist = normal_dist.pow(0.5)
                    print("normal dist min: ", normal_dist.min())

            ## additional loss
            # kl_div_score = LMC.Local_Match_Consensus_loss(X1[b], X1[(b+1)%B], smcorr_1a)
            kl_div_score = self.qap_loss(X1[b], X1[(b+1)%B], smcorr_1a)
            kl_loss += 1.0*kl_div_score
            # print(f"QAP loss: {kl_div_score}")
            
            # dve loss smcorr_1a2
            L = dist * smcorr_1a2
            a = L.sum()/N
            weight=torch.log(1.0/a.detach())

            # constraint to permutation
            ## for reference
            perm_to_I += 3.0*F.l1_loss(torch.eye(N).to(device), smcorr_1a2_sink, reduction='sum')/N
            constraint_to_perm = "1a2_perm"
            if constraint_to_perm == "1a2_perm":
                Lc_b = F.l1_loss(smcorr_1a2_sink, smcorr_1a2, reduction='sum')/N
            elif constraint_to_perm == "1a2_identity":
                Lc_b = F.l1_loss(torch.eye(N).to(device), smcorr_1a2, reduction='sum')/N
                print("constraint smcorr_1a2_sink to identity")
            elif constraint_to_perm == "1a_perm":
                Lc_b = F.l1_loss(smcorr_1a_sink, smcorr_1a, reduction='sum')/N; del smcorr_1a
                print("constraint smcorr_1a_sink to perm")
            Lc += 3.0*Lc_b
            # Lc_b = F.l1_loss(smcorr_1a_sink, smcorr_1a, reduction='sum')/N; del smcorr_1a
            # Lc += 3.0*Lc_b

            ## w/o dve loss
            corr_12 = torch.matmul(f1.t(), f2)
            smcorr_12 = F.softmax(corr_12/self.temperature, dim=1); del corr_12
            L12 = dist * smcorr_12
            if True:
                L += 1.0*L12
            else:
                print("no rotational invariance constraint")

            if has_normal:
                L_n = normal_dist * smcorr_1a2
                L_12n = normal_dist * smcorr_12
                L_n += 1.0*L_12n
            
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
            normal_loss += (L_n.sum()/N)
            # print(f"Loss: {L.sum()/N:.6f}, Loss 12: {L12.sum()/N:.6f}, mask_penalty {mask_penalty_b}, smcorr1a2 to perm: {Lc_b:.6f}")
            print(f"Loss: {L.sum()/N:.6f}, Loss 12: {L12.sum()/N:.6f}, smcorr1a2 to perm: {Lc_b:.6f}")
            print(f"Normal Loss: {L_n.sum()/N:.6f}")  
            # print(f"sharp_constraint Loss: {sharp_constraint.sum()/N:.6f}")  


            ## record & check
            with torch.no_grad():
                ## f1 f1 correlation
                _, d = feature_reconstruction(f1, f1, self.temperature)
                # print(f"recon: {d}")
                diff_via_recon += d

                # ## f1 fa correlation
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
        total_loss = (loss + self.lambda_ln*normal_loss + self.lambda_lc*Lc + self.lambda_qap*kl_loss)

        print("epoch", epoch)
        print("loss ", loss/B)
        print("Lc ", Lc/B)
        print("kl_loss ", kl_loss/B)
        print("total_loss ", total_loss/B)
        # print(f"concentrated_entropy {concentrated_entropy}")
        # print(f"divergent_entropy {divergent_entropy}")
        # print(f"sink_concentrated_entropy {sink_concentrated_entropy}")
        # print(f"sink_divergent_entropy {sink_divergent_entropy}")

        output_loss = {
            'total_loss': total_loss/B,
            'cycle_loss': loss/B, 
            'perm_loss': Lc/B, 
            'qap_loss': kl_loss/B, 
            'ch_dist': ch_dist/B,
            'smcorr_to_I': perm_to_I/B,
            'normal_loss': normal_loss/B
        }
        output_info = {
            'correct_match': correct_match/B, 
            'diff_via_recon': diff_via_recon/B
        }
        return output_loss, output_info
        # return total_loss/B, loss/B, Lc/B, kl_loss/B, correct_match/B, diff_via_recon/B


    """
    this is a old version of forward function that allows recursive 
    """
    '''
    def forward(self, feats, meta, epoch, step=0, fname_vis=None, ):
        print("--------dense_correlation_loss_combined_sink--------")
        device = feats.device
        X1 = meta['pc0'].to(device)
        # X1 = meta['pc1'].to(device)
        # X2 = meta['pc2'].to(device)
        feats1 = feats[0::2]
        feats2 = feats[1::2] # deformed
        B, N, C = feats1.shape

        # parameters
        loss = 0.
        correct_match = 0.
        diff_via_recon = 0.
        f_recon = 0.0
        Lc = 0.0
        kl_loss = 0.0
        
        concentrated_entropy=0.0
        divergent_entropy=0.0
        sink_concentrated_entropy=0.0
        sink_divergent_entropy=0.0

        for b in range(B):
            # C X N
            f1 = feats1[b].permute(1,0)  # source to B, C, N
            f2 = feats2[b].permute(1,0)  # target
            fa = feats1[(b+1)%B].permute(1,0) # auxiliary

            scale = 5.0
            if self.normalize_vectors:
                f1 = F.normalize(f1, p=2, dim=0) * scale
                f2 = F.normalize(f2, p=2, dim=0) * scale
                fa = F.normalize(fa, p=2, dim=0) * scale
            
            ## rotational equivariance constraint
            f_diff = torch.sum((f1-f2)*(f1-f2), dim=0)
            f_recon += torch.mean(f_diff)
            # print("rot equivariance constraint f_diff: ", f_diff)
            
            ## f1 && fa correlation
            corr_1a = torch.matmul(f1.t(), fa)/self.temperature # f1 in 3xN, f2 in 3xM, corr in NxM 
            smcorr_1a = F.softmax(corr_1a, dim=1)
            # smcorr_1a_sink, _ = sinkFunc.gumbel_sinkhorn(
            #     corr_1a, temp=self.sink_tau[0], n_iters=self.sink_iters[0]); del corr_1a
            # if smcorr_1a_sink.shape[0]==1:
            #     smcorr_1a_sink = smcorr_1a_sink.squeeze(0) 
            # else:
            #     smcorr_1a_sink = torch.mean(smcorr_1a_sink, dim=0)
            
            ## f1 reconstructed by fa
            f1_via_fa_t = torch.matmul(smcorr_1a, fa.t())
            corr_1a2 = torch.matmul(f1_via_fa_t, f2)/self.temperature
            smcorr_1a2 = F.softmax(corr_1a2, dim=1)

            if self.recursive:
                # print(f"step {step}")
                if step == 0:
                    with torch.no_grad():
                        # smcorr_1a_sink, _ = sinkFunc.gumbel_sinkhorn(
                        #     corr_1a, temp=self.sink_tau[0], n_iters=self.sink_iters[0]); del corr_1a
                        # if smcorr_1a_sink.shape[0]==1:
                        #     smcorr_1a_sink = smcorr_1a_sink.squeeze(0) 
                        # else:
                        #     smcorr_1a_sink = torch.mean(smcorr_1a_sink, dim=0)
                        smcorr_1a2_sink, _ = sinkFunc.gumbel_sinkhorn(
                            corr_1a2, temp=self.sink_tau[1], n_iters=self.sink_iters[1]); del corr_1a2
                        if smcorr_1a2_sink.shape[0]==1:
                            smcorr_1a2_sink = smcorr_1a2_sink.squeeze(0) 
                        else:
                            smcorr_1a2_sink = torch.mean(smcorr_1a2_sink, dim=0)

                    # self.sink_1a_matrices.append(smcorr_1a_sink)
                    self.sink_1a2_matrices.append(smcorr_1a2_sink)

                    diff = X1[b, :, None, :] - X1[b, None, :, :]
                    diff = (diff * diff).sum(2).sqrt()
                    diff = diff.pow(0.5) # make distance more sharper
                    self.diff.append(diff)
                else:
                    smcorr_1a_sink = self.sink_1a_matrices[b]
                    # smcorr_1a2_sink = self.sink_1a2_matrices[b]
                    diff = self.diff[b]
            else:           
                with torch.no_grad():
                # sink_approximated_assignment_matrix: in this step, we need to lower the effect of the sinkhorn
                    # smcorr_1a_sink, _ = sinkFunc.gumbel_sinkhorn(
                    #     corr_1a, temp=self.sink_tau[0], n_iters=self.sink_iters[0]); del corr_1a
                    # if smcorr_1a_sink.shape[0]==1:
                    #     smcorr_1a_sink = smcorr_1a_sink.squeeze(0) 
                    # else:
                    #     smcorr_1a_sink = torch.mean(smcorr_1a_sink, dim=0)
                    smcorr_1a2_sink, _ = sinkFunc.gumbel_sinkhorn(
                        corr_1a2, temp=self.sink_tau[1], n_iters=self.sink_iters[1]); del corr_1a2
                    if smcorr_1a2_sink.shape[0]==1:
                        smcorr_1a2_sink = smcorr_1a2_sink.squeeze(0) 
                    else:
                        smcorr_1a2_sink = torch.mean(smcorr_1a2_sink, dim=0)
                    diff = X1[b, :, None, :] - X1[b, None, :, :]
                    diff = (diff * diff).sum(2).sqrt()
                    diff = diff.pow(0.5) # make distance more sharper

            ## additional loss
            # kl_div_score = LMC.Local_Match_Consensus_loss(X1[b], X1[(b+1)%B], smcorr_1a)
            kl_div_score = self.qap_loss(X1[b], X1[(b+1)%B], smcorr_1a)
            kl_loss += 5.0*kl_div_score
            # print(f"QAP loss: {kl_div_score}")
            
            # dve loss smcorr_1a2
            L = diff * smcorr_1a2
            a = L.sum()/N
            weight=torch.log(1.0/a.detach())
            
            # constraint to permutation
            Lc_b = F.l1_loss(smcorr_1a2_sink, smcorr_1a2, reduction='sum')/N; del smcorr_1a2
            Lc += 3.0*Lc_b
            # Lc_b_1a = F.l1_loss(smcorr_1a_sink, smcorr_1a, reduction='sum')/N; del smcorr_1a
            # Lc += 3.0*Lc_b_1a

            ## w/o dve loss
            corr_12 = torch.matmul(f1.t(), f2)
            smcorr_12 = F.softmax(corr_12/self.temperature, dim=1); del corr_12
            L12 = diff * smcorr_12
            L += 1.0*L12

            ## final loss
            loss += (L.sum()/N)
            print(f"Loss: {L.sum()/N}, Loss 12: {L12.sum()/N}, smcorr1a2 to perm: {Lc_b}")

            ## record & check
            with torch.no_grad():
                ## f1 f1 correlation
                _, d = feature_reconstruction(f1, f1, self.temperature)
                # print(f"recon: {d}")
                diff_via_recon += d

                # ## f1 fa correlation
                corr2 = torch.matmul(f1_via_fa_t, f2)/self.temperature
                smcorr2 = F.softmax(corr2, dim=1); del corr2
                max_idx = torch.argmax(smcorr2, dim=1)
                count = 0.0
                tol_count = 0.0
                for i, max_id in enumerate(max_idx):
                    if diff[max_id, i] < 0.27: tol_count+=1
                    if max_id == i: count += 1
                correct_match += count
                del diff

                # print("correct match {}".format(count))
                # print("correct match with tol {}".format(tol_count))
        
        print("--------LOSS with DVE: {}--------".format(loss/B))
        total_loss = (loss+self.lambda_lc*Lc+self.lambda_qap*kl_loss)
        print("epoch", epoch)
        # print("loss ", loss/B)
        # print("Lc ", Lc/B)
        # print("kl_loss ", kl_loss/B)
        # print("total_loss ", total_loss/B)
        # print(f"concentrated_entropy {concentrated_entropy}")
        # print(f"divergent_entropy {divergent_entropy}")
        # print(f"sink_concentrated_entropy {sink_concentrated_entropy}")
        # print(f"sink_divergent_entropy {sink_divergent_entropy}")
        return total_loss/B, loss/B, Lc/B, kl_loss/B, correct_match/B, diff_via_recon/B
    '''
    
# def dense_correlation_loss_combined_sink_lmc(feats, meta, epoch,
#     pow=0.5, fold_corr=False, normalize_vectors=True, 
#     temperature=1.0, fname_vis=None, train=False,
#     sink_tau=[0.7,0.7], sink_iters=[20,20]):

#     print("--------dense_correlation_loss_combined_sink--------")

#     # feats = feats[0]
#     device = feats.device
    
#     X1 = meta['pc1'].to(device)
#     X2 = meta['pc2'].to(device)

#     feats1 = feats[0::2]
#     feats2 = feats[1::2] # deformed

#     B, N, C = feats1.shape
    
#     #

#     # print(temperature)
#     # input()
#     loss = 0.
#     correct_match = 0.
#     diff_via_recon = 0.
#     f_recon = 0.0
#     Lc = 0.0
#     kl_loss = 0.0
#     lambda_Lc = 0.01
#     lambda_kl = 0.01
#     for b in range(B):
#         # C X N
#         f1 = feats1[b].permute(1,0)  # source to B, C, N
#         f2 = feats2[b].permute(1,0)  # target
#         fa = feats1[(b+1)%B].permute(1,0) # auxiliary

#         if normalize_vectors:
#             f1 = F.normalize(f1, p=2, dim=0) * 20
#             f2 = F.normalize(f2, p=2, dim=0) * 20
#             fa = F.normalize(fa, p=2, dim=0) * 20
        
#         ## rotational equivariance constraint
#         print(f1.shape)
#         f_diff = torch.sum((f1-f2)*(f1-f2), dim=0)
#         f_recon += torch.mean(f_diff)
#         print("rot equivariance constraint f_diff: ", f_diff)
        
#         ## f1 && fa correlation
#         corr_1a = torch.matmul(f1.t(), fa) # f1 in 3xN, f2 in 3xM, corr in NxM 
#         # corr_1a = torch.matmul(f1.t(), fa) / temperature # f1 in 3xN, f2 in 3xM, corr in NxM 
#         """        
#         # this part cannot be used, because this initialization may have the sink output stuck
#         # smcorr_1a = F.softmax(corr_1a, dim=1); del corr_1a
#         """
#         # sink_approximated_assignment_matrix: in this step, we need to lower the effect of the sinkhorn
#         smcorr_1a, _ = sinkFunc.gumbel_sinkhorn(corr_1a, temp=sink_tau[0], n_iters=sink_iters[0])
#         if smcorr_1a.shape[0]==1:
#             smcorr_1a = smcorr_1a.squeeze(0) 
#         else:
#             smcorr_1a = torch.mean(smcorr_1a, dim=0)
#         print("smcorr_1a.shape: ", smcorr_1a.shape)
#         print("smcorr_1a.sum(0): ", smcorr_1a.sum(0))
#         print("smcorr_1a.sum(1): ", smcorr_1a.sum(1))

#         ## f1 reconstructed by fa
#         f1_via_fa_t = torch.matmul(smcorr_1a, fa.t())
#         corr_1a2 = torch.matmul(f1_via_fa_t, f2)
#         # sink_approximated_assignment_matrix: in this step, we need to lower the effect of the sinkhorn
#         with torch.no_grad():
#             smcorr_1a2_sink, _ = sinkFunc.gumbel_sinkhorn(corr_1a2, temp=sink_tau[1], n_iters=sink_iters[1])
#             if smcorr_1a2_sink.shape[0]==1:
#                 smcorr_1a2_sink = smcorr_1a2_sink.squeeze(0) 
#             else:
#                 smcorr_1a2_sink = torch.mean(smcorr_1a2_sink, dim=0)
#             print("smcorr_1a2_sink.shape: ", smcorr_1a2_sink.shape)
#             print("smcorr_1a2_sink.sum(0): ", smcorr_1a2_sink.sum(0))
#             print("smcorr_1a2_sink.sum(1): ", smcorr_1a2_sink.sum(1))
        
#             diff = X1[b, :, None, :] - X1[b, None, :, :]
#             diff = (diff * diff).sum(2).sqrt()
#             diff = diff.pow(0.5) # make distance more sharper

#         ## additional loss
#         # kl_div_score = LMC.Local_Match_Consensus_loss(X1[b], X1[(b+1)%B], smcorr_1a)
#         kl_div_score = LMC.Local_Match_Consensus_loss_qap(X1[b], X1[(b+1)%B], smcorr_1a); del smcorr_1a
#         kl_loss += kl_div_score
#         print(f"QAP loss: {kl_div_score}")
        
#         # smcorr_1a2
#         smcorr_1a2 = F.softmax(corr_1a2/temperature, dim=1); del corr_1a2
#         L = diff * smcorr_1a2
#         a = L.sum()/N
#         weight=torch.log(1.0/a.detach())

#         # constraint to permutation
#         Lc_b = F.l1_loss(smcorr_1a2_sink, smcorr_1a2, reduction='sum')/N
#         Lc += weight*Lc_b

#         ## w/o dve loss
#         corr_12 = torch.matmul(f1.t(), f2)
#         smcorr_12 = F.softmax(corr_12/temperature, dim=1); del corr_12
#         L12 = diff * smcorr_12
#         L += L12

#         ## final loss
#         loss += (L.sum()/N)
#         print("Loss 1a2: {0:4f}, Loss 12: {1:4f}, cst. to BP: {2:4f}, re-weight: {3:4f}".format(
#             L.sum()/N, L12.sum()/N, Lc_b, weight))

#         ## record & check
#         with torch.no_grad():
#             ## f1 f1 correlation
#             _, d = feature_reconstruction(f1, f1, temperature)
#             diff_via_recon += d

#             # ## f1 fa correlation
#             corr2 = torch.matmul(f1_via_fa_t, f2) / temperature
#             smcorr2 = F.softmax(corr2, dim=1); del corr2
#             max_idx = torch.argmax(smcorr2, dim=1)
#             count = 0.0
#             tol_count = 0.0
#             for i, max_id in enumerate(max_idx):
#                 if diff[max_id, i] < 0.3: tol_count+=1
#                 if max_id == i: count += 1
#             correct_match += count
#             del diff

#             print("correct match {}".format(count))
#             print("correct match with tol {}".format(tol_count))
    
#     print("--------LOSS with DVE: {}--------".format(loss/B))
#     print("are we here?")
#     total_loss = (loss+0.01*Lc+0.01*kl_loss)
#     # if (epoch % 1) == 0 and lambda_kl < 0.01:
#     #     lambda_kl *= 5
#     # total_loss = (loss+lambda_Lc*Lc+lambda_kl*kl_loss)

#     print("epoch", epoch)
#     print("loss ", loss/B)
#     print("Lc ", Lc/B)
#     print("kl_loss ", kl_loss/B)
#     print("total_loss ", total_loss/B)


#     return total_loss/B, loss/B, Lc/B, kl_loss/B, correct_match/B, diff_via_recon/B