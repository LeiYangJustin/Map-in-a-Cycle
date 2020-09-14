import torch.nn.functional as F
import torch.nn as nn
import time
import torch
from PIL import Image
import numpy as np
from pointcloud_utils import pointcloud_vis
import pointnet3.sinkhorn_approximate as sinkFunc

def save_data_as_image(filename, data):
    min_val = np.amin(data)
    max_val = np.amax(data)
    # RESCALING THE DATA to 0 255
    img = (data - min_val) / (max_val-min_val) * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img, 'P')
    img.save(filename)

class DVE_loss(nn.Module):
    loss_type = ['reconstruction', 'cycle', 'cyle_new']
    def __init__(
        self,
        pow=0.5, fold_corr=False, normalize_vectors=True, 
        temperature=1.0,
        sink_tau=[0.3,0.3], sink_iters=[30,30],
        lambda_lc=0.0, lambda_qap=0.0, lambda_ln=0.0,
        local_args=None
        ):
        super(DVE_loss, self).__init__()
        self.pow=pow
        self.fold_corr = fold_corr
        self.normalize_vectors=normalize_vectors
        self.temperature = temperature
        self.sink_tau = sink_tau
        self.sink_iters = sink_iters
        self.lambda_lc = lambda_lc
        self.lambda_qap = lambda_qap
        self.lambda_ln = lambda_ln
        
        print(f"temperature {temperature} sink_tau {sink_tau}, sink_iters {sink_iters}, pow {pow}")


    def forward(self, feats, meta, epoch, step=0, fname_vis=None):
        device = feats.device
        X1 = meta['pc0'].to(device)
        if X1.shape[2] > 3:
            N1 = X1[:,:,3:]
            X1 = X1[:,:,:3]

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
        Lc = 0.0
        perm_to_I = 0.0

        for b in range(B):
            # C X N
            f1 = feats1[b].permute(1,0)  # source to B, C, N
            f2 = feats2[b].permute(1,0)  # target
            fa = feats1[(b+1)%B].permute(1,0) # auxiliary

            if self.normalize_vectors:
                f1 = F.normalize(f1, p=2, dim=0) * 20
                f2 = F.normalize(f2, p=2, dim=0) * 20
                fa = F.normalize(fa, p=2, dim=0) * 20

            ## f1 && fa correlation
            corr_1a = torch.matmul(f1.t(), fa)/self.temperature ## [C, M]T X [C, N] = [M, N]
            smcorr_1a = F.softmax(corr_1a, dim=1)
            ## f1 reconstructed by fa
            f1_via_fa_t = torch.sum( smcorr_1a[:, None, :]*fa[None, :, :], dim=-1 )   ## [M, 1, N] X [1, C, N] --> [M, C, N] --> [M, C]
            corr_1a2 = torch.matmul(f1_via_fa_t, f2)/self.temperature ## [M, C] X [C, K] = [M, K]
            smcorr_1a2 = F.softmax(corr_1a2, dim=1)

            with torch.no_grad():
                smcorr_1a2_sink, _ = sinkFunc.gumbel_sinkhorn(
                    corr_1a2, temp=self.sink_tau[1], n_iters=self.sink_iters[1]); del corr_1a2
                if smcorr_1a2_sink.shape[0]==1:
                    smcorr_1a2_sink = smcorr_1a2_sink.squeeze(0) 
                else:
                    smcorr_1a2_sink = torch.mean(smcorr_1a2_sink, dim=0)

                diff = X1[b, :, None, :] - X1[b, None, :, :]
                dist = (diff * diff).sum(2).sqrt()
                dist = dist.pow(self.pow) # make distance more sharper
            
            # C1*C2
            L = dist * smcorr_1a2

            ## rotational invariance
            corr_12 = torch.matmul(f1.t(), f2)
            smcorr_12 = F.softmax(corr_12/self.temperature, dim=1); del corr_12
            L12 = dist * smcorr_12

            ## for reference
            perm_to_I += 3.0*F.l1_loss(torch.eye(N).to(device), smcorr_1a2_sink, reduction='sum')/N

            ## Sinkhorn regularization
            ## ablation
            ## 1) constraint to permutation
            constraint_to_perm = "1a2_perm"
            # constraint_to_perm = "1a_perm"
            if constraint_to_perm == "1a2_perm":
                Lc_b = F.l1_loss(smcorr_1a2_sink, smcorr_1a2, reduction='sum')/N
            ## 2) constraint to identity
            elif constraint_to_perm == "1a2_identity":
                Lc_b = F.l1_loss(torch.eye(N).to(device), smcorr_1a2, reduction='sum')/N
                print("constraint smcorr_1a2_sink to identity")
            ## 3) constraint on 1-a correspondence
            elif constraint_to_perm == "1a_perm":
                Lc_b = F.l1_loss(smcorr_1a_sink, smcorr_1a, reduction='sum')/N; del smcorr_1a
                print("constraint smcorr_1a_sink to perm")

            Lc += 3.0*Lc_b
            
            ## finall loss
            L += 1.0*L12
            loss += (L.sum()/N)
            
            print(f"Loss: {L.sum():.6f}, Loss 12: {L12.sum():.6f}, smcorr1a2 to perm: {Lc_b:.6f}")
            
            ## record & check
            with torch.no_grad():
                # ## f1 fa correlation
                max_idx = torch.argmax(smcorr_1a2, dim=1)
                count = 0.0
                for i, max_id in enumerate(max_idx):
                    if max_id == i: count += 1
                correct_match += count

                if fname_vis is not None and np.sum(vis_idx==b) == 1:                    
                    txt_fname = fname_vis+str(b) + "smcorr_1a2_sink.png"
                    npdata = smcorr_1a2_sink.cpu().detach().numpy()
                    save_data_as_image(txt_fname, npdata)

                    txt_fname = fname_vis+str(b) + "smcorr_1a2.png"
                    npdata = smcorr_1a2.cpu().detach().numpy()
                    save_data_as_image(txt_fname, npdata)
                    print("saved files")
                del diff
            
           

        print("--------LOSS with DVE: {}--------".format(loss/B))
        total_loss = loss + self.lambda_lc*Lc

        output_loss = {
            'total_loss': total_loss/B,
            'cycle_loss': loss/B, 
            'perm_loss': Lc/B, 
        }
        output_info = {
            'correct_match': correct_match/B, 
            'smcorr_to_I': perm_to_I/B,
        }
        return output_loss, output_info


   