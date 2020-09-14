import torch
import numpy as np

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud data, [B, npoint, C]
    """
    # print("xyz.shape: ", xyz.shape)
    squeezed = False
    if len(xyz.shape) == 2:
        xyz = xyz.unsqueeze(0)
        squeezed = True

    if xyz.shape[-1] != 3:
        xyz = xyz[:, :, :3]

    device = xyz.device
    B, N, C = xyz.shape
    if N < 2048: print("N < 2048")
    assert B > 0, "B <= 0"
    assert C == 3, "C != 3"
    # print("B={}, N={}, C={}".format(B, N, C))
    S = npoint
    # print("S={}".format(S))
    centroids = torch.zeros(B, S, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(S):
        centroids[:, i] = farthest
        # print("farthest: ", farthest)
        centroid = xyz[batch_indices, farthest, :]
        try:
            centroid.view(B, 1, 3)
            centroid = centroid.view(B, 1, 3)
        except RuntimeError:
            print("xyz.shape: ", xyz.shape)
            print("farthest: ", farthest)
            print("batch_indices: ", batch_indices)
            print("centroid.shape: ", centroid.shape)
            print("xyz: ", xyz)
            raise

        dist = torch.sum((xyz - centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    if squeezed is True:
        centroids = centroids.squeeze(0)

    return centroids