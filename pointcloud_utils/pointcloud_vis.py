# visualization of pointcloud
import torch
import numpy as np
from matplotlib import cm

def write_obj(filename, xyz, color=None, mode="w+"):
    f= open(filename, mode)
    f.write(f"#{mode}\n")
    f.write("#point cloud\n")

    print(color.shape)
    print(xyz.shape)

    if color is not None:
        assert color.shape[0] == xyz.shape[0] or color.shape[0] == 1
        if color.shape[0] == 1:
            color = color.expand(xyz.shape)
            # print(color.shape)

    for i in range(xyz.shape[0]):
        if color is not None:
            f.write("v {0:4f} {1:4f} {2:4f} {3:4f} {4:4f} {5:4f}\n".format(xyz[i,0], xyz[i,1], xyz[i,2], color[i,0], color[i,1], color[i,2]))
        else:
            f.write("v {0:4f} {1:4f} {2:4f} {3:4f} {4:4f} {5:4f}\n".format(xyz[i,0], xyz[i,1], xyz[i,2], 0,0,1))
    f.close()

def write_obj_with_labels(filename, xyz, lbls):
    # init colormap matrix
    colormap = cm.get_cmap("tab10", 10).colors
    color = colormap[lbls.type(torch.LongTensor),:]
    write_obj(filename, xyz, color)

def write_paired_pointclouds_obj(filename, data1, pair, data2=None, offset=None):
    """    
    # file contains both point clouds (xyz1 and xyz2) and 
    # the line segments linking the correspondence point pairs
    # data1 and data2 are dicts of data containing xyz (coords), gt_lbls, and pred_lbls (all are tensors)
    """

    # init colormap matrix
    colormap = cm.get_cmap("tab10", 10).colors

    xyz1 = data1["xyz"]
    gt_lbls_1 = data1.get("gt_lbls", None)
    # print("xyz1.shape", xyz1.shape)
    # print("gt_lbls_1.shape", gt_lbls_1.shape)
    # set vertex colors
    if gt_lbls_1 is not None:
        cm_lbls_1 = colormap[gt_lbls_1.type(torch.LongTensor).squeeze(1),:]

    # init offset
    if offset is None:
        offset = [0.0,0.0,0.0]
    else:
        assert 3==len(offset)
    
    # write xyz1
    f= open(filename,"w+")
    msg_vert = "v {} {} {} {} {} {}\n"

    f.write("# first point cloud\n")
    num_pts_1 = xyz1.shape[0]
    for i in range(num_pts_1):
        if gt_lbls_1 is not None:
            f.write(msg_vert .format(xyz1[i,0], xyz1[i,1], xyz1[i,2], cm_lbls_1[i,0], cm_lbls_1[i,1], cm_lbls_1[i,2]))
        else:
            f.write(msg_vert .format(xyz1[i,0], xyz1[i,1], xyz1[i,2], 1, 0, 0))

    # if write xyz2
    if data2 is not None:
        xyz2 = data2["xyz"]
        gt_lbls_2 = data2.get("gt_lbls", None)
        pred_lbls_2 = data2.get("pred_lbls", None)

        if gt_lbls_2 is not None:
            cm_lbl2_gt = colormap[gt_lbls_2.type(torch.LongTensor).squeeze(1),:]
        if pred_lbls_2 is not None:
            cm_lbl2_pred = colormap[pred_lbls_2.type(torch.LongTensor).squeeze(1),:]

        f.write("# second point cloud\n")
        num_pts_2 = xyz2.shape[0]
        for i in range(num_pts_2):
            if pred_lbls_2 is not None:
                f.write(msg_vert
                    .format(xyz2[i,0]+offset[0], xyz2[i,1]+offset[1], xyz2[i,2]+offset[2], 
                    cm_lbl2_pred[i,0], cm_lbl2_pred[i,1], cm_lbl2_pred[i,2]))
            elif gt_lbls_2 is not None:
                f.write(msg_vert
                    .format(xyz2[i,0]+offset[0], xyz2[i,1]+offset[1], xyz2[i,2]+offset[2], 
                    cm_lbl2_gt[i,0], cm_lbl2_gt[i,1], cm_lbl2_gt[i,2]))
            else:
                f.write(msg_vert .format(xyz2[i,0]+offset[0], xyz2[i,1]+offset[1], xyz2[i,2]+offset[2], 0, 1, 0))
    
    # write links
    f.write("# correspondence pair\n")

    #
    DRAW_INCORRECT_PREDICTS_ONLY = True

    if data2 is not None and gt_lbls_2 is not None and pred_lbls_2 is not None and DRAW_INCORRECT_PREDICTS_ONLY:
            incorrect_match_indices = np.where(gt_lbls_2!=pred_lbls_2)[0]
            for i in incorrect_match_indices:
                f.write("l {} {}\n".format(pair[i, 0]+1, pair[i, 1]+1+num_pts_1))
    else:
        for i in range(pair.shape[0]):
            f.write("l {} {}\n".format(pair[i, 0]+1, pair[i, 1]+1+num_pts_1))
    
    #
    f.close()





