import numpy as np 
import pointcloud_utils.farthest_point_sampling as fps
import torch
import torch.nn.functional as F
# from pointnet3.loss.chamfer import *
from scipy.spatial.transform import Rotation as RotTool
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
from sklearn import manifold
import pointnet3.sinkhorn_approximate as sinkFunc

def write_paired_pointclouds_obj(filename, xyz1, pair, xyz2, color1=None, color2=None, offset=None):
    # write xyz1
    f= open(filename,"w+")
    msg_vert = "v {} {} {} {} {} {}\n"

    f.write("# first point cloud\n")
    num_pts_1 = xyz1.shape[0]
    for i in range(num_pts_1):
        if color1 is None:
            f.write(msg_vert .format(xyz1[i,0], xyz1[i,1], xyz1[i,2], 1, 0, 0))
        else:
            f.write(msg_vert .format(xyz1[i,0], xyz1[i,1], xyz1[i,2], color1[i,0], color1[i,1], color1[i,2]))


    # init offset
    if offset is None:
        offset = [0.0,0.0,0.0]
    else:
        assert 3==len(offset)
    
    f.write("# second point cloud\n")
    num_pts_2 = xyz2.shape[0]
    for i in range(num_pts_2):
        if color2 is None:
            f.write(msg_vert .format(xyz2[i,0]+offset[0], xyz2[i,1]+offset[1], xyz2[i,2]+offset[2], 0, 1, 0))
        else:
            f.write(msg_vert .format(xyz2[i,0]+offset[0], xyz2[i,1]+offset[1], xyz2[i,2]+offset[2], color2[i,0], color2[i,1], color2[i,2]))

    
    # write links
    f.write("# correspondence pair\n")
    for i in range(pair.shape[0]):
        f.write("l {} {}\n".format(pair[i, 0]+1, pair[i, 1]+1+num_pts_1))
    #
    f.close()


def save_data_as_image(filename, data):
    min_val = np.amin(data)
    max_val = np.amax(data)
    # RESCALING THE DATA to 0 255
    img = (data - min_val) / (max_val-min_val) * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img, 'P')
    img.save(filename)

## from PRNet
def farthest_subsample_points_no_batch(data, viewpoint, num_subsampled_points=768):
    np_data = np.asarray(data).copy()
    # assert np_data.shape[0] > num_subsampled_points
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(np_data)
    return nbrs1.kneighbors(viewpoint, return_distance=False).reshape((num_subsampled_points,))


def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle
    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about
    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()

def random_rotation(axis, angle_range):
    """
    axis=np.array([0.0, 1.0, 0.0])
    angle_range=15.0
    """
    rotation_angle = np.random.uniform(low=-1.0, high=1.0) * angle_range/180.0 *np.pi # [-15, 15]
    rotation_matrix = angle_axis(rotation_angle, axis)
    rotation_angle_degree = rotation_angle/np.pi*180
    return  rotation_matrix, rotation_angle_degree ## torch

def random_translation(translate_range, dim=3):
    translation = np.random.uniform(-translate_range, translate_range)
    translation = [np.random.uniform(-translate_range, translate_range) for i in range(dim)]
    return torch.tensor(translation)

def align(A, B, weights=None):
    """
    A, B are two D-dim point lists, where each entry of one corresponds to that of the other.
    weights may be the likelyhood for each pair of correspondence in A and B (at the same entry)
    The function uses svd to compute a rigid transformation (R, t) that aligns the two point lists
    
    specifically, RA + t = B 
    as we assume A to be N-by-3, we return the transpose of R for convenience 
    
    more info can be found in this note: https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    """
    assert len(A.shape) == 2
    assert B.shape == A.shape

    A = np.array(A)
    B = np.array(B)

    N, D = A.shape

    if weights is not None:
        assert len(weights.shape) == 2
        assert weights.shape[1] == 1
        assert weights.shape[0] == A.shape[0]
        weights = np.asarray(weights.cpu())
    else:
        weights = np.ones((N, 1))

    _debug = False

    cent_A = np.sum(weights*A, axis=0) / sum(weights)
    cent_B = np.sum(weights*B, axis=0) / sum(weights)
    # print("cent A: ", cent_A)
    # print("cent B: ", cent_B)
    A = A - cent_A
    B = B - cent_B
    W = weights*np.eye(N) ## make a diag matrix
    S = np.matmul(W, B)
    S = np.matmul(A.transpose(1,0), S)
    
    U, s, Vh = np.linalg.svd(S)
    R_ab_t = np.matmul(U, Vh) ## transpose of R_ab
    t_ab = cent_B - np.matmul(cent_A,R_ab_t)

    ## check
    if _debug:
        print("R_ab_t: ", R_ab_t)
        print("t_ab: ", t_ab)
        diff = np.mean(B - np.matmul(A,R_ab_t) + t_ab)
        print("diff: ", diff)
    return R_ab_t.transpose(1,0), t_ab

def find_nearest_neighbor(query, reference, temperature):
    """
    # Input #
    feat_query: per-point features of the query point set
    feat_ref: per-point features of the collection of the point sets with known labels
    temperature: a hyperparameter for softmax normalization; 
                this is required to be the same as set for training
    
    assume the input has a shape of N x D, where N is the number of queries, and D is the dimension

    # Output #
    nn1: a 2-d tensor that first column is indices of query and the second is indices of reference
    """
    N, C = query.shape
    src_idx = torch.tensor(range(0, query.shape[0]))
    # print(reference.shape)

    # swap the dims as we assume inputs are N x D
    query = query.permute(1,0)
    reference = reference.permute(1,0)

    # correlation
    corr_1a = torch.matmul(query.t(), reference)/temperature
    smcorr_1a = F.softmax(corr_1a, dim=1)
    # smcorr_1a_sink, _ = sinkFunc.gumbel_sinkhorn(corr_1a, temp=0.3, n_iters=30)
    # smcorr_1a = smcorr_1a_sink.squeeze(0)
    # save_data_as_image('smcorr_1a.png', np.asarray(smcorr_1a.cpu()))
    # input()
    # print("smcorr_1a.shape: ", smcorr_1a.shape)
    # sum_smcorr_1a = torch.sum(smcorr_1a**2)
    # print("smcorr_1a sq sum", sum_smcorr_1a)
    # for each query, its correspondence in the reference
    score, _ = torch.max(smcorr_1a, dim=1)
    nn1 = torch.argmax(smcorr_1a, dim=1).view(-1).type(torch.LongTensor)
    # nn1 =torch.argmax(smcorr_1a, dim=0).type(torch.LongTensor)
    # make 2d tensor
    # print("1-NN: ", nn1)
    # input()
    nn1 = torch.stack((src_idx, nn1.view(-1)), dim=1)
    # print(f"nn1 {nn1.shape}")
    return nn1, score

class AlignmentCheck(object):
    def __init__(self, temperature, translation_range=None, rotation_angle_range=None, registration=False, _debug=False):
        self.temperature = temperature
        self.translation_range = translation_range
        self.rotation_angle_range= rotation_angle_range
        if translation_range is None:
            print("translation_range is None")
            print("rotation_angle_range range is None")

        self.rotation_list =[]
        self.translation_list=[]
        self.angle_list=[]

        self.axis_1 = np.array([1.0, 0.0, 0.0])
        self.axis_2 = np.array([0.0, 1.0, 0.0])
        self.axis_3 = np.array([0.0, 0.0, 1.0])

        self._debug = _debug
        self.registration = registration

    def _sample_shape_fps(self, xyz, num_sample):
        assert len(xyz.shape) == 2
        pt_idxs = fps.farthest_point_sample(xyz.unsqueeze(0), num_sample)
        return pt_idxs.squeeze(0).type(torch.LongTensor)

    def _align(self, A, B, weights=None):
        ## R*A + t = B
        device = A.device
        R, t = align(A.cpu(), B.cpu(), weights)
        R  = torch.from_numpy(R).type(torch.FloatTensor)
        t = torch.from_numpy(t).type(torch.FloatTensor)
        return R.to(device), t.to(device) 

    def _apply_transform(self, X, R, t):
        X_transform = torch.matmul(X, R.t()) + t
        return X_transform

    def transform_data(self, data, has_normal=False, random_perm=True):
        B, N, C = data.shape
        device = data.device
        data_transform = []

        if has_normal:
            normals=data[...,3:]
            data=data[...,:3]
        else:
            data=data[...,:3]

        ## reset
        self.rotation_list.clear()
        self.translation_list.clear()
        self.angle_list.clear()
        for i in range(B):
            R_gt_1, r_angle_1 = random_rotation(axis=self.axis_1, angle_range=self.rotation_angle_range)
            R_gt_2, r_angle_2 = random_rotation(axis=self.axis_2, angle_range=self.rotation_angle_range)
            R_gt_3, r_angle_3 = random_rotation(axis=self.axis_3, angle_range=self.rotation_angle_range)
            R_gt = torch.matmul(torch.matmul(R_gt_1, R_gt_2), R_gt_3)
            angle_gt = RotTool.from_matrix(R_gt).as_euler('zyx', degrees=True)
            t_gt = random_translation(self.translation_range)
            self.rotation_list.append(R_gt)
            self.translation_list.append(t_gt)
            self.angle_list.append(angle_gt)
            data_transform.append(self._apply_transform(data[i], R_gt.to(device), t_gt.to(device)))

        assert len(self.rotation_list) == B
        assert len(self.translation_list) == B
        data_transform = torch.stack(data_transform, dim=0)
        if has_normal:
            data_transform = torch.cat([data_transform, normals], dim=-1)

        # ## random permutation of the set        
        if random_perm:
            print("do permutation")
            idx = torch.randperm(1024)
            data_transform = data_transform[:, idx, :]        
        return data_transform

    def transform_data_as_PRNet(
        self, data, rotations, translations, eulers=None, 
        has_normal=False, partial_vp=None, random_perm=True):
        device = data.device
        B, N, C = data.shape
        assert rotations.shape[0] == B and translations.shape[0] == rotations.shape[0]
        # print(rotations.shape)
        # if eulers is not None:
        #     print(rotations.shape)

        if not has_normal:
            assert C==3
        else:
            normals=data[...,3:]
            data=data[...,:3]

        data_transform = []
        normals_transform =[]
        # print(eulers)
        for i in range(B):
            R_gt, t_gt = rotations[i], translations[i]
            data_transform.append(self._apply_transform(data[i], R_gt, t_gt))
            if has_normal:
                normals_transform.append(torch.matmul(normals[i,...], R_gt.t()))

        data_transform = torch.stack(data_transform, dim=0)

        if has_normal:
            normals_transform = torch.stack(normals_transform, dim=0)
            data_transform = torch.cat([data_transform, normals_transform], dim=-1)

        # ## random permutation of the set        
        if random_perm:
            print("do permutation")
            idx = torch.randperm(1024)
            data_transform = data_transform[:, idx, :]
        
        return data_transform

    def subsample(self, data, partial_vp, num_samples=768, has_normal=False):
        B, N, C = data.shape
        if not has_normal:
            assert C==3
        else:
            normals=data[...,3:]
            data=data[...,:3]
        
        assert B==1
        data=data.squeeze(0)
        idx = farthest_subsample_points_no_batch(data.cpu(), partial_vp, num_samples)

        if has_normal:
            normals=normals.squeeze(0)
            data = torch.cat([data, normals],dim=-1)
        
        data_fps = data[idx, :]
        print("Subsampleing shape:", data_fps.shape)
        return data_fps.unsqueeze(0)

    def __call__(self, data, feat, data_transform, feat_transform, has_normal=False):
        assert self.registration
        B, N, C = data.shape
        assert B == 1

        if has_normal:
            normals = data[...,3:]
            data=data[...,:3] # exclude normals
            normals_transform = data_transform[...,3:]
            data_transform = data_transform[...,:3] # exclude normals

        data_i = data[0]
        feat_i = feat[0]
        data_transform_i = data_transform[0]
        feat_transform_i = feat_transform[0]
            
        sort_tgt_idx, score = find_nearest_neighbor(feat_i, feat_transform_i, self.temperature)
        sort_tgt_idx = sort_tgt_idx[score>0.9,:]
        sort_data_transform_i = data_transform_i[sort_tgt_idx[:,1],:]
        sort_data_i = data_i[sort_tgt_idx[:,0], :]

        R, t = self._align(sort_data_i, sort_data_transform_i)
        eulers = RotTool.from_matrix(R.cpu()).as_euler('zyx', degrees=True)
        est_data = self._apply_transform(data_i, R, t)

        if has_normal:
            est_normals = torch.matmul(normals[0], R.t())
            est_shape = torch.cat([est_data, est_normals], dim=-1)
        else:
            est_shape = est_data
        
        if False:
            tsne_method = manifold.TSNE(n_components=3, init='pca', random_state=0)
            F_i = np.array(feat_i.cpu().reshape((B*N,-1)))
            Ft_i = np.array(feat_transform_i.cpu().reshape((B*N,-1)))
            F_cat = np.concatenate([F_i, Ft_i], axis=0)
            print(f"F_cat.shape {F_cat.shape}")
            Y = tsne_method.fit_transform(F_cat)
            print("t-sne done")
            ## normalize to color space
            Ymax = np.zeros((1, 3), dtype=float)+40
            Ymin = np.zeros((1, 3), dtype=float)-40
            Ymax = np.fmin(Ymax, np.max(Y, axis=0))
            Ymin = np.fmax(Ymin, np.min(Y, axis=0))
            print(f"Ymax {Ymax}, Ymin {Ymin}")
            Ys = (Y - Ymin) / (Ymax-Ymin)
            color = Ys[:N, :]
            color_transform = Ys[N:, :]
            write_paired_pointclouds_obj(
                "test.obj", data_i.cpu(), sort_tgt_idx, data_transform_i.cpu(), 
                color1=color, color2=color_transform)
        # else:
        #     write_paired_pointclouds_obj(
        #         "test.obj", data_i.cpu(), sort_tgt_idx, data_transform_i.cpu())

        return eulers, R, t, est_shape