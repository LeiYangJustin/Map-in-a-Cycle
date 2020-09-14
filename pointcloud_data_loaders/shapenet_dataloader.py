import os
import os.path as osp
import shutil
import json
import torch
import torch.utils.data as data
import numpy as np
import os
import h5py
import subprocess
import shlex
import random
import time

from torchvision import transforms
import pointcloud_data_loaders.pc_data_utils as dutils
import pointcloud_utils.farthest_point_sampling as fps

# def _download(self):
#     path = download_url(self.url, self.root)
#     extract_zip(path, self.root)
#     os.unlink(path)
#     shutil.rmtree(self.raw_dir)
#     name = self.url.split('/')[-1].split('.')[0]
#     os.rename(osp.join(self.root, name), self.raw_dir)
#     print("download")

def save_filename_list(path_to_save, fname_list, mode="w+"):
    f= open(path_to_save, mode)
    for fname in fname_list:
        f.write(fname)
    f.close()

def load_model_data(filename, sep=','):
    with open(filename) as f:
        point_list = [np.fromstring(line, dtype=float, sep=sep) for line in f]
        N = len(point_list)
        data = np.concatenate(point_list, 0).reshape(N, -1) # shape to N*(3+3)
    
    if data.shape[1] == 4:
        xyz = data[:, :3]
        labels = data[:, 3:4]
        return xyz, labels
    elif data.shape[1] == 7:
        xyz = data[:, :6]
        labels = data[:, 6:7]
        return xyz, labels
    else:
        print("data.shape[1] != 4 or 7")
        return data

def compute_laplacian_on_pointcloud_with_normals(xyz, normals, radius=0.1, nsample=32):
    squeezed = False
    if len(xyz.shape) == 2:
        xyz = xyz.unsqueeze(0)
        normals = normals.unsqueeze(0)
        squeezed = True
    B, N, C = xyz.shape
    idx = query_ball_point(radius, nsample, xyz, xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz -= xyz.view(B, N, 1, C) ## delta_ji = x_j - x_i
    laplacian_vector = grouped_xyz.permute(0,1,3,2).mean(dim=-1) ## delta_ji / deg
    # print("laplacian_vector.shape ", laplacian_vector.shape)
    # print("normals.shape ", normals.shape)
    laplacian = torch.abs((laplacian_vector*normals).sum(dim=-1))
    if squeezed:
        return laplacian.squeeze(0)
    return laplacian

class ShapeNetPartSegLoader():
    url = ('https://shapenet.cs.stanford.edu/media/'
           'shapenetcore_partanno_segmentation_benchmark_v0_normal.zip')

    category_ids = {
        'Airplane': '02691156',
        'Bag': '02773838',
        'Cap': '02954340',
        'Car': '02958343',
        'Chair': '03001627',
        'Earphone': '03261776',
        'Guitar': '03467517',
        'Knife': '03624134',
        'Lamp': '03636649',
        'Laptop': '03642806',
        'Motorbike': '03790512',
        'Mug': '03797390',
        'Pistol': '03948459',
        'Rocket': '04099429',
        'Skateboard': '04225987',
        'Table': '04379243',
    }

    seg_classes = {
        'Airplane': [0, 1, 2, 3],
        'Bag': [4, 5],
        'Cap': [6, 7],
        'Car': [8, 9, 10, 11],
        'Chair': [12, 13, 14, 15],
        'Earphone': [16, 17, 18],
        'Guitar': [19, 20, 21],
        'Knife': [22, 23],
        'Lamp': [24, 25, 26, 27],
        'Laptop': [28, 29],
        'Motorbike': [30, 31, 32, 33, 34, 35],
        'Mug': [36, 37],
        'Pistol': [38, 39, 40],
        'Rocket': [41, 42, 43],
        'Skateboard': [44, 45, 46],
        'Table': [47, 48, 49],
    }

    split_types = ['train', 'val', 'test']

    def __init__(
        self, num_points, root, category, 
        split, has_warper=False, augmentation=False, non_uniform=False,
        rotsd=15.0, transsd=0.2, scale_lo=0.8, scale_hi=1.25, truncated_dataset=None
        ):
        super().__init__()
        ## if there is no data on the disk, download it
        if not osp.exists(root):
            raise "No specified path"

        self.category_list = []
        assert category in self.category_ids
        self.category_list.append(self.category_ids[category])
        print(f"category: {category}-{self.category_list}")

        ##
        self.root = root
        self.num_points = num_points
        self.has_warper = has_warper
        self.split = split
        self.augmentation = augmentation
        self.non_uniform = non_uniform

        print(
            f"root {self.root}, split {self.split}", 
            f"has_warper {self.has_warper}, do augmentation {self.augmentation}",
            f"non_uniform sampling {self.non_uniform}" )

        print(
            f"rotation: {rotsd}deg, translation {transsd}", 
            f"scale: {scale_lo} {scale_hi}" )

        ## load data
        self.fname_list = self._make_datalist(self.split)
        point_list, label_list = self._load(self.fname_list)
        
        if truncated_dataset is not None:
            assert isinstance(truncated_dataset, float)
            # assert truncated_dataset < 0.5
            K = int(truncated_dataset*len(point_list))
            rand_indices=torch.randperm(len(point_list))[0:K]
            print(f"training sample ids: {rand_indices}")
            self.point_list = [point_list[idx] for idx in rand_indices]
            self.label_list = [label_list[idx] for idx in rand_indices]
        else:
            self.point_list = point_list
            self.label_list = label_list

        print("# shapes loaded: ", len(point_list))

        ## process the label
        self.min_label = np.concatenate(self.label_list).min()
        print("min label", self.min_label)

        # to tensor
        self.toTensor = dutils.PointcloudToTensor()

        ## data warper
        self.warper_da = transforms.Compose(
            [
                dutils.PointcloudRotate(angle=rotsd,axis=np.array([1, 0, 0])),
                dutils.PointcloudRotate(angle=rotsd,axis=np.array([0, 1, 0])),
                dutils.PointcloudRotate(angle=rotsd,axis=np.array([0, 0, 1])),
                dutils.PointcloudScale(hi=scale_hi, lo=scale_lo),
                dutils.PointcloudTranslate(translate_range=transsd),
                # dutils.PointcloudJitter()
            ]
        )

        # data augmentation
        self.transforms = transforms.Compose(
            [
                dutils.PointcloudRotate(angle=rotsd,axis=np.array([1, 0, 0])),
                dutils.PointcloudRotate(angle=rotsd,axis=np.array([0, 1, 0])),
                dutils.PointcloudRotate(angle=rotsd,axis=np.array([0, 0, 1])),
                dutils.PointcloudScale(hi=scale_hi, lo=scale_lo),
                dutils.PointcloudTranslate(translate_range=transsd),
                # dutils.PointcloudJitter()
            ]
        )
        
    def _make_datalist(self, split):
        path = osp.join(self.root, 'train_test_split',
                        f'shuffled_{split}_file_list.json')
        filenames = []
        with open(path, 'r') as f:
            for name in json.load(f):
                category, fname = name.split(osp.sep)[1:] # Removing first directory.
                category_fname = osp.join(category, fname)
                if category in self.category_list:
                    filenames.append(category_fname)
        return filenames

    def _load(self, fname_list):
        xyz_list, normals_list, labels_list = [], [], []
        for fname in fname_list:
            fname = osp.join(self.root, fname+'.txt')
            xyz, labels = load_model_data(fname, sep=' ')
            xyz_list.append(xyz)
            labels_list.append(labels)
        return xyz_list, labels_list

    def __getitem__(self, index):
        current_points = self.point_list[index].copy()
        current_points = self.toTensor(current_points)
        current_labels = self.label_list[index].copy()
        current_labels = self.toTensor(current_labels).type(torch.LongTensor) - self.min_label # start from 1
        
        if self.num_points is not None:
            if not self.non_uniform:
                pt_idxs = fps.farthest_point_sample(current_points, self.num_points)
                pt_idxs = pt_idxs[torch.randperm(pt_idxs.shape[0])]
            else:
                print(current_points.shape[0])
                pt_idxs = torch.randperm(current_points.shape[0])[:1024]
                print(pt_idxs.shape[0])

            X = current_points[pt_idxs,:]
            labels = current_labels[pt_idxs]
        else:
            X = current_points
            labels=current_labels
            return X
        
        data, meta = self.prepare_wrap_data2(X)
        meta = {**meta, 'pc0': X, 'index': index, 'label': labels}
        
        return {"data": data, "meta": meta}

    def __len__(self):
        return len(self.point_list)

    def size(self):
        return len(self.point_list)

    def prepare_wrap_data2(self, X):
        if self.has_warper: ## training
            TX1 = self.warper_da(X)
            TX2 = self.warper_da(X)
            data = torch.stack((TX1, TX2), 0)
            meta = {'pc1': TX1, 'pc2': TX2}
        else: ## testing
            if self.transforms is not None and self.augmentation:
                # print("do data augmentation")
                X = self.transforms(X)
            data = X[None, ...] # to keep the dim
            meta = {'pc1': X}
        return data, meta

    def get_data_with_indices(self, indices):
        output_list=[]
        for idx in indices:
            output_list.append(self.__getitem__(idx))

        # make a batch
        data_batch = [x["data"] for x in output_list]
        data_batch = torch.stack(data_batch, dim=0)

        def merge_dict(ori, add):
            if ori is None:
                ori = {}
                for key, val in add.items():
                    ori[key] = [val] # list
            else:
                for key, val in add.items():
                    ori[key].append(val) # list
            return ori

        meta_batch = None
        for i, x in enumerate(output_list):
            meta_batch = merge_dict(meta_batch, x["meta"])
        # make list to torch.tensor
        for key, val in meta_batch.items():
            if key == 'index':
                meta_batch[key] = torch.tensor(val)
            else:
                meta_batch[key] = torch.stack(val, dim=0)

        ##
        return {"data": data_batch, "meta": meta_batch}

    def get_data_with_index(self, index):
        assert isinstance(index, int)
        return self.__getitem__(index)

###
if __name__ == "__main__":
    root = "../DataModelNet/PyGeoData/Data"
    make_shapenet = ShapeNetMaker(root, category_list=["Airplane"], split='test')
    data = make_shapenet.get_data()
    
    xyz = data['xyz'][0]
    print(xyz)
    gauss_curvature = data['gauss_curvature'][0]
    
    if True:
        # gauss_curvature as color
        color = (gauss_curvature-gauss_curvature.min())/(gauss_curvature.max()-gauss_curvature.min())
        color = color.unsqueeze(1).repeat(1, 3)
    else:
        # distance as color
        color = dist[:, 364]
        color = color.unsqueeze(1).repeat(1, 3)

    write_obj('shape_color_xyz_curvature_364.obj', xyz, color=color)
    print('done')
    
