
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
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

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _get_data_files(path):
    with open(path) as f:
        return [line.rstrip() for line in f]

def _load_data_file(name, has_normal=False):
    print(name)
    f = h5py.File(name, 'r')
    print([key for key in f.keys()])
    
    data = f["data"][:]
    
    if has_normal:
        normals = f["normal"][:]
        data = np.concatenate([data, normals], -1)

    label = f["label"][:]
    return data, label

##
class ModelNet40Cls(data.Dataset):
    url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"   
    split_types = ['train', 'test']


    category_ids = {
        "airplane": 0,
        "bathtub": 1,
        "bed": 2,
        "bench": 3,
        "bookshelf": 4,
        "bottle": 5,
        "bowl": 6,
        "car": 7,
        "chair": 8,
        "cone": 9,
        "cup": 10,
        "curtain": 11,
        "desk": 12,
        "door": 13,
        "dresser": 14,
        "flower_pot": 15,
        "glass_box": 16,
        "guitar": 17,
        "keyboard": 18,
        "lamp": 19,
        "laptop": 20,
        "mantel": 21,
        "monitor": 22,
        "night_stand": 23,
        "person": 24,
        "piano": 25,
        "plant": 26,
        "radio": 27,
        "range_hood": 28,
        "sink": 29,
        "sofa": 39,
        "stairs": 31,
        "stool": 32,
        "table": 33,
        "tent": 34,
        "toilet": 35,
        "tv_stand": 36,
        "vase": 37,
        "wardrobe": 38,
        "xbox": 39,
    }

    def __init__(
        self, num_points, root, category, 
        split, has_warper=False, augmentation=False,
        rotsd=15.0, transsd=0.2, scale_lo=0.8, scale_hi=1.25
        ):
        super().__init__()
        ## if there is no data on the disk, download it
        if not os.path.exists(root):
            raise "No specified path"
            
        ##
        self.root = root
        self.has_warper = has_warper
        self.split = split
        self.augmentation = augmentation

        print(
            f"root {self.root}, split {self.split}", 
            f"has_warper {self.has_warper}, do augmentation {self.augmentation}" )

        print(
            f"rotation: {rotsd}deg, translation {transsd}", 
            f"scale: {scale_lo} {scale_hi}" )

        ## load data
        point_list, label_list=[], []
        self.fname_list = _get_data_files(os.path.join(self.root, f"{self.split}_files.txt"))
        for f in self.fname_list:
            points, labels = _load_data_file(os.path.join(self.root, f), has_normal=False)
            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)
        self.labels = self.labels.reshape(-1)

        if category is not None:
            category = category.lower()
            assert category in self.category_ids
            category_id = self.category_ids[category]
            self.points = self.points[self.labels==category_id]
            self.labels = self.labels[self.labels==category_id]

        self.set_num_points(num_points)

        print()
        print("# shapes loaded: ", self.points.shape)
        print("# labels loaded: ", self.labels.shape)
        print()

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
                # dutils.PointcloudJitter(),
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
                # dutils.PointcloudJitter(),
            ]
        )
    
    def set_num_points(self, pts):
        self.num_points = min(self.points.shape[1], pts)

    def __getitem__(self, index):

        current_points = self.points[index, :].copy()
        # current_points[:,:3] = dutils.pc_normalize(current_points[:,:3])
        current_points = self.toTensor(current_points)
        
        if self.num_points is not None:
            pt_idxs = fps.farthest_point_sample(current_points, self.num_points)
            pt_idxs = pt_idxs[torch.randperm(pt_idxs.shape[0])]
            X = current_points[pt_idxs,:]
        else:
            X = current_points
            return X
        
        # if self.transforms is not None:
        #     X_aug = self.transforms(X)

        data, meta = self.prepare_wrap_data2(X)
        meta = {**meta, 'pc0': X, 'index': index}
        return {"data": data, "meta": meta}

    def __len__(self):
        return self.points.shape[0]

    def size(self):
        return self.points.shape[0]

    def prepare_wrap_data2(self, X):
        if self.has_warper:
            # TX1 = self.warper_da(X)
            TX1 = X
            TX2 = self.warper_da(X)
            data = torch.stack((TX1, TX2), 0)
            meta = {'pc1': TX1, 'pc2': TX2}
        else:
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

    def get_data_with_label(self, label):
        assert isinstance(label, int)
        assert label < 40 and label >= 0
        return np.where(self.labels==label)[0].tolist()


###
if __name__ == "__main__":

    folder = "modelnet40_ply_hdf5_2048"
    data_dir = os.path.join(BASE_DIR, folder)

    dataset = ModelNet40Cls(
        num_points=1024, root=data_dir, category='Airplane', 
        split='test', has_warper=False, augmentation=False,
        rotsd=15.0, transsd=0.2, scale_lo=0.8, scale_hi=1.25
    )
    
    print("data loaded")
