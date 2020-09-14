import os
import torch
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # NOQA

sys.path.insert(0, str(Path.home() / "coding/src/zsvision/python"))
try:
    from zsvision.zs_iterm import zs_dispFig  # NOQA
except:
    print('No zs_dispFig, figures will not be displayed in iterm')


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict


def get_instance(module, name, config, *args, **kwargs):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'],
                                                 **kwargs)


def coll(batch):
    b = torch.utils.data.dataloader.default_collate(batch)
    # Flatten to be 4D
    return [
        bi.reshape((-1,) + bi.shape[-3:]) if isinstance(bi, torch.Tensor) else bi
        for bi in b
    ]


def dict_coll(batch):
    cb = torch.utils.data.dataloader.default_collate(batch)
    cb["data"] = cb["data"].reshape((-1,) + cb["data"].shape[-3:])  # Flatten to be 4D
    if False:
        from torchvision.utils import make_grid
        from utils.visualization import norm_range
        ims = norm_range(make_grid(cb["data"])).permute(1, 2, 0).cpu().numpy()
        plt.imshow(ims)
    return cb


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
