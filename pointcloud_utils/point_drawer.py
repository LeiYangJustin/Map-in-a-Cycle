import numpy as np 
import torch
import torch.nn.functional as F
import os 
import matplotlib as plt

class Drawer():
    def __init__(self, draw_format, color_pallete_name='tab10'):
        self.draw_format = draw_format
        self.colormap = plt.cm.get_cmap(color_pallete_name, 10).colors
        self.content = []
        self.content_manager = {}

    def get_content(self, content_name):
        start, length = self.content_manager[content_name]
        return self.content[start:start+length]

    def save_all_to_file(self, filename):
        print(self.content_manager)
        with open(filename, 'w+') as f:
            for line in self.content:
                ## convert line to str
                # print(line)
                f.write(line)
            print(f'save all content to {filename}')

    def save_content_to_file(self, filename, content_name):
        with open(filename, 'w+') as f:
            content = self.get_content(content_name)
            for line in content:
                ## convert line to str
                f.write(line)
            print(f'save {content_name} to {filename}')

    def clear(self):
        self.content_manager.clear()
        self.content.clear()


    def log_to_content(self, content_name, content_length):
        assert isinstance(content_length, int)
        start = len(self.content)
        self.content_manager = { **self.content_manager, content_name: (start, content_length) }

    def draw_one_shape(self, content_name, xyz, color=None):
        self.log_to_content(content_name, xyz.shape[0])

        if color is not None:
            assert color.shape[0] == 3 or color.shape[1] == 3
            has_color = True
            if color.shape[0] == xyz.shape[0]:
                per_pt_color = True
            else:
                per_pt_color = False
        else:
            has_color=False

        draw_point_format = self.draw_format['draw_point_format']
        for i, pt in enumerate(xyz):
            if not has_color:
                ## use gray color
                self.content.append( draw_point_format.format(pt[0], pt[1], pt[2], 0.5, 0.5, 0.5) )
            elif not per_pt_color:
                self.content.append( draw_point_format.format(pt[0], pt[1], pt[2], color[0], color[1], color[2]) )
            else:
                self.content.append( draw_point_format.format(pt[0], pt[1], pt[2], color[i, 0], color[i, 1], color[i, 2]) )


    def draw_a_list_of_shapes(self, content_name_base, xyz_list, color=None):
        if color is not None:
            assert color.shape[0] == 3 or color.shape[1] == 3, "support single color only for now"
        content_name_list = []
        for i, xyz in enumerate(xyz_list):
            content_name = content_name_base+f'_{i}'
            self.draw_one_shape(content_name, xyz, color)
            content_name_list.append(content_name)
        return content_name_list


    def draw_links_between_two_shapes(self, content_name, shape1_name, shape2_name, links, num_sample=None):
        self.log_to_content(content_name, links.shape[0])
        # print(links.shape)
        assert links.shape[1] == 2
        draw_link_format = self.draw_format['draw_link_format']
        print(self.content_manager)
        Nlinks=links.shape[0]
        if num_sample is not None:
            rand_indices = torch.randperm(Nlinks)[:num_sample]
            for idx in rand_indices:
                idx1 = self.content_manager[shape1_name][0] + links[idx,0] + 1 ## for obj file
                idx2 = self.content_manager[shape2_name][0] + links[idx,1] + 1 
                self.content.append( draw_link_format.format(idx1, idx2) )
        else:
            for link in links:
                idx1 = self.content_manager[shape1_name][0] + link[0] + 1 ## for obj file
                idx2 = self.content_manager[shape2_name][0] + link[1] + 1 
                self.content.append( draw_link_format.format(idx1, idx2) )


    def draw_shape_with_segmentation_labels(self, xyz, labels):
        raise NotImplementedError

    ## 
    def teaser_shape(self, filename, src_xyz, tgt_xyz, tgt_idx, src_offset=None, tgt_offset=None):
        assert len(tgt_idx.shape) == 1
        src_idx = torch.arange(tgt_idx.shape[0])
        links = torch.stack([src_idx, tgt_idx], dim=0).permute(1,0)
        #print(links.shape)

        if src_offset is None:
            self.draw_one_shape(content_name='src_shape', xyz=src_xyz, color=torch.tensor([0,0,1]))
        else:
            src_xyz = src_xyz+src_offset
            self.draw_one_shape(content_name='src_shape', xyz=src_xyz, color=torch.tensor([0,0,1]))

        if tgt_offset is None:
            self.draw_one_shape(content_name='tgt_shape', xyz=tgt_xyz, color=torch.tensor([0,1,0]))
        else:
            tgt_xyz = tgt_xyz+tgt_offset
            self.draw_one_shape(content_name='tgt_shape', xyz=tgt_xyz, color=torch.tensor([0,1,0]))

        self.draw_links_between_two_shapes(
            content_name='links', shape1_name='src_shape', shape2_name='tgt_shape', 
            links=links, num_sample=256)

        filename_all = filename + '_all.obj'
        filename_src = filename + '_src.obj'
        filename_tgt = filename + '_tgt.obj'

        self.save_all_to_file(filename_all)
        self.save_content_to_file(filename_src, content_name='src_shape')
        self.save_content_to_file(filename_tgt, content_name='tgt_shape')
