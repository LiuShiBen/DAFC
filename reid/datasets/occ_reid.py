from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp
import glob
from .bases import BaseImageDataset

class Occluded_REID(BaseImageDataset):
    def __init__(self, root='', **kwargs):
        super(Occluded_REID, self).__init__()
        self.data_dir = root
        #print('self.data_dir',self.data_dir)
        self.query_dir=osp.join(self.data_dir, 'occluded_body_images')
        self.gallery_dir=osp.join(self.data_dir, 'whole_body_images')

        train = []
        query = self.process_dir(self.query_dir, relabel=False)    #occluded_body_images
        gallery = self.process_dir(self.gallery_dir, relabel=False, is_query=False)  #whole_body_images
        
        print("=> Occluded_REID loaded")
        self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    @property
    def images_dir(self):
        return self.data_dir
    def process_dir(self, dir_path, relabel=False, is_query=True):
        img_paths = glob.glob(osp.join(dir_path,'*','*.tif'))   #原join(dir_path,'*','*.jpg')
        if is_query:
            camid = 0
        else:
            camid = 1
        pid_container = set()
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid, 1))
        return data
