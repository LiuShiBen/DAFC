from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import copy
from ..utils.data import BaseImageDataset

class CUHK_SYSU(BaseImageDataset):


    def __init__(self, root, verbose=True, **kwargs):
        super(CUHK_SYSU, self).__init__()
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False, query=True)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.replay = 0
        self.sub_set()
        if verbose:
            print("=> CUHK-SYSU loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)


        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    @property
    def images_dir(self):
        return None


    def _process_dir(self, dir_path, relabel=False,query=False):

        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        #pattern = re.compile(r'id_([-\d]+)_img_([-\d]+)')
        pattern = re.compile(r'([-\d]+)_s([-\d]+)_([-\d]+)')
        pid_container = set()
        for img_path in img_paths:
            pid, _,_ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, _,frame = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored

            if relabel: pid = pid2label[pid]

            if query:
                dataset.append((img_path, pid, 1, 2))
            else:
                dataset.append((img_path, pid, 0, 2))

        return dataset

    def sub_set(self):
        results, bigger4_list, sub_train = {}, [], []
        for it in self.train:
            if it[1] not in results.keys():
                results[it[1]] = 1
            else:
                results[it[1]] += 1
        for key, value in results.items():
            if value >= 4:
                bigger4_list.append(key)
        for it in self.train:
            if it[1] in bigger4_list:
                sub_train.append(it)
        sub_train = self._relabels_incremental(sub_train, 1, is_mix=False)

        '''ramdomly select 500 ID'''
        global_id = []
        for index, (_, pid, cam, frame) in enumerate(sub_train):
            if pid not in global_id:
                global_id.append(pid)
        local_id = sorted(global_id)[:500]
        train_set = []
        for index, (img_path, pid, cam, frame) in enumerate(sub_train):
            if pid in local_id:
                # print(img_path)
                train_set.append((img_path, pid, cam, frame))
            else:
                1
        # print("train_set:", len(train_set), len(local_id), local_id)
        self.train = train_set

    def _relabels_incremental(self, samples, label_index, is_mix=False):
        '''
        reorder labels
        map labels [1, 3, 5, 7] to [0,1,2,3]
        '''
        ids = []
        pid2label = {}
        for sample in samples:
            ids.append(sample[label_index])
        # delete repetitive elments and order
        ids = list(set(ids))
        ids.sort()

        # reorder
        for sample in samples:
            sample = list(sample)
            pid2label[sample[label_index]] = ids.index(sample[label_index])
        new_samples = copy.deepcopy(samples)
        for i, sample in enumerate(samples):
            new_samples[i] = list(new_samples[i])
            new_samples[i][label_index] = pid2label[sample[label_index]]
        if is_mix:
            return samples, pid2label
        else:
            return new_samples
