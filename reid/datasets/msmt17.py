from __future__ import print_function, absolute_import
import os.path as osp

import re

def _pluck_msmt(list_file, subdir, pattern=re.compile(r'([-\d]+)_([-\d]+)_([-\d]+)')):
    with open(list_file, 'r') as f:
        lines = f.readlines()
    ret = []
    pids = []
    for line in lines:
        line = line.strip()
        fname = line.split(' ')[0]
        pid, _, cam = map(int, pattern.search(osp.basename(fname)).groups())
        if pid not in pids:
            pids.append(pid)
        ret.append((osp.join(subdir,fname), pid, cam,3))
    return ret, pids

def _pluck_train(list_file, subdir, pattern=re.compile(r'([-\d]+)_([-\d]+)_([-\d]+)')):
    with open(list_file, 'r') as f:
        lines = f.readlines()
    ret = []
    pids = []
    for line in lines:
        line = line.strip()
        fname = line.split(' ')[0]
        pid, _, cam = map(int, pattern.search(osp.basename(fname)).groups())
        if pid not in pids:
            pids.append(pid)
        ret.append((osp.join(subdir,fname), pid, cam,3))
    '''ramdomly select 500 ID'''
    global_id = []
    for index, (_, pid, cam, frame) in enumerate(ret):
        if pid not in global_id:
            global_id.append(pid)
    local_id = sorted(global_id)[:500]
    train_set = []
    for index, (img_path, pid, cam, frame) in enumerate(ret):
        #print("img_path", img_path)
        if pid in local_id:
            #
            train_set.append((img_path, pid, cam, frame))
        else:
            1
    # print("train_set:", len(train_set), len(local_id), local_id)
    return train_set, local_id

class Dataset_MSMT(object):
    def __init__(self, root):

        self.root = osp.join(root, 'MSMT17_V2')
        print("MSMT17_V2", self.root)
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    @property
    def images_dir(self):
        return None

    def load(self, verbose=True):
        exdir = self.root
        self.train, train_pids = _pluck_train(osp.join(exdir, 'list_train.txt'), osp.join(exdir, 'mask_train_v2'))
        self.val, val_pids = _pluck_train(osp.join(exdir, 'list_val.txt'), osp.join(exdir, 'mask_train_v2'))

        self.train = self.train + self.val
        self.replay = 0
        self.query, query_pids = _pluck_msmt(osp.join(exdir, 'list_query.txt'), osp.join(exdir, 'mask_test_v2'))
        self.gallery, gallery_pids = _pluck_msmt(osp.join(exdir, 'list_gallery.txt'), osp.join(exdir, 'mask_test_v2'))
        self.num_train_pids = len(list(set(train_pids).union(set(val_pids))))

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_pids, len(self.train)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(query_pids), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(gallery_pids), len(self.gallery)))

class MSMT17(Dataset_MSMT):

    def __init__(self, root, split_id=0, download=True):
        super(MSMT17, self).__init__(root)

        self.load()



