from __future__ import division, print_function, absolute_import
import glob
import os.path as osp
from scipy.io import loadmat


from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import read_json, write_json

class GRID(Dataset):
    """GRID.

    Reference:
        Loy et al. Multi-camera activity correlation analysis. CVPR 2009.

    URL: `<http://personal.ie.cuhk.edu.hk/~ccloy/downloads_qmul_underground_reid.html>`_
    
    Dataset statistics:
        - identities: 250.
        - images: 1275.
        - cameras: 8.
    """
    dataset_dir = 'Grid'
    dataset_url = 'http://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/underground_reid.zip'

    def __init__(self, root='', split_id=0, verbose=True, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.data_dir = self.root
        #self.download_dataset(self.dataset_dir, self.dataset_url)

        self.probe_path = osp.join(
            self.data_dir, 'probe'
        )
        self.gallery_path = osp.join(
            self.data_dir, 'gallery'
        )
        self.split_mat_path = osp.join(
            self.data_dir, 'features_and_partitions.mat'
        )
        self.split_path = osp.join(self.data_dir, 'splits.json')

        required_files = [
            self.data_dir, self.probe_path, self.gallery_path,
            self.split_mat_path
        ]
        #self.check_before_run(required_files)

        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, '
                'but expected between 0 and {}'.format(
                    split_id,
                    len(splits) - 1
                )
            )
        split = splits[split_id]

        train = split['train']
        query = split['query']
        gallery = split['gallery']

        train = [tuple(item + [item[1]]) for item in train]
        query = [tuple(item + [item[1]]) for item in query]
        gallery = [tuple(item + [item[1]]) for item in gallery]

        self.train, self.query, self.gallery = train, query, gallery

        self.num_train_imgs, self.num_train_pids, self.num_train_cams = self._show_info(self.train)
        self.num_query_imgs, self.num_query_pids, self.num_query_cams = self._show_info(self.query)
        self.num_gallery_imgs, self.num_gallery_pids, self.num_gallery_cams = self._show_info(self.query)
        #self._show_info(self.train, self.query, self.gallery)
        if verbose:
            print("=> GRID loaded")
            self.print_dataset_statistics(train, query, gallery)
    def _show_info(self, sample):

        def analyze(samples):
            pid_num = len(set([sample[1] for sample in samples]))
            cid_num = len(set([sample[2] for sample in samples]))
            sample_num = len(samples)
            return sample_num, pid_num, cid_num

        sample = analyze(sample)
        return sample[0], sample[1], sample[2]

    @property
    def images_dir(self):
        return self.data_dir
    def prepare_split(self):
        if not osp.exists(self.split_path):
            print('Creating 10 random splits')
            split_mat = loadmat(self.split_mat_path)
            trainIdxAll = split_mat['trainIdxAll'][0] # length = 10
            probe_img_paths = sorted(
                glob.glob(osp.join(self.probe_path, '*.jpeg'))
            )
            gallery_img_paths = sorted(
                glob.glob(osp.join(self.gallery_path, '*.jpeg'))
            )

            splits = []
            for split_idx in range(10):
                train_idxs = trainIdxAll[split_idx][0][0][2][0].tolist()
                assert len(train_idxs) == 125
                idx2label = {
                    idx: label
                    for label, idx in enumerate(train_idxs)
                }

                train, query, gallery = [], [], []

                # processing probe folder
                for img_path in probe_img_paths:
                    img_name = osp.basename(img_path)
                    img_idx = int(img_name.split('_')[0])
                    camid = int(
                        img_name.split('_')[1]
                    ) - 1 # index starts from 0
                    if img_idx in train_idxs:
                        train.append((img_path, idx2label[img_idx], camid))
                    else:
                        query.append((img_path, img_idx, camid))

                # process gallery folder
                for img_path in gallery_img_paths:
                    img_name = osp.basename(img_path)
                    img_idx = int(img_name.split('_')[0])
                    camid = int(
                        img_name.split('_')[1]
                    ) - 1 # index starts from 0
                    if img_idx in train_idxs:
                        train.append((img_path, idx2label[img_idx], camid))
                    else:
                        gallery.append((img_path, img_idx, camid))

                split = {
                    'train': train,
                    'query': query,
                    'gallery': gallery,
                    'num_train_pids': 125,
                    'num_query_pids': 125,
                    'num_gallery_pids': 900
                }
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file saved to {}'.format(self.split_path))
