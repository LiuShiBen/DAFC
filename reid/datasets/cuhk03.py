from __future__ import print_function, absolute_import
import os.path as osp
from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import read_json, write_json

class CUHK03(Dataset):
    """CUHK03.

    Reference:
        Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.

    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html#!>`_


    """
    dataset_url = None

    def __init__(
            self,
            root='',
            verbose=True,
            split_id=0,
            cuhk03_labeled=False,
            cuhk03_classic_split=False,
            **kwargs
    ):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = self.root
        print("self.dataset_dir:", self.dataset_dir)
        #self.download_dataset(self.dataset_dir, self.dataset_url)

        self.data_dir = osp.join(self.dataset_dir, 'cuhk03_release')
        self.raw_mat_path = osp.join(self.data_dir, 'cuhk-03.mat')

        self.imgs_detected_dir = osp.join(self.dataset_dir, 'images_detected')
        self.imgs_labeled_dir = osp.join(self.dataset_dir, 'images_labeled')

        self.split_classic_det_json_path = osp.join(
            self.dataset_dir, 'splits_classic_detected.json'
        )
        self.split_classic_lab_json_path = osp.join(
            self.dataset_dir, 'splits_classic_labeled.json'
        )

        self.split_new_det_json_path = osp.join(
            self.dataset_dir, 'splits_new_detected.json'
        )
        self.split_new_lab_json_path = osp.join(
            self.dataset_dir, 'splits_new_labeled.json'
        )

        self.split_new_det_mat_path = osp.join(
            self.dataset_dir, 'cuhk03_new_protocol_config_detected.mat'
        )
        self.split_new_lab_mat_path = osp.join(
            self.dataset_dir, 'cuhk03_new_protocol_config_labeled.mat'
        )

        self.preprocess_split()

        if cuhk03_labeled:
            split_path = self.split_classic_lab_json_path if cuhk03_classic_split else self.split_new_lab_json_path
        else:
            split_path = self.split_classic_det_json_path if cuhk03_classic_split else self.split_new_det_json_path

        splits = read_json(split_path)
        assert split_id < len(
            splits
        ), 'Condition split_id ({}) < len(splits) ({}) is false'.format(
            split_id, len(splits)
        )
        split = splits[split_id]

        training = [tuple(item + [item[1]]) for item in split['train']]
        query = [tuple(item + [item[1]]) for item in split['query']]
        gallery = [tuple(item + [item[1]]) for item in split['gallery']]
        '''ramdomly select 500 ID'''
        global_id = []
        for index, (_, pid, cam, frame) in enumerate(training):
            if pid not in global_id:
                global_id.append(pid)
        local_id = sorted(global_id)[:500]
        train = []
        for index, (img_path, pid, cam, frame) in enumerate(training):
            if pid in local_id:
                # print(img_path)
                train.append((img_path, pid, cam, frame))
            else:
                1

        self.train, self.query, self.gallery = train, query, gallery
        #self.num_train_pids = 500
        self.num_train_imgs, self.num_train_pids, self.num_train_cams = self._show_info(self.train)
        self.num_query_imgs, self.num_query_pids, self.num_query_cams = self._show_info(self.query)
        self.num_gallery_imgs, self.num_gallery_pids, self.num_gallery_cams = self._show_info(self.query)
        #self._show_info(self.train, self.query, self.gallery)
        if verbose:
            print("=> CUHK03 loaded")
            self.print_dataset_statistics(train, query, gallery)

    @property
    def images_dir(self):
        return None

    def _show_info(self, sample):

        def analyze(samples):
            pid_num = len(set([sample[1] for sample in samples]))
            cid_num = len(set([sample[2] for sample in samples]))
            sample_num = len(samples)
            return sample_num, pid_num, cid_num

        sample = analyze(sample)
        return sample[0], sample[1], sample[2]

    def preprocess_split(self):
        # This function is a bit complex and ugly, what it does is
        # 1. extract data from cuhk-03.mat and save as png images
        # 2. create 20 classic splits (Li et al. CVPR'14)
        # 3. create new split (Zhong et al. CVPR'17)
        if osp.exists(self.imgs_labeled_dir) \
                and osp.exists(self.imgs_detected_dir) \
                and osp.exists(self.split_classic_det_json_path) \
                and osp.exists(self.split_classic_lab_json_path) \
                and osp.exists(self.split_new_det_json_path) \
                and osp.exists(self.split_new_lab_json_path):
            return

        import h5py
        import imageio
        from scipy.io import loadmat

        mkdir_if_missing(self.imgs_detected_dir)
        mkdir_if_missing(self.imgs_labeled_dir)

        print(
            'Extract image data from "{}" and save as png'.format(
                self.raw_mat_path
            )
        )
        mat = h5py.File(self.raw_mat_path, 'r')

        def _deref(ref):
            return mat[ref][:].T

        def _process_images(img_refs, campid, pid, save_dir):
            img_paths = []  # Note: some persons only have images for one view
            for imgid, img_ref in enumerate(img_refs):
                img = _deref(img_ref)
                if img.size == 0 or img.ndim < 3:
                    continue  # skip empty cell
                # images are saved with the following format, index-1 (ensure uniqueness)
                # campid: index of camera pair (1-5)
                # pid: index of person in 'campid'-th camera pair
                # viewid: index of view, {1, 2}
                # imgid: index of image, (1-10)
                viewid = 1 if imgid < 5 else 2
                img_name = '{:01d}_{:03d}_{:01d}_{:02d}.png'.format(
                    campid + 1, pid + 1, viewid, imgid + 1
                )
                img_path = osp.join(save_dir, img_name)
                if not osp.isfile(img_path):
                    imageio.imwrite(img_path, img)
                img_paths.append(img_path)
            return img_paths

        def _extract_img(image_type):
            print('Processing {} images ...'.format(image_type))
            meta_data = []
            imgs_dir = self.imgs_detected_dir if image_type == 'detected' else self.imgs_labeled_dir
            for campid, camp_ref in enumerate(mat[image_type][0]):
                camp = _deref(camp_ref)
                num_pids = camp.shape[0]
                for pid in range(num_pids):
                    img_paths = _process_images(
                        camp[pid, :], campid, pid, imgs_dir
                    )
                    assert len(img_paths) > 0, \
                        'campid{}-pid{} has no images'.format(campid, pid)
                    meta_data.append((campid + 1, pid + 1, img_paths))
                print(
                    '- done camera pair {} with {} identities'.format(
                        campid + 1, num_pids
                    )
                )
            return meta_data

        meta_detected = _extract_img('detected')
        meta_labeled = _extract_img('labeled')

        def _extract_classic_split(meta_data, test_split):
            train, test = [], []
            num_train_pids, num_test_pids = 0, 0
            num_train_imgs, num_test_imgs = 0, 0
            for i, (campid, pid, img_paths) in enumerate(meta_data):

                if [campid, pid] in test_split:
                    for img_path in img_paths:
                        camid = int(
                            osp.basename(img_path).split('_')[2]
                        ) - 1  # make it 0-based
                        test.append((img_path, num_test_pids, camid))
                    num_test_pids += 1
                    num_test_imgs += len(img_paths)
                else:
                    for img_path in img_paths:
                        camid = int(
                            osp.basename(img_path).split('_')[2]
                        ) - 1  # make it 0-based
                        train.append((img_path, num_train_pids, camid))
                    num_train_pids += 1
                    num_train_imgs += len(img_paths)
            return train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs

        print('Creating classic splits (# = 20) ...')
        splits_classic_det, splits_classic_lab = [], []
        for split_ref in mat['testsets'][0]:
            test_split = _deref(split_ref).tolist()

            # create split for detected images
            train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
                _extract_classic_split(meta_detected, test_split)
            splits_classic_det.append(
                {
                    'train': train,
                    'query': test,
                    'gallery': test,
                    'num_train_pids': num_train_pids,
                    'num_train_imgs': num_train_imgs,
                    'num_query_pids': num_test_pids,
                    'num_query_imgs': num_test_imgs,
                    'num_gallery_pids': num_test_pids,
                    'num_gallery_imgs': num_test_imgs
                }
            )

            # create split for labeled images
            train, num_train_pids, num_train_imgs, test, num_test_pids, num_test_imgs = \
                _extract_classic_split(meta_labeled, test_split)
            splits_classic_lab.append(
                {
                    'train': train,
                    'query': test,
                    'gallery': test,
                    'num_train_pids': num_train_pids,
                    'num_train_imgs': num_train_imgs,
                    'num_query_pids': num_test_pids,
                    'num_query_imgs': num_test_imgs,
                    'num_gallery_pids': num_test_pids,
                    'num_gallery_imgs': num_test_imgs
                }
            )

        write_json(splits_classic_det, self.split_classic_det_json_path)
        write_json(splits_classic_lab, self.split_classic_lab_json_path)

        def _extract_set(filelist, pids, pid2label, idxs, img_dir, relabel):
            tmp_set = []
            unique_pids = set()
            for idx in idxs:
                img_name = filelist[idx][0]
                camid = int(img_name.split('_')[2]) - 1  # make it 0-based
                pid = pids[idx]
                if relabel:
                    pid = pid2label[pid]
                img_path = osp.join(img_dir, img_name)
                tmp_set.append((img_path, int(pid), camid))
                unique_pids.add(pid)
            return tmp_set, len(unique_pids), len(idxs)

        def _extract_new_split(split_dict, img_dir):
            train_idxs = split_dict['train_idx'].flatten() - 1  # index-0
            pids = split_dict['labels'].flatten()
            train_pids = set(pids[train_idxs])
            pid2label = {pid: label for label, pid in enumerate(train_pids)}
            query_idxs = split_dict['query_idx'].flatten() - 1
            gallery_idxs = split_dict['gallery_idx'].flatten() - 1
            filelist = split_dict['filelist'].flatten()
            train_info = _extract_set(
                filelist, pids, pid2label, train_idxs, img_dir, relabel=True
            )
            query_info = _extract_set(
                filelist, pids, pid2label, query_idxs, img_dir, relabel=False
            )
            gallery_info = _extract_set(
                filelist,
                pids,
                pid2label,
                gallery_idxs,
                img_dir,
                relabel=False
            )
            return train_info, query_info, gallery_info

        print('Creating new split for detected images (767/700) ...')
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_det_mat_path), self.imgs_detected_dir
        )
        split = [
            {
                'train': train_info[0],
                'query': query_info[0],
                'gallery': gallery_info[0],
                'num_train_pids': train_info[1],
                'num_train_imgs': train_info[2],
                'num_query_pids': query_info[1],
                'num_query_imgs': query_info[2],
                'num_gallery_pids': gallery_info[1],
                'num_gallery_imgs': gallery_info[2]
            }
        ]
        write_json(split, self.split_new_det_json_path)

        print('Creating new split for labeled images (767/700) ...')
        train_info, query_info, gallery_info = _extract_new_split(
            loadmat(self.split_new_lab_mat_path), self.imgs_labeled_dir
        )
        split = [
            {
                'train': train_info[0],
                'query': query_info[0],
                'gallery': gallery_info[0],
                'num_train_pids': train_info[1],
                'num_train_imgs': train_info[2],
                'num_query_pids': query_info[1],
                'num_query_imgs': query_info[2],
                'num_gallery_pids': gallery_info[1],
                'num_gallery_imgs': gallery_info[2]
            }
        ]
        write_json(split, self.split_new_lab_json_path)

'''class CUHK03(Dataset):
    url = 'https://docs.google.com/spreadsheet/viewform?usp=drive_web&formkey=dHRkMkFVSUFvbTJIRkRDLWRwZWpONnc6MA#gid=0'
    md5 = '728939e58ad9f0ff53e521857dd8fb43'

    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(CUHK03, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import h5py
        import hashlib
        import imageio
        from zipfile import ZipFile

        raw_dir = self.root
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'cuhk03_release.zip')
        print("fpath:", fpath)
        if osp.isfile(fpath) and \
          hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually from {} "
                               "to {}".format(self.url, fpath))

        # Extract the file
        exdir = osp.join(raw_dir, 'cuhk03_release')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)
        matdata = h5py.File(osp.join(exdir, 'cuhk-03.mat'), 'r')

        def deref(ref):
            return matdata[ref][:].T

        def dump_(refs, pid, cam, fnames):
            for ref in refs:
                img = deref(ref)
                if img.size == 0 or img.ndim < 2: break
                fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, cam, len(fnames))
                imageio.imsave(osp.join(images_dir, fname), img)
                fnames.append(fname)

        identities = []
        for labeled, detected in zip(
                matdata['labeled'][0], matdata['detected'][0]):
            labeled, detected = deref(labeled), deref(detected)
            assert labeled.shape == detected.shape
            for i in range(labeled.shape[0]):
                pid = len(identities)
                images = [[], []]
                dump_(labeled[i, :5], pid, 0, images[0])
                dump_(detected[i, :5], pid, 0, images[0])
                dump_(labeled[i, 5:], pid, 1, images[1])
                dump_(detected[i, 5:], pid, 1, images[1])
                identities.append(images)

        # Save meta information into a json file
        meta = {'name': 'cuhk03', 'shot': 'multiple', 'num_cameras': 2,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save training and test splits
        splits = []
        view_counts = [deref(ref).shape[0] for ref in matdata['labeled'][0]]
        vid_offsets = np.r_[0, np.cumsum(view_counts)]
        for ref in matdata['testsets'][0]:
            test_info = deref(ref).astype(np.int32)
            test_pids = sorted(
                [int(vid_offsets[i-1] + j - 1) for i, j in test_info])
            trainval_pids = list(set(range(vid_offsets[-1])) - set(test_pids))
            split = {'trainval': trainval_pids,
                     'query': test_pids,
                     'gallery': test_pids}
            splits.append(split)
        write_json(splits, osp.join(self.root, 'splits.json'))'''
