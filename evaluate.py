from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys

from torch.backends import cudnn
import copy
import torch.nn as nn
import random

from reid.utils.metrics import R1_mAP_eval
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, copy_state_dict
from reid.utils.my_tools import *
from reid.models.resnet import build_resnet_backbone
from reid.models.layers import DataParallel
from reid.datasets import get_data
from reid.models.vit_pytorch import build_vit_backbone
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import copy
from config import cfg

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)

def main_worker(args):
    cudnn.benchmark = True
    log_name = 'test.txt'
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    # read parameters
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Create data loaders
    dataset_viper, num_classes_viper, train_loader_viper, test_loader_viper, _ = \
        get_data('viper', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)
    dataset_Grid, num_classes_Grid, train_loader_Grid, test_loader_Grid, _ = \
        get_data('Grid', args.data_dir, args.height, args.width, args.batch_size, args.workers,
                 args.num_instances)
    dataset_market, num_classes_market, train_loader_market, test_loader_market, init_loader_market = \
        get_data('market1501', args.data_dir, args.height, args.width, args.batch_size, args.workers,
                 args.num_instances)

    dataset_dukemtmc, num_classes_dukemtmc, train_loader_dukemtmc, test_loader_dukemtmc, init_loader_dukemtmc = \
        get_data('dukemtmc', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_cuhksysu, num_classes_cuhksysu, train_loader_cuhksysu, test_loader_cuhksysu, init_loader_chuksysu = \
        get_data('cuhk_sysu', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_msmt17, num_classes_msmt17, train_loader_msmt17, test_loader_msmt17, init_loader_msmt17 = \
        get_data('msmt17', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_cuhk03, num_classes_cuhk03, _, test_loader_cuhk03, _ = \
        get_data('CUHK03', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_cuhk02, num_classes_cuhk02, _, test_loader_cuhk02, _ = \
        get_data('CUHK02', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_prid, num_classes_prid, train_loader_prid, test_loader_prid, init_loader_prid = \
        get_data('prid2011', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)

    dataset_Duke, num_classes_Duke, _, test_loader_Duke, _ = \
        get_data('Occluded_Duke', args.data_dir, args.height, args.width, args.batch_size, args.workers,
                 args.num_instances)

    dataset_Reid, num_classes_Reid, _, test_loader_Reid, _ = \
        get_data('Occluded_REID', args.data_dir, args.height, args.width, args.batch_size, args.workers,
                 args.num_instances)


    # Create model
    #num_classes_total = num_classes_viper + num_classes_Grid + num_classes_cuhk03 + num_classes_cuhk02 + num_classes_Duke + num_classes_Reid + num_classes_prid + num_classes_market + num_classes_dukemtmc + num_classes_cuhksysu + num_classes_msmt17
    num_classes_total = num_classes_cuhk03 + num_classes_market + num_classes_dukemtmc + num_classes_cuhksysu + num_classes_msmt17
    # model = build_resnet_backbone(num_class=num_classes_total, depth='50x')
    model = build_vit_backbone(num_class=num_classes_total, args=args)
    old_model = build_vit_backbone(num_class=num_classes_total, args=args)
    model.cuda()
    old_model.cuda()

    # Load checkpoints
    if args.resume_current:
        current_checkpoint = load_checkpoint(args.resume_current)
        old_checkpoint = load_checkpoint(args.resume_old)
        # model.load_param(args.resume_working)
        copy_state_dict(current_checkpoint['state_dict'], model)
        copy_state_dict(old_checkpoint['state_dict'], old_model)


    epoch = current_checkpoint['epoch']

    # Setup evaluators
    names = ['viper', 'Grid', 'CUHK03', 'CUHK02', 'Occluded_Duke', 'Occluded_REID', 'prid2011', 'market', 'dukemtmc',
             'cuhksysu', 'msmt17']
    evaluators = [R1_mAP_eval(len(dataset_viper.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_Grid.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_cuhk03.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_cuhk02.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_Duke.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_Reid.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_prid.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_market.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_dukemtmc.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_cuhksysu.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_msmt17.query), max_rank=50, feat_norm=True)]
    test_loaders = [test_loader_viper, test_loader_Grid, test_loader_cuhk03, test_loader_cuhk02, test_loader_Duke, test_loader_Reid,
                    test_loader_prid, test_loader_market, test_loader_dukemtmc, test_loader_cuhksysu,
                    test_loader_msmt17]
    '''names = ['viper', 'Grid', 'CUHK02', 'Occluded_Duke', 'Occluded_REID', 'prid2011', 'market']
    evaluators = [R1_mAP_eval(len(dataset_viper.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_Grid.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_cuhk02.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_Duke.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_Reid.query), max_rank=50, feat_norm=True),
                  R1_mAP_eval(len(dataset_prid.query), max_rank=50, feat_norm=True)]
    test_loaders = [test_loader_viper, test_loader_Grid, test_loader_cuhk02, test_loader_Duke,
                    test_loader_Reid, test_loader_prid]'''
    training_phase = args.num_step
    # Start evaluating
    for evaluator, name, test_loader in zip(evaluators, names, test_loaders):
        #eval_func(epoch, evaluator, model, test_loader, name, old_model)
        eval_func(epoch, evaluator, model, test_loader, name, training_phase, old_model=old_model)

    print('finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-br', '--replay-batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    # model
    parser.add_argument('--size_train', type=int, default=[256,128])
    parser.add_argument('--stride_size', type=int, default=[16,16])
    parser.add_argument('--num_step', type=int, default=5)
    parser.add_argument('--num_task_experts', type=int, default=5)
    parser.add_argument('--total_experts', type=int, default=25)   #num_task_experts* num_task
    parser.add_argument('--prompt_param', type=int, default=[128,10,20])

    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume_current', type=str, default='./***/new_checkpoint_step_5.pth.tar',
                        metavar='PATH')
    parser.add_argument('--resume_old', type=str, default='./***/old_checkpoint_step_5.pth.tar',
                        metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')
    # path
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join('***'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default='***')
    parser.add_argument('--rr-gpu', action='store_true',
                        help="use GPU for accelerating clustering")
    main()