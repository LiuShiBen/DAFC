from __future__ import print_function, absolute_import
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import os.path as osp
import sys
import torch.optim
from torch.backends import cudnn
import torch.nn as nn
import random
from reid.datasets import get_data
from reid.utils.metrics import R1_mAP_eval
from reid.utils.logging import Logger
from reid.utils.serialization import save_checkpoint
from reid.utils.my_tools import *
from reid.models.vit_pytorch import build_vit_backbone
from reid.trainer import Trainer
from reid.utils.lr_scheduler import create_scheduler
from reid.utils.make_optimizer import make_optimizer

def initclassifier(model, num_class=[], data="market", init_loader=None, training_step=None, training_order="order-1"):
    if training_order == "order-1":
        if data == "cuhksysu":
            org_classifier1_params = model.classifier.weight.data
            model.classifier = nn.Linear(768, num_class[0] + num_class[1], bias=False)
            model.cuda()
            model.classifier.weight.data[:(num_class[0])].copy_(org_classifier1_params)
            # Initialize classifer with class centers
            class_centers = initial_classifier(model, init_loader, training_step)
            model.classifier.weight.data[num_class[0]:].copy_(class_centers)
            model.cuda()

        if data == "dukemtmc":
            org_classifier1_params = model.classifier.weight.data
            model.classifier = nn.Linear(768, num_class[0] + num_class[1] + num_class[2], bias=False)
            model.cuda()
            model.classifier.weight.data[:(num_class[0] + num_class[1])].copy_(org_classifier1_params)
            # Initialize classifer with class centers
            class_centers = initial_classifier(model, init_loader, training_step)
            model.classifier.weight.data[num_class[0] + num_class[1]:].copy_(class_centers)
            model.cuda()
        if data == "msmt17":
            org_classifier1_params = model.classifier.weight.data
            model.classifier = nn.Linear(768, num_class[0] + num_class[1] + num_class[2] + num_class[3], bias=False)
            model.cuda()
            model.classifier.weight.data[:(num_class[0] + num_class[1] + num_class[2])].copy_(org_classifier1_params)
            # Initialize classifer with class centers
            class_centers = initial_classifier(model, init_loader, training_step)
            model.classifier.weight.data[num_class[0] + num_class[1] + num_class[2]:].copy_(class_centers)
            model.cuda()

        if data == "CUHK03":

            org_classifier1_params = model.classifier.weight.data
            model.classifier = nn.Linear(768, num_class[0] + num_class[1] + num_class[2] + num_class[3] + num_class[4], bias=False)
            model.cuda()
            model.classifier.weight.data[:(num_class[0] + num_class[1] + num_class[2] + num_class[3])].copy_(org_classifier1_params)
            # Initialize classifer with class centers
            class_centers = initial_classifier(model, init_loader, training_step)
            model.classifier.weight.data[num_class[0] + num_class[1] + num_class[2] + num_class[3]:].copy_(class_centers)
            model.cuda()
    if training_order == "order-2":
        #print("Training-order:", training_order)
        if data == "msmt17":
            org_classifier1_params = model.classifier.weight.data
            model.classifier = nn.Linear(768, num_class[0] + num_class[1], bias=False)
            model.cuda()
            model.classifier.weight.data[:(num_class[0])].copy_(org_classifier1_params)
            # Initialize classifer with class centers
            class_centers = initial_classifier(model, init_loader, training_step)
            model.classifier.weight.data[num_class[0]:].copy_(class_centers)
            model.cuda()

        if data == "market1501":
            org_classifier1_params = model.classifier.weight.data
            model.classifier = nn.Linear(768, num_class[0] + num_class[1] + num_class[2], bias=False)
            model.cuda()
            model.classifier.weight.data[:(num_class[0] + num_class[1])].copy_(org_classifier1_params)
            # Initialize classifer with class centers
            class_centers = initial_classifier(model, init_loader, training_step)
            model.classifier.weight.data[num_class[0] + num_class[1]:].copy_(class_centers)
            model.cuda()

        if data == "cuhksysu":
            org_classifier1_params = model.classifier.weight.data
            model.classifier = nn.Linear(768, num_class[0] + num_class[1] + num_class[2] + num_class[3], bias=False)
            model.cuda()
            model.classifier.weight.data[:(num_class[0] + num_class[1] + num_class[2])].copy_(org_classifier1_params)
            # Initialize classifer with class centers
            class_centers = initial_classifier(model, init_loader, training_step)
            model.classifier.weight.data[num_class[0] + num_class[1] + num_class[2]:].copy_(class_centers)
            model.cuda()

        if data == "CUHK03":
            org_classifier1_params = model.classifier.weight.data
            model.classifier = nn.Linear(768, num_class[0] + num_class[1] + num_class[2] + num_class[3] + num_class[4],
                                          bias=False)
            model.cuda()
            model.classifier.weight.data[:(num_class[0] + num_class[1] + num_class[2] + num_class[3])].copy_(org_classifier1_params)
            # Initialize classifer with class centers
            class_centers = initial_classifier(model, init_loader, training_step)
            model.classifier.weight.data[num_class[0] + num_class[1] + num_class[2] + num_class[3]:].copy_(class_centers)
            model.cuda()


def lifelong_trainer(args, model, dataset_load, num_classes, evaluators, add_num=0, test_loaders=None, test_name=None, train_phase=1, old_model=None):

    # creat old model and Expand the dimension of classifier

    # initialize Opitimizer and lr
    optimizer = make_optimizer(args, model)
    lr_scheduler = create_scheduler(optimizer=optimizer, epochs=args.epochs, lr=args.lr)
    #print("num_classes:", num_classes, add_num)
    trainer = Trainer(args, model=model, optimizer=optimizer, num_classes=num_classes,
            data_loader_train=dataset_load[2], training_phase=train_phase, add_num=add_num, margin=args.margin)
    for epoch in range(args.epochs):
        #print("epoch:", epoch)
        dataset_load[2].new_epoch()
        trainer.train(epoch)
        lr_scheduler.step(epoch)
        #print("task1::", topk_indices)
        '''if (epoch % 10 == 0):
            for evaluator, name, test_loader in zip(evaluators, names, test_loaders):
                cmc, mAP_market = eval_func(epoch, evaluator, model, test_loader, name, training_phase=1, old_model=None, use_fsc=False)'''

        if (epoch == args.epochs - 1):
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
            }, True, fpath=osp.join(args.logs_dir, 'new_checkpoint_step_{}.pth.tar').format(train_phase))

            if old_model is not None:
                use_fsc = True
                save_checkpoint({
                    'state_dict': old_model.state_dict(),
                    'epoch': epoch + 1,
                }, True, fpath=osp.join(args.logs_dir, 'old_checkpoint_step_{}.pth.tar').format(train_phase))
            else:
                use_fsc = False
            '''for evaluator, name, test_loader in zip(evaluators, test_name, test_loaders):
                #cmc, mAP_market = eval_func(epoch, evaluator, model, test_loader, name, training_phase=1, old_model=old_model, use_fsc=use_fsc)
                eval_func(epoch, evaluator, model, test_loader, name, training_phase = train_phase, old_model=old_model, use_fsc=use_fsc)'''
            #print('{} Finish Training  '.format(test_name[-1]))
    # 冻结当前任务的专家
    expert_ids = torch.arange((train_phase-1)*5, train_phase*5, 1).tolist()
    print(expert_ids)
    for expert_id in expert_ids:
        for param in model.tws.experts[expert_id].parameters():
            param.requires_grad = False
def main_worker(args):
    cudnn.benchmark = True
    log_name = 'train.txt'
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders  order1
    #(dataset_cuhksysu, num_classes, train_loader, test_loader, init_loader)
    market_load= get_data('market1501', args.data_dir, args.height, args.width, args.batch_size, args.workers,args.num_instances)
    chuksysu_load = get_data('cuhk_sysu', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)
    dukemtmc_load = get_data('dukemtmc', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)
    msmt17_load = get_data('msmt17', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)
    cuhk03_load = get_data('CUHK03', args.data_dir, args.height, args.width, args.batch_size, args.workers, args.num_instances)
    #

    print('Using {} GPUs for training'.format(torch.cuda.device_count()))

    datasets_load = []
    datasets_name = []
    # Training order-1
    if args.training_order == "order-1":
        datasets_load = [market_load, chuksysu_load, dukemtmc_load, msmt17_load, cuhk03_load]
        datasets_name = ["market1501", "cuhksysu", "dukemtmc", "msmt17", "CUHK03"]

    #Training order-2
    if args.training_order == "order-2":
        #print("Training-order:", args.training_order)
        datasets_load = [dukemtmc_load, msmt17_load, market_load, chuksysu_load, cuhk03_load]
        datasets_name = ["dukemtmc", "msmt17", "market1501", "cuhksysu", "CUHK03"]

    import time
    start = time.time()
    local_time = time.localtime(start)
    start = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    print("start time:", start)

    # Create model

    model = build_vit_backbone(datasets_load[0][1], args)
    model.cuda()

    "----------Traing Step 1-----------"
    evaluators = [R1_mAP_eval(len(datasets_load[0][0].query), max_rank=50, feat_norm=True)]
    num_classes = datasets_load[0][1]
    test_loaders = [datasets_load[0][3]]
    lifelong_trainer(args, model, datasets_load[0], num_classes, evaluators = evaluators, test_loaders=test_loaders,
                     test_name=datasets_name[0:1], train_phase=1, old_model=None)
    print('----------{} Finish Training----------'.format(datasets_name[0]))

    "----------Traing Step 2-----------"
    evaluators.append(R1_mAP_eval(len(datasets_load[1][0].query), max_rank=50, feat_norm=True))
    num_classes = datasets_load[0][1] + datasets_load[1][1]
    init_classes = [datasets_load[0][1], datasets_load[1][1]]
    add_num = datasets_load[0][1]
    old_model = build_vit_backbone(datasets_load[0][1], args)
    #Expand the dimension of classifier
    initclassifier(model, num_class=init_classes, data=datasets_name[1],
                   init_loader=datasets_load[1][-1], training_step=2, training_order=args.training_order)
    initclassifier(old_model, num_class=init_classes, data=datasets_name[1],
                   init_loader=datasets_load[1][-1], training_step=2, training_order=args.training_order)
    model.cuda()
    #Create old frozen model
    tmp_state_dict = old_model.state_dict()
    for k in model.state_dict().keys():
        tmp_state_dict[k] = model.state_dict()[k]
    old_model.load_state_dict(tmp_state_dict)
    old_model = old_model.cuda()

    test_loaders = [datasets_load[0][3], datasets_load[1][3]]
    #print("sucessful:")
    lifelong_trainer(args, model, datasets_load[1], num_classes, evaluators = evaluators, add_num=add_num, test_loaders=test_loaders,
                     test_name=datasets_name[0:2], train_phase=2, old_model=old_model)
    print('----------{} Finish Training----------'.format(datasets_name[1]))

    "----------Traing Step 3-----------"
    print('Traing Step 3')
    evaluators.append(R1_mAP_eval(len(datasets_load[2][0].query), max_rank=50, feat_norm=True))
    num_classes = datasets_load[0][1] + datasets_load[1][1] + datasets_load[2][1]
    add_num = datasets_load[0][1] + datasets_load[1][1]
    init_classes = [datasets_load[0][1], datasets_load[1][1], datasets_load[2][1]]
    print('Traing Step 3')
    #Model space consolidation
    alpha = 1 / 2
    tmp_state_dict = model.state_dict()
    for k in model.state_dict().keys():
        tmp_state_dict[k] = alpha * model.state_dict()[k] + (1 - alpha) * old_model.state_dict()[k]
    model.load_state_dict(tmp_state_dict)
    print('Traing Step 3')
    # Expand the dimension of classifier
    initclassifier(model, num_class=init_classes, data=datasets_name[2],
                   init_loader=datasets_load[2][-1], training_step=3, training_order=args.training_order)
    initclassifier(old_model, num_class=init_classes, data=datasets_name[2],
                   init_loader=datasets_load[2][-1], training_step=3, training_order=args.training_order)
    print('Traing Step 3', datasets_name[2])
    # Create old frozen model
    tmp_state_dict = old_model.state_dict()
    for k in model.state_dict().keys():
        tmp_state_dict[k] = model.state_dict()[k]
    old_model.load_state_dict(tmp_state_dict)
    old_model = old_model.cuda()

    test_loaders = [datasets_load[0][3], datasets_load[1][3], datasets_load[2][3]]
    lifelong_trainer(args, model, datasets_load[2], num_classes, evaluators = evaluators, add_num=add_num, test_loaders=test_loaders,
                     test_name=datasets_name[0:3], train_phase=3, old_model=old_model)
    print('----------{} Finish Training----------'.format(datasets_name[2]))

    "----------Traing Step 4-----------"
    evaluators.append(R1_mAP_eval(len(datasets_load[3][0].query), max_rank=50, feat_norm=True))
    num_classes = datasets_load[0][1] + datasets_load[1][1] + datasets_load[2][1] + datasets_load[3][1]
    add_num = datasets_load[0][1] + datasets_load[1][1] + datasets_load[2][1]
    init_classes = [datasets_load[0][1], datasets_load[1][1], datasets_load[2][1], datasets_load[3][1]]
    # model space consolidation
    alpha = 1 / 3
    tmp_state_dict = model.state_dict()
    for k in model.state_dict().keys():
        tmp_state_dict[k] = alpha * model.state_dict()[k] + (1 - alpha) * old_model.state_dict()[k]
    model.load_state_dict(tmp_state_dict)

    # Expand the dimension of classifier
    initclassifier(model, num_class=init_classes, data=datasets_name[3],
                   init_loader=datasets_load[3][-1], training_step=4, training_order=args.training_order)
    initclassifier(old_model, num_class=init_classes, data=datasets_name[3],
                   init_loader=datasets_load[3][-1], training_step=4, training_order=args.training_order)

    # Create old frozen model
    tmp_state_dict = old_model.state_dict()
    for k in model.state_dict().keys():
        tmp_state_dict[k] = model.state_dict()[k]
    old_model.load_state_dict(tmp_state_dict)
    old_model = old_model.cuda()

    test_loaders = [datasets_load[0][3], datasets_load[1][3], datasets_load[2][3], datasets_load[3][3]]
    lifelong_trainer(args, model, datasets_load[3], num_classes, evaluators = evaluators, add_num=add_num, test_loaders=test_loaders,
                     test_name=datasets_name[0:4], train_phase=4, old_model=old_model)
    print('----------{} Finish Training----------'.format(datasets_name[3]))

    "----------Traing Step 5-----------"
    evaluators.append(R1_mAP_eval(len(datasets_load[4][0].query), max_rank=50, feat_norm=True))
    num_classes = datasets_load[0][1] + datasets_load[1][1] + datasets_load[2][1] + datasets_load[3][1] + datasets_load[4][1]
    add_num = datasets_load[0][1] + datasets_load[1][1] + datasets_load[2][1] + datasets_load[3][1]
    init_classes = [datasets_load[0][1], datasets_load[1][1], datasets_load[2][1], datasets_load[3][1], datasets_load[4][1]]
    # model space consolidation
    alpha = 1 / 4
    tmp_state_dict = model.state_dict()
    for k in model.state_dict().keys():
        tmp_state_dict[k] = alpha * model.state_dict()[k] + (1 - alpha) * old_model.state_dict()[k]
    model.load_state_dict(tmp_state_dict)

    # Expand the dimension of classifier
    initclassifier(model, num_class=init_classes, data=datasets_name[4],
                   init_loader=datasets_load[4][-1], training_step=5, training_order=args.training_order)
    initclassifier(old_model, num_class=init_classes, data=datasets_name[4],
                   init_loader=datasets_load[4][-1], training_step=5, training_order=args.training_order)

    # Create old frozen model
    tmp_state_dict = old_model.state_dict()
    for k in model.state_dict().keys():
        tmp_state_dict[k] = model.state_dict()[k]
    old_model.load_state_dict(tmp_state_dict)
    old_model = old_model.cuda()

    test_loaders = [datasets_load[0][3], datasets_load[1][3], datasets_load[2][3], datasets_load[3][3], datasets_load[4][3]]
    lifelong_trainer(args, model, datasets_load[4], num_classes, evaluators = evaluators, add_num=add_num, test_loaders=test_loaders,
                     test_name=datasets_name[0:5], train_phase=5, old_model=old_model)

    end = time.time()
    local_time = time.localtime(end)
    end = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    print("start time:", end)
    print('----------{} Finish Training----------'.format(datasets_name[4]))

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    #print('cuda:', torch.cuda.is_available())
    main_worker(args)
if __name__ == '__main__':
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
    # model
    parser.add_argument('--size_train', type=int, default=[256,128])
    parser.add_argument('--stride_size', type=int, default=[16,16])
    parser.add_argument('--num_step', type=int, default=5)
    parser.add_argument('--num_task_experts', type=int, default=5)
    parser.add_argument('--total_experts', type=int, default=25)   #num_task_experts* num_task
    parser.add_argument('--prompt_param', type=int, default=[128,8,20])

    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--optimizer_name', type=str, default='Adam')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--lr', type=float, default=0.000005,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[20, 40],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--training_order', type=str, default="order-1")  #order-1/order-2
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join('***'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--rr-gpu', action='store_true',
                        help="use GPU for accelerating clustering")
    parser.add_argument("--gpu", default=[0], type=int)
    #Fusion
    main()
