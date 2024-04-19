import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn 
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
# import tensorboard_logger as tb_logger
from utils.helper import set_seed, args_print
from utils.logger import Logger
import numpy as np
from model import GCCL
from utils.utils_algo import *
from utils.utils_loss import partial_loss, SupConLoss
from utils.graphsst5 import load_loader
from datetime import datetime

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,7"
torch.set_printoptions(precision=2, sci_mode=False)
# mp.set_start_method('fork')

parser = argparse.ArgumentParser(description='PyTorch implementation of ICLR 2022 Oral paper PiCO')
parser.add_argument('--dataset', default='Graph_SST5', type=str, 
                    choices=['Graph_SST5','Graph_Twitter','COLLAB','REDDIT_MULTI_5K'],
                    help='dataset name') 
parser.add_argument('--partial_rate', default=0.1, type=float, 
                    help='ambiguity level (q)')
parser.add_argument('--uniform', default=0, type=int,
                    help='uniform label')
parser.add_argument('--modelPath', default='./data/model.tar', type=str,
                    help='model path')

parser.add_argument('--num-class', default=5, type=int,
                    help='number of class')
parser.add_argument('--exp-dir', default='experiment/GCCL', type=str,
                    help='experiment directory for saving checkpoints and logs')

parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-j', '--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('--start-epoch', default=0, type=int, 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr_decay_epochs', type=str, default='700,800,900',
                    help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=True,
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://localhost:10003', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=123, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', default=True,action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')
parser.add_argument('--moco_queue', default=8192, type=int, 
                    help='queue size; number of negative samples')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='momentum for updating momentum encoder')
parser.add_argument('--proto_m', default=0.99, type=float,
                    help='momentum for computing the momving average of prototypes')
parser.add_argument('--conf_ema_range', default='0.95,0.8', type=str,
                    help='pseudo target updating coefficient (phi)')
# parser.add_argument('--conf_ema_range', default='0.9999,0.95', type=str,
#                     help='pseudo target updating coefficient (phi)')
parser.add_argument('--prot_start', default=1, type=int, 
                    help = 'Start Prototype Updating')
# parser.add_argument('--hierarchical', action='store_true', 
#                     help='for CIFAR-100 fine-grained training')
parser.add_argument('--loss_weight', default=0.5, type=float,
                    help='contrastive loss weight')

parser.add_argument('--neg_loss_weight', default=1., type=float,
                    help='negative loss weight')
parser.add_argument('--channels', default='128', type=int,
                    help='use for split graph')
parser.add_argument('--epochs', default=50, type=int, 
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--ratio', default=0.7, type=float,
                    help='causal_ratio')
parser.add_argument('--ngpus', default=1, type=int, help='ngpus')

def main():
    args = parser.parse_args()
    args.conf_ema_range = [float(item) for item in args.conf_ema_range.split(',')]
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    print(args)

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    model_path = 'ds_{ds}_ep_{ep}_bs_{bs}_pr_{pr}_rt_{rt}_ng_{ng}_nw_{nw}_sd_{seed}'.format(
                                            ds=args.dataset,
                                            ep=args.epochs,
                                            bs=args.batch_size,
                                            pr=args.partial_rate,
                                            rt=args.ratio,
                                            ng=args.ngpus,
                                            nw=args.neg_loss_weight,
                                            seed=args.seed)

    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.exp_dir = os.path.join(args.exp_dir, args.dataset)
    args.exp_dir = os.path.join(args.exp_dir, model_path)
    args.exp_dir = os.path.join(args.exp_dir, datetime_now)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    
    ngpus_per_node = torch.cuda.device_count()
    if args.ngpus > 0: ngpus_per_node = args.ngpus
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = True
    args.gpu = gpu
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.dataset == 'Graph_SST5':
        args.in_channels = 768
        args.hid_channels = 128
        args.feat_dim = 128
        args.num_class = 5
        train_loader, train_givenY, train_sampler, test_loader, val_loader = load_loader(args.dataset, partial_rate=args.partial_rate, batch_size=args.batch_size, 
                                                                                         uniform=args.uniform, in_channels=args.in_channels, modelPath=args.modelPath)
    elif args.dataset == 'Graph_Twitter':
        args.in_channels = 768
        args.hid_channels = 128
        args.feat_dim = 128
        args.num_class = 5
        train_loader, train_givenY, train_sampler, test_loader, val_loader = load_loader(args.dataset, partial_rate=args.partial_rate, batch_size=args.batch_size, 
                                                                                         uniform=args.uniform, in_channels=args.in_channels, modelPath=args.modelPath)
    elif args.dataset == 'COLLAB':
        args.in_channels = 768
        args.hid_channels = 128
        args.feat_dim = 128
        args.num_class = 5
        train_loader, train_givenY, train_sampler, test_loader, val_loader = load_loader(args.dataset, partial_rate=args.partial_rate, batch_size=args.batch_size, 
                                                                                         uniform=args.uniform, in_channels=args.in_channels, modelPath=args.modelPath)
    elif args.dataset == 'REDDIT_MULTI_5K':
        args.in_channels = 32
        args.hid_channels = 128
        args.feat_dim = 128
        args.num_class = 5
        train_loader, train_givenY, train_sampler, test_loader, val_loader = load_loader(args.dataset, partial_rate=args.partial_rate, batch_size=args.batch_size, 
                                                                                         uniform=args.uniform, in_channels=args.in_channels, modelPath=args.modelPath)
    else:
        raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")
    # this train loader is the partial label training loader

    # create model
    model = GCCL(args)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print('Calculating uniform targets...')
    tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
    confidence = train_givenY.float()/tempY
    confidence = confidence.cuda()
    # calculate confidence

    loss_fn = partial_loss(confidence)
    loss_cont_fn = SupConLoss()
    # set loss functions (with pseudo-targets maintained)

    if args.gpu==0:
        file_logger = Logger.init_logger(filename=args.exp_dir + '/_output_.log')
        args_print(args, file_logger)
    else:
        file_logger = None

    print('\nStart Training\n')

    bestVal,coTest = 0,0
    mmc = 0 #mean max confidence
    for epoch in range(args.start_epoch, args.epochs):
        is_best = False
        start_upd_prot = epoch>=args.prot_start
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        adjust_learning_rate(args, optimizer, epoch)
        train(train_loader, model, loss_fn, loss_cont_fn, optimizer, epoch, args, start_upd_prot)
        loss_fn.set_conf_ema_m(epoch, args)
        # reset phi
        
        acc_train = 0
        acc_train = get_train_acc(model, train_loader, args)*100.
        acc_test = get_test_acc(model, test_loader, args)*100.
        acc_val = get_test_acc(model, val_loader, args)*100.
        mmc = loss_fn.confidence.max(dim=1)[0].mean()

        if acc_val > bestVal:
            bestVal = acc_val
            coTest = acc_test
            is_best = True
        
        if args.gpu == 0:
            file_logger.info("Epoch [{:3d}/{:d}] Train_ACC:{:.3f}  Test_ACC:{:.3f}  Val_ACC:{:.3f}  lr:{:.3f}  mmc:{:.3f}".format(
                        epoch, args.epochs, acc_train, acc_test, acc_val, optimizer.param_groups[0]['lr'], mmc))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
            best_file_name='{}/checkpoint_best.pth.tar'.format(args.exp_dir))
    if args.gpu == 0:
        file_logger.info("Best_Val_ACC:{:.3f}  Corresponding_Test_ACC:{:.3f}".format(
                        bestVal, coTest))

def train(train_loader, model, loss_fn, loss_cont_fn, optimizer, epoch, args, start_upd_prot=False):

    # # switch to train mode
    model.train()
    
    end = time.time()

    for i, (graphs, labels, true_labels, index) in enumerate(train_loader):

        X, Y, index = graphs.cuda(), labels.cuda(), index.cuda()
        Y_true = true_labels.long().detach().cuda()
        # for showing training accuracy and will not be used when training

        cls_out_q, cls_out_k, features_cont, pseudo_target_cont, score_prot = model(X, Y, args)
        batch_size = cls_out_q.shape[0]
        pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)

        if start_upd_prot:
            loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=Y)
            # warm up ended
        
        if start_upd_prot:
            mask = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().cuda()
            # get positive set by contrasting predicted labels
        else:
            mask = None
            # Warmup using MoCo

        # contrastive loss
        loss_cont = loss_cont_fn(features=features_cont, mask=mask, batch_size=batch_size)
        # classification loss
        loss_cls = loss_fn(cls_out_q, index)

        loss = loss_cls + args.loss_weight * loss_cont

        # negative samples loss
        loss_neg = torch.sum(cls_out_k ** 2) / cls_out_k.shape[0]
        loss += loss_neg * args.neg_loss_weight


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
    
def get_train_acc(model, train_loader, args):
    with torch.no_grad():
        model.eval()
        acc = 0
        for i, (graphs, labels, true_labels, index) in enumerate(train_loader):
            X = graphs.cuda()
            Y_true = true_labels.long().detach().cuda()
            outputs = model(X, eval_only=True)
            acc += torch.sum(outputs.argmax(-1).view(-1) == Y_true.view(-1))
        acc = float(acc) / len(train_loader.dataset)
    return acc

def get_test_acc(model, test_loader, args):
    with torch.no_grad():     
        model.eval()
        acc = 0
        for graphs in test_loader:
            X = graphs.cuda()
            Y_true = graphs.y.long().detach().cuda()
            outputs = model(X, eval_only=True)
            acc += torch.sum(outputs.argmax(-1).view(-1) == Y_true.view(-1))
        acc = float(acc) / len(test_loader.dataset)
    return acc
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)

if __name__ == '__main__':
    main()
