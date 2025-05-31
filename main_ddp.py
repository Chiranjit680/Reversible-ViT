"""Train CIFAR10 with PyTorch."""
import argparse
import os
import json
from functools import partial

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler

from build_model import build_model
from data import get_data_loader_ddp

import timm
import time
import numpy as np
import pandas as pd
import csv
from timm.loss import SoftTargetCrossEntropy

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import warnings
import logging
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'
logging.captureWarnings(True)

def setup(rank, world_size):    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")

parser.add_argument("--expt_name", type=str, help="Experiment Name", required=True)

# Optimizer options
parser.add_argument("--lr", default=5e-4, type=float, help="learning rate")
parser.add_argument("--bs", default=128, type=int, help="batch size")
parser.add_argument("--deit_scheme", action="store_true")

parser.add_argument(
    "--epochs", default=200, type=int, help="number of classes in the dataset"
)

# Transformer options
parser.add_argument("--model", default="vit", type=str, help="model name")
parser.add_argument(
    "--embed_dim",
    default=384,
    type=int,
    help="embedding dimension of the transformer",
)
parser.add_argument(
    "--n_head", default=6, type=int, help="number of heads in the transformer"
)
parser.add_argument(
    "--depth", default=12, type=int, help="number of transformer blocks"
)
parser.add_argument(
    "--patch_size", default="(4, 4)", help="patch size in patchification"
)
parser.add_argument("--image_size", default="(32, 32)", help="input image size")
parser.add_argument(
    "--num_classes",
    default=10,
    type=int,
    help="number of classes in the dataset",
)
parser.add_argument("--token_mixer", default="attention", type=str, help="Token Mixer")
parser.add_argument("--pool_size", default=3, type=int, help="Pooling Token Mixer pool size")
parser.add_argument("--num_registers", default=0, type=int, help="# of registers to be used")


parser.add_argument("--dataset", required=True) 

# To train the reversible architecture with or without reversible backpropagation
parser.add_argument(
    "--vanilla_bp",
    default=False,
    type=bool,
    help="whether to use reversible backpropagation or not",
)
parser.add_argument(
    "--pareprop",
    default=False,
    type=bool,
    help="whether to use fast, parallel reversible backpropagation or not",
)
parser.add_argument(
    "--amp",
    default=False,
    type=bool,
    help="whether to use mixed precision training or not",
)

args = parser.parse_args()
args.image_size = eval(args.image_size)
args.patch_size = eval(args.patch_size)

def main_worker(rank, world_size, args):

    assert (args.bs % world_size) == 0
    assert (args.bs / world_size) % 2 == 0

    print(f"Rank {rank} spawned")
    args.world_size = world_size

    args.bs = int(args.bs / world_size)

    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    trainloader, testloader = get_data_loader_ddp(args=args, rank=rank, world_size=world_size, pin_memory=True, num_workers=world_size)
    
    model = build_model(args)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, static_graph=True)
    # Need to dig into why this static graph is needed, something to do with forward pass in backward for rev network

    # Criterion, LR & Mixup
    mixup_fn = None
    if args.deit_scheme:
        print("DEIT SCHEME BEING USED")
        criterion = SoftTargetCrossEntropy()
        args.lr = 5e-4 * (args.bs * world_size/512)

        mixup_fn = timm.data.Mixup(
                mixup_alpha=0.8, cutmix_alpha=1.0,
                prob=1, switch_prob=0.5, mode="batch",
                label_smoothing=0.1, num_classes=args.num_classes)
    else:
        print("DEIT SCHEME NOT BEING USED")
        criterion = nn.CrossEntropyLoss()

    # LR
    if args.lr != 5e-4 * (args.bs * world_size/512):
        print(f"Base LR of {args.lr} does not match 5e-4 * ({args.bs * world_size}/512) = {5e-4 * (args.bs * world_size/512)}")
    else:
        print(f"LR set to {args.lr} as per 5e-4 * ({args.bs * world_size}/512)")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=0.05)

    # Scheduler
    if args.deit_scheme:
        warmup_epochs = 5
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8 / args.lr, total_iters=warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs - warmup_epochs), eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        print(f"Using cosine scheduler with {warmup_epochs} warmup epochs followed by cosine annealing")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    scaler = GradScaler()

    with open(f"expt_logs/{args.expt_name}/metadata.json", "w") as f:
        args_dict = vars(args)
        json.dump(args_dict, f, indent=4)

    for epoch in range(args.epochs):
        trainloader.sampler.set_epoch(epoch)

        start_time = time.time()

        train_loss, batch_time = train(epoch, model, trainloader, mixup_fn, criterion, optimizer, scaler, device)
        train_epoch_end_time = time.time()

        test_loss, test_acc1, test_acc5 = test(epoch, model, testloader, criterion, args, device)
        test_epoch_end_time = time.time()

        if rank == 0:
            print("----------------------------------------------------------------------------------------------------------------------------")
            print(f"Epoch [{epoch}/{args.epochs}] | Train Time: {train_epoch_end_time - start_time:.2f}s | Test Time : {test_epoch_end_time - train_epoch_end_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Mean Batch Time: {batch_time:.2f}")
            print(f"Test Loss:  {test_loss:.4f} | Acc@1: {test_acc1:.2f} | Acc@5: {test_acc5:.2f}")
            print("----------------------------------------------------------------------------------------------------------------------------")


            with open(f'expt_logs/{args.expt_name}/{args.expt_name}_logs.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, train_loss, test_loss, test_acc1, test_acc5])


        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict()
            }
            torch.save(checkpoint, f"./expt_logs/{args.expt_name}/ckp_epoch_{epoch}.pth")
            print(f"Checkpoint saved: ./expt_logs/{args.expt_name}/ckp_epoch_{epoch}.pth")

            if epoch != 0:
                os.system(f"rm -rf ./expt_logs/{args.expt_name}/ckp_epoch_{epoch-10}.pth")
                print(f"Checkpoint deleted: ./expt_logs/{args.expt_name}/ckp_epoch_{epoch-10}.pth")
            
            
        scheduler.step()

    dist.destroy_process_group()

# Training
def train(epoch, model, trainloader, mixup_fn, criterion, optimizer, scaler, device):
    print("\nTraining Epoch: %d" % epoch)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(trainloader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    time_ = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        data_time.update(time.time() - time_)
        # We do not need to specify AMP autocast in forward pass here since
        # that is taken care of already in the forward of individual modules.
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            if inputs.shape[0] % 2:
                print("One sample dropped from batch")
                inputs, targets = inputs[:-1], targets[:-1]
            inputs, targets = mixup_fn(inputs, targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        # standard pytorch AMP training setup
        # scaler also works without amp training.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        batch_time.update(time.time() - time_)
        time_ = time.time()

        if batch_idx % 10 == 0:
            progress.display(batch_idx + 1)

    return losses.avg, batch_time.avg
           
def test(epoch, model, testloader, criterion, args, device):

    print("\nTesting Epoch: %d" % epoch)

    batch_time = AverageMeter('Time', ':6.3f', None)
    losses = AverageMeter('Loss', ':.4e', None)
    top1 = AverageMeter('Acc@1', ':6.2f', "avg")
    top5 = AverageMeter('Acc@5', ':6.2f', "avg")
    progress = ProgressMeter(
        len(testloader) + (len(testloader.sampler) * args.world_size < len(testloader.dataset)),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    model.eval()

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            time_ = time.time()
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_idx = batch_idx + base_progress
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, F.one_hot(targets, num_classes=args.num_classes).float())

                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))

                batch_time.update(time.time() - time_)
                time_ = time.time()

                if batch_idx % 1 == 0:
                    progress.display(batch_idx + 1)

    run_validate(testloader)

    top1.all_reduce()
    top5.all_reduce()

    if (len(testloader.sampler) * args.world_size < len(testloader.dataset)):
        aux_val_dataset = torch.utils.data.Subset(testloader.dataset, range(len(testloader.sampler) * args.world_size, len(testloader.dataset)))
        aux_testloader = torch.utils.data.DataLoader(aux_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.world_size, pin_memory=True)
        run_validate(aux_testloader, len(testloader))

    progress.display_summary()

    return losses.avg, top1.avg, top5.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type="avg"):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=torch.device(f"cuda:{torch.cuda.current_device()}"))
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is None:
            fmtstr = ''
        elif self.summary_type == "avg":
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type == "sum":
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type == "count":
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

import random

import numpy as np

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if not os.path.exists("expt_logs"):
    os.makedirs("expt_logs")

if not os.path.exists(f"expt_logs/{args.expt_name}"):
    os.makedirs(f"expt_logs/{args.expt_name}")
    with open(f'expt_logs/{args.expt_name}/{args.expt_name}_logs.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "test_acc", "test_acc5"])


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print("World Size: ", world_size)
    torch.multiprocessing.spawn(main_worker, nprocs=world_size, args=(world_size, args), join=True)

# based on https://github.com/kentaroy47/vision-transformers-cifar10
