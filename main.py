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
from data import get_data_loader

import timm
import time
import numpy as np
import pandas as pd
from timm.loss import SoftTargetCrossEntropy

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


device = "cuda" if torch.cuda.is_available() else "cpu"

# Data
trainloader, testloader = get_data_loader(args)

model = build_model(args)
model = model.to(device)

# Criterion, LR & Mixup
mixup_fn = None
if args.deit_scheme:
    print("DEIT SCHEME BEING USED")
    criterion = SoftTargetCrossEntropy()
    args.lr = 5e-4 * (args.bs/512)

    mixup_fn = timm.data.Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0,
            prob=1, switch_prob=0.5, mode="batch",
            label_smoothing=0.1, num_classes=args.num_classes)
else:
    print("DEIT SCHEME NOT BEING USED")
    criterion = nn.CrossEntropyLoss()

# LR
if args.lr != 5e-4 * (args.bs/512):
    print(f"Base LR of {args.lr} does not match 5e-4 * ({args.bs}/512) = {5e-4 * (args.bs/512)}")
else:
    print(f"LR set to {args.lr} as per 5e-4 * ({args.bs}/512)")

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


# Training
def train(epoch, logs):
    print("\nTraining Epoch: %d" % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    batch_times = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        start_event_batch = torch.cuda.Event(enable_timing=True)
        end_event_batch = torch.cuda.Event(enable_timing=True)
        start_event_batch.record()

        # We do not need to specify AMP autocast in forward pass here since
        # that is taken care of already in the forward of individual modules.
        inputs, targets = inputs.to(device), targets.to(device)

        if mixup_fn is not None:
            if inputs.shape[0] % 2:
                print("One sample dropped from batch")
                inputs, targets = inputs[:-1], targets[:-1]
            inputs, targets = mixup_fn(inputs, targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # standard pytorch AMP training setup
        # scaler also works without amp training.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

        end_event_batch.record()

        torch.cuda.synchronize()  # Ensures all CUDA ops finish before measuring time
        batch_times.append(start_event_batch.elapsed_time(end_event_batch))  
        
    end_time = time.perf_counter() 

    peak_memory = torch.cuda.max_memory_allocated()  
    peak_reserved = torch.cuda.max_memory_reserved()
    print(f"Peak Allocated Memory : {peak_memory/1e6}MB | Peak Reserved Memory : {peak_reserved/1e6}MB")

    epoch_time = end_time - start_time
    print(f"Mean Batch Time : {np.mean(batch_times):.4f}ms | Std dev in Batch Time : {np.std(batch_times):.4f}ms")
    print(f"Epoch time : {epoch_time:.4f} seconds")

    print(f"Training Accuracy:{100.*correct/total: 0.2f}")
    print(f"Training Loss:{train_loss/(batch_idx+1): 0.3f}")
    print(f"Current LR : {scheduler.get_last_lr()[0]}")

    logs[-1]["train_loss"] = train_loss/(batch_idx+1)
    logs[-1]["train_acc"] = 100.*correct/total
    logs[-1]["train_peak_allocated_mem"] = peak_memory
    logs[-1]["train_peak_reserved_mem"] = peak_reserved
    logs[-1]["train_mean_batch_time"] = np.mean(batch_times)
    logs[-1]["lr"] = scheduler.get_last_lr()[0]

    return 100.0 * correct / total, train_loss / (batch_idx + 1)


def test(epoch, logs):
    print("\nTesting Epoch: %d" % epoch)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    batch_times = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            start_event_batch = torch.cuda.Event(enable_timing=True)
            end_event_batch = torch.cuda.Event(enable_timing=True)
            start_event_batch.record()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, F.one_hot(targets, num_classes=args.num_classes).float())

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            end_event_batch.record()

            torch.cuda.synchronize()  # Ensures all CUDA ops finish before measuring time
            batch_times.append(start_event_batch.elapsed_time(end_event_batch))  

        end_time = time.perf_counter() 

        print(f"Batch Shape : {next(iter(testloader))[0].shape}")

        peak_memory = torch.cuda.max_memory_allocated()  
        peak_reserved = torch.cuda.max_memory_reserved()
        print(f"Peak Allocated Memory : {peak_memory/1e6}MB | Peak Reserved Memory : {peak_reserved/1e6}MB")

        epoch_time = end_time - start_time
        print(f"Mean Batch Time : {np.mean(batch_times):.4f}ms | Std dev in Batch Time : {np.std(batch_times):.4f}ms")
        print(f"Epoch time : {epoch_time:.4f} seconds")
        
        print(f"Test Accuracy:{100.*correct/total: 0.2f}")
        print(f"Test Loss:{test_loss/(batch_idx+1): 0.3f}")

        logs[-1]["test_loss"] = test_loss/(batch_idx+1)
        logs[-1]["test_acc"] = 100.*correct/total
        logs[-1]["test_peak_allocated_mem"] = peak_memory
        logs[-1]["test_peak_reserved_mem"] = peak_reserved
        logs[-1]["test_mean_batch_time"] = np.mean(batch_times)

        return 100.0 * correct / total, test_loss / (batch_idx + 1)


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
else:
    print("Experiment Log with identical expt_name found, run terminated !")
    quit()

with open(f"expt_logs/{args.expt_name}/metadata.json", "w") as f:
    args_dict = vars(args)
    args_dict["model_params"] = sum(p.numel() for p in model.parameters())
    json.dump(args_dict, f, indent=4)

logs = []
for epoch in range(args.epochs):
    logs.append({"epoch" : epoch})

    train_acc, train_loss = train(epoch, logs)
    test_acc, test_loss = test(epoch, logs)

    # Add logging/plot code if needed
    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(f"./expt_logs/{args.expt_name}/{args.expt_name}_logs.csv")

    if epoch % 10 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, f"./expt_logs/{args.expt_name}/ckp_epoch_{epoch}.pth")
        print(f"Checkpoint saved: ./expt_logs/{args.expt_name}/ckp_epoch_{epoch}.pth")

        if epoch != 0:
            os.system(f"rm -rf ./expt_logs/{args.expt_name}/ckp_epoch_{epoch-10}.pth")
            print(f"Checkpoint deleted: ./expt_logs/{args.expt_name}/ckp_epoch_{epoch-10}.pth")
        
        
    scheduler.step()

# based on https://github.com/kentaroy47/vision-transformers-cifar10
