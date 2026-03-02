"""
ImageNet training script for SONIC ResNet-50.

Standard modern recipe: AdamW + cosine annealing + linear warmup + AMP.

Usage:
    # single GPU
    python examples/imagenet_train.py /path/to/imagenet --epochs 300

    # multi-GPU (torchrun)
    torchrun --nproc_per_node=4 examples/imagenet_train.py /path/to/imagenet
"""

import argparse
import csv
import os
import random
import shutil
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from resnet50_sonic import sonic_net

try:
    import wandb
except ImportError:
    wandb = None

best_acc1 = 0

def train(train_loader, model, criterion, optimizer, epoch, args, scaler):
    model.train()
    loss_sum, top1_sum, top5_sum, total = 0.0, 0.0, 0.0, 0

    for i, (images, target) in enumerate(train_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=args.amp):
            output = model(images)
            loss = criterion(output, target)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        loss_sum += loss.item() * bs
        acc1, acc5 = accuracy(output.float(), target, topk=(1, 5))
        top1_sum += acc1[0].item() * bs
        top5_sum += acc5[0].item() * bs
        total += bs

        if i % args.print_freq == 0 and (not args.distributed or args.rank == 0):
            print(f"Epoch [{epoch}][{i}/{len(train_loader)}]  "
                  f"Loss: {loss.item():.4e}  Acc@1: {acc1[0]:.2f}  "
                  f"Acc@5: {acc5[0]:.2f}")

    return loss_sum / total, top1_sum / total, top5_sum / total


def validate(val_loader, model, args):
    model.eval()
    top1_sum, top5_sum, total = 0.0, 0.0, 0
    with torch.no_grad():
        for images, target in val_loader:
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                output = model(images)
            acc1, acc5 = accuracy(output.float(), target, topk=(1, 5))
            top1_sum += acc1[0] * images.size(0)
            top5_sum += acc5[0] * images.size(0)
            total += images.size(0)

    top1_avg = top1_sum / total
    top5_avg = top5_sum / total
    if not args.distributed or args.rank == 0:
        print(f" * Acc@1 {top1_avg:.3f}  Acc@5 {top5_avg:.3f}")
    return top1_avg, top5_avg

def save_checkpoint(state, is_best, ckt_dir="checkpoints"):
    os.makedirs(ckt_dir, exist_ok=True)
    filepath = os.path.join(ckt_dir, "checkpoint.pth.tar")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(ckt_dir, "model_best.pth.tar"))


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(
            100.0 / batch_size) for k in topk]

def main():
    parser = argparse.ArgumentParser(description="SONIC ImageNet Training")
    parser.add_argument("data", metavar="DIR", help="path to ImageNet dataset")
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("-b", "--batch-size", default=1024, type=int)
    parser.add_argument("--lr", default=4e-3, type=float)
    parser.add_argument("--wd", default=0.05, type=float, dest="weight_decay")
    parser.add_argument("-j", "--workers", default=12, type=int)
    parser.add_argument("-p", "--print-freq", default=200, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("-e", "--evaluate", action="store_true")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument("--ckt-path", default="checkpoints", type=str)
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--rank", default=-1, type=int)
    parser.add_argument("--dist-url", default="tcp://224.66.41.62:23456", type=str)
    parser.add_argument("--dist-backend", default="nccl", type=str)
    parser.add_argument("--multiprocessing-distributed", action="store_true")
    parser.add_argument("--warmup-epochs", default=20, type=int)
    parser.add_argument("--grad-clip", default=1.0, type=float,
                        help="max gradient norm (0 to disable, default: 1.0)")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="sonic-imagenet", type=str)
    parser.add_argument("--wandb-run-name", default=None, type=str)
    # Architecture
    parser.add_argument("--size", default="normal", choices=["tiny", "normal", "large"],
                        help="model size: tiny (~5.7M), normal (~15.0M), large (~31.7M)")
    parser.add_argument("--modes", type=int, default=128,
                        help="Sonic M_modes per block (default: 128)")
    parser.add_argument("--drop-path-rate", default=0.2, type=float,
                        help="stochastic depth rate (default: 0.2)")
    args = parser.parse_args()

    if args.wandb and wandb is None:
        raise ImportError("wandb required: pip install wandb")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("Seeded training: CUDNN deterministic mode enabled (slower).")

    if args.gpu is not None:
        warnings.warn("Specific GPU selected; data parallelism is disabled.")

    # Auto-detect torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.dist_url = "env://"
        args.distributed = True
        args.multiprocessing_distributed = False
    else:
        if args.dist_url == "env://" and args.world_size == -1:
            args.world_size = int(os.environ.get("WORLD_SIZE", 1))
        args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    is_main = (not args.distributed) or args.rank == 0

    if args.wandb and is_main:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=vars(args))

    model = sonic_net(size=args.size,
                      M_modes=args.modes,
                      drop_path_rate=args.drop_path_rate)

    if is_main:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {total:,} total ({total/1e6:.2f}M), "
              f"{trainable:,} trainable ({trainable/1e6:.2f}M)")

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  fused=torch.cuda.is_available())

    if args.resume and os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        loc = f"cuda:{args.gpu}" if args.gpu is not None else None
        checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint["best_acc1"]
        if args.gpu is not None:
            best_acc1 = best_acc1.to(args.gpu)
        # Strip 'module.' prefix from DDP checkpoints when loading on single GPU
        state_dict = checkpoint["state_dict"]
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError:
            warnings.warn("Optimizer param-group mismatch — "
                          "loading model weights only (optimizer state reset).")
        print(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")

    # ---- LR schedule: linear warmup → cosine annealing ----
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs)
    if args.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=args.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[args.warmup_epochs])
    else:
        scheduler = cosine
    scaler = torch.amp.GradScaler(enabled=args.amp)
    cudnn.benchmark = True

    # ---- Data loading ----
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    train_sampler = (torch.utils.data.distributed.DistributedSampler(train_dataset)
                     if args.distributed else None)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        drop_last=True, persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )

    if args.evaluate:
        validate(val_loader, model, args)
        return

    run_dir_created = False

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc1, train_acc5 = train(
            train_loader, model, criterion, optimizer, epoch, args, scaler)
        acc1, acc5 = validate(val_loader, model, args)
        scheduler.step()

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # After first successful epoch, create a unique run checkpoint dir
        if not run_dir_created and best_acc1 != 0 and is_main:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_ckt_dir = os.path.join(args.ckt_path, run_id)
            os.makedirs(run_ckt_dir, exist_ok=True)
            args.ckt_path = run_ckt_dir

            # Append entry to local checkpoints index file
            index_file = os.path.join(os.path.dirname(run_ckt_dir), "checkpoints.csv")
            write_header = not os.path.exists(index_file)
            with open(index_file, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["run_id", "timestamp", "size", "modes",
                                     "lr", "batch_size", "epochs", "path"])
                writer.writerow([run_id, datetime.now().isoformat(), args.size,
                                 args.modes, args.lr, args.batch_size,
                                 args.epochs, run_ckt_dir])
            print(f"=> Created checkpoint dir: {run_ckt_dir}")
            run_dir_created = True

        if args.wandb and is_main:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc1": train_acc1,
                "train/acc5": train_acc5,
                "val/acc1": acc1,
                "val/acc5": acc5,
                "val/best_acc1": best_acc1,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

        if is_main:
            save_checkpoint(
                {"epoch": epoch + 1, "state_dict": model.state_dict(),
                 "best_acc1": best_acc1, "optimizer": optimizer.state_dict()},
                is_best=is_best, ckt_dir=args.ckt_path)

if __name__ == "__main__":
    main()
