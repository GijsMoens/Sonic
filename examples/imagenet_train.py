"""ImageNet training script for SONIC-ResNet.

Usage::

    python examples/imagenet_train.py /path/to/imagenet --epochs 120

    # shallow default (4 blocks, 128 modes, ~7.4M params)
    python examples/imagenet_train.py /path/to/imagenet

    # ResNet-50 depth (16 blocks, 32 modes, ~14.7M params)
    python examples/imagenet_train.py /path/to/imagenet --layers 3 4 6 3 --modes 32

Supports single-GPU, DataParallel, and DistributedDataParallel training.
"""

import argparse
import os
import random
import shutil
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from resnet50_sonic import resnet_sonic

try:
    import wandb
except ImportError:
    wandb = None


best_acc1 = 0


def main():
    parser = argparse.ArgumentParser(description="SONIC ImageNet Training")
    parser.add_argument("data", metavar="DIR", help="path to ImageNet dataset")
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("-b", "--batch-size", default=256, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--wd", default=0.05, type=float, dest="weight_decay")
    parser.add_argument("-j", "--workers", default=12, type=int)
    parser.add_argument("-p", "--print-freq", default=1000, type=int)
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
    parser.add_argument("--warmup-epochs", default=5, type=int, help="LR warmup epochs")
    parser.add_argument("--wandb", action="store_true", help="enable wandb logging")
    parser.add_argument("--wandb-project", default="sonic-imagenet", type=str)
    parser.add_argument("--wandb-run-name", default=None, type=str)
    parser.add_argument("--layers", nargs=4, type=int, default=[1, 1, 1, 1],
                        metavar=("L1", "L2", "L3", "L4"),
                        help="blocks per stage (default: 1 1 1 1)")
    parser.add_argument("--modes", type=int, default=128,
                        help="Sonic M_modes per block (default: 128)")
    args = parser.parse_args()

    if args.wandb and wandb is None:
        raise ImportError("wandb is required when --wandb is set. Install with: pip install wandb")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "Seeded training enabled. This turns on CUDNN deterministic mode, "
            "which can slow down training considerably."
        )

    if args.gpu is not None:
        warnings.warn("Specific GPU selected; data parallelism is disabled.")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

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
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    model = resnet_sonic(layers=args.layers, M_modes=args.modes)

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {total_params:,} total ({total_params/1e6:.2f}M), "
              f"{trainable:,} trainable ({trainable/1e6:.2f}M)")

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    if args.resume and os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        loc = f"cuda:{args.gpu}" if args.gpu is not None else None
        checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint["best_acc1"]
        if args.gpu is not None:
            best_acc1 = best_acc1.to(args.gpu)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.start_epoch,
        last_epoch=args.start_epoch - 1,
    )
    if args.warmup_epochs > 0 and args.start_epoch == 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=args.warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs],
        )
    else:
        scheduler = cosine
    cudnn.benchmark = True

    # Data loading
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
    )

    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset)
        if args.distributed else None
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
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
    )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc1, train_acc5 = train(train_loader, model, criterion, optimizer, epoch, args)
        acc1, acc5 = validate(val_loader, model, criterion, args)
        scheduler.step()

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

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
            })

        if is_main:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=is_best,
                ckt_dir=args.ckt_path,
            )


def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    loss_sum, top1_sum, top5_sum, total = 0.0, 0.0, 0.0, 0
    for i, (images, target) in enumerate(train_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        loss_sum += loss.item() * bs
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1_sum += acc1[0].item() * bs
        top5_sum += acc5[0].item() * bs
        total += bs

        if i % args.print_freq == 0 and (not args.distributed or args.rank == 0):
            print(f"Epoch [{epoch}][{i}/{len(train_loader)}]  "
                  f"Loss: {loss.item():.4e}  Acc@1: {acc1[0]:.2f}  Acc@5: {acc5[0]:.2f}")

    return loss_sum / total, top1_sum / total, top5_sum / total


def validate(val_loader, model, criterion, args):
    model.eval()
    top1_sum, top5_sum, total = 0.0, 0.0, 0
    with torch.no_grad():
        for images, target in val_loader:
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
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
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
