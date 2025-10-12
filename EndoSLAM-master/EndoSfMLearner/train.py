import argparse
import time
import csv
import datetime as dt
from path import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import models
import custom_transforms
from utils import tensor2array, save_checkpoint
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
from models.endosfm_model import EndoSfMLearner

# --------------------- ARGUMENTOS ---------------------
parser = argparse.ArgumentParser(description='Train EndoSfMLearner on SCARED Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--split', type=str, default='splits/SCARED', help='path to split txt files')
parser.add_argument('--folder-type', type=str, choices=['sequence', 'pair'], default='sequence')
parser.add_argument('--sequence-length', type=int, default=3)
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--epoch-size', default=0, type=int)
parser.add_argument('-b', '--batch-size', default=4, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--beta', default=0.999, type=float)
parser.add_argument('--weight-decay', default=0, type=float)
parser.add_argument('--print-freq', default=10, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--log-summary', default='progress_log_summary.csv')
parser.add_argument('--log-full', default='progress_log_full.csv')
parser.add_argument('--log-output', action='store_true')
parser.add_argument('--resnet-layers', type=int, default=18, choices=[18, 50])
parser.add_argument('--num-scales', type=int, default=1)
parser.add_argument('-p', '--photo-loss-weight', type=float, default=1)
parser.add_argument('-s', '--smooth-loss-weight', type=float, default=0.1)
parser.add_argument('-c', '--geometry-consistency-weight', type=float, default=0.5)
parser.add_argument('--with-ssim', type=int, default=1)
parser.add_argument('--with-mask', type=int, default=1)
parser.add_argument('--with-auto-mask', type=int, default=0)
parser.add_argument('--with-pretrain', type=int, default=1)
parser.add_argument('--dataset', type=str, choices=['kitti', 'nyu', 'scared'], default='scared')
parser.add_argument('--pretrained', dest='pretrained', default=None)
parser.add_argument('--name', dest='name', type=str, required=True)
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros')
parser.add_argument('--with-gt', action='store_true')
parser.add_argument("--use_brightness_affine", action="store_true",
                    help="Activa corrección local A,B (iluminación) del paper; por defecto OFF")

# W&B
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--wandb_project', type=str, default='endosfm-scarED')
parser.add_argument('--wandb_entity', type=str, default=None)
parser.add_argument('--wandb_run', type=str, default=None)
parser.add_argument('--wandb_offline', action='store_true')
parser.add_argument('--wandb_log_images_every', type=int, default=500)

wandb_run = None
best_error = -1
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

# --------------------- MAIN ---------------------
def main():
    global best_error, n_iter, wandb_run
    args = parser.parse_args()

    # --- W&B init ---
    if args.wandb:
        import os, wandb
        if args.wandb_offline:
            os.environ['WANDB_MODE'] = 'offline'
        run_name = args.wandb_run or f"{args.name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_kwargs = dict(project=args.wandb_project, name=run_name, config=vars(args))
        if args.wandb_entity:
            wandb_kwargs['entity'] = args.wandb_entity
        wandb_run = wandb.init(**wandb_kwargs)

    # --- Setup ---
    timestamp = dt.datetime.now().strftime("%m-%d-%H:%M")
    args.save_path = Path('checkpoints') / args.name / timestamp
    args.save_path.makedirs_p()
    print(f"=> Saving to {args.save_path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    training_writer = SummaryWriter(args.save_path)

    # --- Dataset ---
    normalize = custom_transforms.Normalize(mean=[0.45,0.45,0.45], std=[0.225,0.225,0.225])
    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize])
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    # 🔹 Selección de dataset
    if args.dataset.lower() == 'scared':
        print("=> Using SCARED dataset")
        from datasets.scared_dataset import SCAREDDataset
        train_filenames = open(Path(args.split) / "train_files.txt").read().splitlines()
        val_filenames = open(Path(args.split) / "val_files.txt").read().splitlines()

        train_set = SCAREDDataset(
            data_path=args.data,
            filenames=train_filenames,
            height=256, width=320,
            frame_idxs=[0, -1, 1],
            num_scales=args.num_scales,
            is_train=True)
        val_set = SCAREDDataset(
            data_path=args.data,
            filenames=val_filenames,
            height=256, width=320,
            frame_idxs=[0],
            num_scales=args.num_scales,
            is_train=False)
    else:
        print("=> Using generic sequence dataset")
        from datasets.sequence_folders import SequenceFolder
        train_set = SequenceFolder(args.data, transform=train_transform, seed=args.seed,
                                   train=True, sequence_length=args.sequence_length, dataset=args.dataset)
        val_set = SequenceFolder(args.data, transform=valid_transform, seed=args.seed,
                                 train=False, sequence_length=args.sequence_length, dataset=args.dataset)

    print(f"{len(train_set)} train samples, {len(val_set)} val samples")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    # --- Model ---
    print("=> Creating EndoSfMLearner model")
    model = EndoSfMLearner(args.resnet_layers, args.with_pretrain,
                           num_scales=args.num_scales,
                           use_brightness_affine=args.use_brightness_affine).to(device)
    model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    # --- Logs ---
    with open(args.save_path/args.log_summary, 'w') as csvfile:
        csv.writer(csvfile, delimiter='\t').writerow(['train_loss', 'val_loss'])

    logger = TermLogger(n_epochs=args.epochs, train_size=len(train_loader), valid_size=len(val_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)
        train_loss = train(args, train_loader, model, optimizer, logger, training_writer)
        logger.train_writer.write(f' * Avg Train Loss : {train_loss:.3f}')

        logger.reset_valid_bar()
        val_loss = validate(args, val_loader, model, epoch, logger)
        logger.valid_writer.write(f' * Avg Val Loss : {val_loss:.3f}')

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            csv.writer(csvfile, delimiter='\t').writerow([train_loss, val_loss])
    logger.epoch_bar.finish()


def train(args, train_loader, model, optimizer, logger, writer):
    global n_iter
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight
    model.train()

    for i, batch in enumerate(train_loader):
        if args.dataset == 'scared':
            inputs = batch
            tgt_img = inputs[("color", 0, 0)].to(device)
            ref_imgs = [inputs[("color", i, 0)].to(device) for i in [-1, 1] if ("color", i, 0) in inputs]
            intrinsics = inputs[("K", 0)].to(device)
        else:
            tgt_img, ref_imgs, intrinsics, _ = batch
            tgt_img, intrinsics = tgt_img.to(device), intrinsics.to(device)
            ref_imgs = [img.to(device) for img in ref_imgs]

        outputs = model(tgt_img, ref_imgs)
        tgt_depths, poses, poses_inv = outputs["depth"], outputs["poses"], outputs["poses_inv"]
        affine_maps = outputs.get("affine", None)

        loss_p, loss_g = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics,
                                                         tgt_depths, None, poses, poses_inv,
                                                         args.num_scales, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask,
                                                         args.padding_mode, affine_maps)
        loss_s = compute_smooth_loss(tgt_depths, tgt_img, None, ref_imgs)
        loss = w1*loss_p + w2*loss_s + w3*loss_g

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), args.batch_size)
        if i % args.print_freq == 0:
            logger.train_writer.write(f"Iter {i}: Loss {loss.item():.4f}")
        n_iter += 1
        if args.epoch_size > 0 and i >= args.epoch_size - 1:
            break
    return losses.avg[0]


@torch.no_grad()
def validate(args, val_loader, model, epoch, logger):
    losses = AverageMeter(precision=4)
    model.eval()
    for i, batch in enumerate(val_loader):
        if args.dataset == 'scared':
            inputs = batch
            tgt_img = inputs[("color", 0, 0)].to(device)
            ref_imgs = [inputs[("color", i, 0)].to(device) for i in [-1, 1] if ("color", i, 0) in inputs]
            intrinsics = inputs[("K", 0)].to(device)
        else:
            tgt_img, ref_imgs, intrinsics, _ = batch
            tgt_img, intrinsics = tgt_img.to(device), intrinsics.to(device)
            ref_imgs = [img.to(device) for img in ref_imgs]

        outputs = model(tgt_img, ref_imgs)
        tgt_depths, poses, poses_inv = outputs["depth"], outputs["poses"], outputs["poses_inv"]
        loss_p, loss_g = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics,
                                                         tgt_depths, None, poses, poses_inv,
                                                         args.num_scales, args.with_ssim,
                                                         args.with_mask, False,
                                                         args.padding_mode)
        loss_s = compute_smooth_loss(tgt_depths, tgt_img, None, ref_imgs)
        total_loss = loss_p + 0.1*loss_s + 0.5*loss_g
        losses.update(total_loss.item(), args.batch_size)
        if i % args.print_freq == 0:
            logger.valid_writer.write(f"Val iter {i}: Loss {total_loss.item():.4f}")
    return losses.avg[0]


if __name__ == "__main__":
    main()
