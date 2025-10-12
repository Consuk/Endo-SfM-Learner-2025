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
import torchvision.utils as vutils
import matplotlib.cm as cm

import models
import custom_transforms
from utils import tensor2array, save_checkpoint
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
from models.endosfm_model import EndoSfMLearner

# --------------------- ARGUMENTOS ---------------------
parser = argparse.ArgumentParser(description='Train EndoSfMLearner (baseline configuration)',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--split', type=str, default='splits/SCARED', help='path to split txt files')
parser.add_argument('--sequence-length', type=int, default=3)
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch-size', default=8, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--beta', default=0.999, type=float)
parser.add_argument('--weight-decay', default=0, type=float)
parser.add_argument('--print-freq', default=10, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--resnet-depth', type=int, default=50)
parser.add_argument('--resnet-pose', type=int, default=18)
parser.add_argument('--num-scales', type=int, default=1)
parser.add_argument('--photo-loss-weight', type=float, default=1.0)
parser.add_argument('--smooth-loss-weight', type=float, default=0.1)
parser.add_argument('--geometry-consistency-weight', type=float, default=0.5)
parser.add_argument('--with-ssim', type=int, default=1)
parser.add_argument('--with-mask', type=int, default=1)
parser.add_argument('--with-auto-mask', type=int, default=0)
parser.add_argument('--with-pretrain', type=int, default=1)
parser.add_argument('--dataset', type=str, choices=['scared'], default='scared')
parser.add_argument('--name', dest='name', type=str, required=True)
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros')
parser.add_argument('--use_brightness_affine', action='store_true')
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--wandb_project', type=str, default='endosfm-scarED')
parser.add_argument('--wandb_entity', type=str, default=None)
parser.add_argument('--wandb_log_images_every', type=int, default=500)

wandb_run = None
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- UTILS ---------------------
def tensor_to_rgb(img_tensor):
    img_tensor = img_tensor.detach().cpu().clamp(0, 1)
    grid = vutils.make_grid(img_tensor, nrow=4)
    np_img = (grid.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    return np_img

def tensor_to_colormap(disp_tensor, cmap="plasma"):
    disp_tensor = disp_tensor.detach().cpu().squeeze(1)
    disp_normalized = (disp_tensor - disp_tensor.min()) / (disp_tensor.max() - disp_tensor.min() + 1e-8)
    disp_np = disp_normalized.numpy()
    colored = [cm.get_cmap(cmap)(d)[:, :, :3] for d in disp_np]
    grid = np.concatenate(colored, axis=1)
    return (grid * 255).astype(np.uint8)

# --------------------- MAIN ---------------------
def main():
    global n_iter
    args = parser.parse_args()

    # --- W&B init ---
    if args.wandb:
        import os, wandb
        run_name = f"{args.name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    print("=> Using SCARED dataset")
    from datasets.scared_dataset import SCAREDDataset
    train_filenames = open(Path(args.split) / "train_files.txt").read().splitlines()
    val_filenames = open(Path(args.split) / "val_files.txt").read().splitlines()

    train_set = SCAREDDataset(args.data, train_filenames, 256, 320, [0, -1, 1], args.num_scales, is_train=True)
    val_set = SCAREDDataset(args.data, val_filenames, 256, 320, [0], args.num_scales, is_train=False)

    print(f"{len(train_set)} train samples, {len(val_set)} val samples")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    # --- Model ---
    print("=> Creating EndoSfMLearner model (ResNet50 + ResNet18)")
    model = EndoSfMLearner(args.resnet_depth, args.resnet_pose, args.with_pretrain,
                           num_scales=args.num_scales,
                           use_brightness_affine=args.use_brightness_affine).to(device)
    model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    # --- Training ---
    logger = TermLogger(n_epochs=args.epochs, train_size=len(train_loader), valid_size=len(val_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)
        train_loss = train(args, train_loader, model, optimizer, logger, epoch)
        val_loss = validate(args, val_loader, model, epoch, logger)

        if args.wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch:03d}: train={train_loss:.4f}, val={val_loss:.4f}")

    logger.epoch_bar.finish()

# --------------------- TRAIN ---------------------
def train(args, loader, model, optimizer, logger, epoch):
    import wandb
    global n_iter
    losses = AverageMeter(precision=4)
    model.train()

    for i, batch in enumerate(loader):
        inputs = batch
        tgt_img = inputs[("color", 0, 0)].to(device)
        ref_imgs = [inputs[("color", idx, 0)].to(device) for idx in [-1, 1] if ("color", idx, 0) in inputs]
        intrinsics = inputs[("K", 0)].to(device)

        outputs = model(tgt_img, ref_imgs)
        tgt_depths, poses, poses_inv = outputs["depth"], outputs["poses"], outputs["poses_inv"]

        loss_p, loss_g = compute_photo_and_geometry_loss(
            tgt_img, ref_imgs, intrinsics, tgt_depths, None,
            poses, poses_inv, args.num_scales, args.with_ssim,
            args.with_mask, args.with_auto_mask, args.padding_mode)

        loss_s = compute_smooth_loss(tgt_depths, tgt_img, None, ref_imgs)
        total_loss = args.photo_loss_weight * loss_p + args.smooth_loss_weight * loss_s + args.geometry_consistency_weight * loss_g

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.update(total_loss.item(), args.batch_size)
        if args.wandb and i % args.wandb_log_images_every == 0:
            wandb.log({
                "Train/Input": wandb.Image(tensor_to_rgb(tgt_img)),
                "Train/Pred_Depth": wandb.Image(tensor_to_colormap(tgt_depths[0]))
            }, step=n_iter)

        n_iter += 1
    return losses.avg[0]

# --------------------- VALIDATE ---------------------
@torch.no_grad()
def validate(args, loader, model, epoch, logger):
    import wandb
    losses = AverageMeter(precision=4)
    model.eval()

    for i, batch in enumerate(loader):
        inputs = batch
        tgt_img = inputs[("color", 0, 0)].to(device)
        ref_imgs = [inputs[("color", idx, 0)].to(device) for idx in [-1, 1] if ("color", idx, 0) in inputs]
        intrinsics = inputs[("K", 0)].to(device)

        outputs = model(tgt_img, ref_imgs)
        tgt_depths, poses, poses_inv = outputs["depth"], outputs["poses"], outputs["poses_inv"]

        loss_p, loss_g = compute_photo_and_geometry_loss(
            tgt_img, ref_imgs, intrinsics, tgt_depths, None,
            poses, poses_inv, args.num_scales, args.with_ssim,
            args.with_mask, False, args.padding_mode)

        loss_s = compute_smooth_loss(tgt_depths, tgt_img, None, ref_imgs)
        total_loss = loss_p + 0.1 * loss_s + 0.5 * loss_g
        losses.update(total_loss.item(), args.batch_size)

        if args.wandb and i % args.wandb_log_images_every == 0:
            wandb.log({
                "Val/Input": wandb.Image(tensor_to_rgb(tgt_img)),
                "Val/Pred_Depth": wandb.Image(tensor_to_colormap(tgt_depths[0]))
            }, step=epoch)
    return losses.avg[0]

if __name__ == "__main__":
    main()
