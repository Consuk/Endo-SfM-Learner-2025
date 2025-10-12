from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from inverse_warp import inverse_warp2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1, self.C2 = 0.01 ** 2, 0.03 ** 2

    def forward(self, x, y):
        x, y = self.refl(x), self.refl(y)
        mu_x, mu_y = self.mu_x_pool(x), self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        ssim = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        denom = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - ssim / denom) / 2, 0, 1)


compute_ssim_loss = SSIM().to(device)


def apply_affine_if_given(img, affine_maps):
    """Aplica corrección de brillo/contraste local A*I + B si se provee"""
    if affine_maps is None:
        return img
    A, B = affine_maps.get("A", None), affine_maps.get("B", None)
    if A is None or B is None:
        return img
    return A * img + B


def compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depths, ref_depths,
                                    poses, poses_inv, max_scales, with_ssim, with_mask,
                                    with_auto_mask, padding_mode, affine_maps=None):
    photo_loss, geometry_loss = 0, 0
    num_scales = min(len(tgt_depths), max_scales)

    for ref_img, pose, pose_inv in zip(ref_imgs, poses, poses_inv):
        for s in range(num_scales):
            tgt_depth = tgt_depths[s]
            ref_img_warped, valid_mask, proj_depth, comp_depth = inverse_warp2(
                ref_img, tgt_depth, tgt_depth, pose, intrinsics, padding_mode)

            # Aplica corrección A,B si hay
            ref_img_warped = apply_affine_if_given(ref_img_warped, affine_maps)

            diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)
            diff_depth = ((comp_depth - proj_depth).abs() / (comp_depth + proj_depth)).clamp(0, 1)

            if with_auto_mask:
                auto_mask = (diff_img.mean(1, True) < (tgt_img - ref_img).abs().mean(1, True)).float() * valid_mask
                valid_mask = auto_mask

            if with_ssim:
                ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
                diff_img = 0.15 * diff_img + 0.85 * ssim_map

            if with_mask:
                diff_img *= (1 - diff_depth)

            photo_loss += mean_on_mask(diff_img, valid_mask)
            geometry_loss += mean_on_mask(diff_depth, valid_mask)

    return photo_loss, geometry_loss


def mean_on_mask(diff, mask):
    mask = mask.expand_as(diff)
    if mask.sum() > 10000:
        return (diff * mask).sum() / mask.sum()
    return torch.tensor(0).float().to(device)


def compute_smooth_loss(tgt_depths, tgt_img, ref_depths, ref_imgs):
    def get_smooth_loss(disp, img):
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        grad_disp_x = (norm_disp[:, :, :, :-1] - norm_disp[:, :, :, 1:]).abs()
        grad_disp_y = (norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :]).abs()
        grad_img_x = (img[:, :, :, :-1] - img[:, :, :, 1:]).abs().mean(1, True)
        grad_img_y = (img[:, :, :-1, :] - img[:, :, 1:, :]).abs().mean(1, True)
        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)
        return grad_disp_x.mean() + grad_disp_y.mean()
    return get_smooth_loss(tgt_depths[0], tgt_img)
