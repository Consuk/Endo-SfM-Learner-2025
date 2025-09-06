import argparse
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import os
from tqdm import tqdm
from path import Path

################### Options ######################
parser = argparse.ArgumentParser(description="NYUv2 Depth options")
parser.add_argument("--dataset", required=True, help="kitti or nyu", choices=['nyu', 'kitti'], type=str)
parser.add_argument("--pred_depth", required=True, help="depth predictions npy", type=str)
parser.add_argument("--gt_depth", required=True, help="gt depth nyu for nyu or folder for kitti", type=str)
parser.add_argument("--vis_dir", help="result directory for saving visualization", type=str)
parser.add_argument("--img_dir", help="image directory for reading image", type=str)
parser.add_argument("--ratio_name", help="names for saving ratios", type=str)

######################################################
args = parser.parse_args()

def load_npz_depths(path):
    """Carga un .npz y devuelve el array correcto (data/depths/arr_0)."""
    f = np.load(path, allow_pickle=True)
    for k in ["data", "depths", "arr_0"]:
        if k in f:
            arr = f[k]
            break
    else:
        # si no hay claves conocidas, toma el primer arreglo
        first = list(f.keys())[0]
        arr = f[first]
    return arr



def mkdir_if_not_exists(path):
    """Make a directory if it does not exist.
    Args:
        path: directory to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    Args:
        gt (N): ground truth depth
        pred (N): predicted depth
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    log10 = np.mean(np.abs((np.log10(gt) - np.log10(pred))))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    if args.dataset == 'nyu':
        return abs_rel, log10, rmse, a1, a2, a3
    elif args.dataset == 'kitti':
        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def depth_visualizer(data):
    """
    Args:
        data (HxW): depth data
    Returns:
        vis_data (HxWx3): depth visualization (RGB)
    """

    inv_depth = 1 / (data + 1e-6)
    vmax = np.percentile(inv_depth, 95)
    normalizer = mpl.colors.Normalize(vmin=inv_depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    vis_data = (mapper.to_rgba(inv_depth)[:, :, :3] * 255).astype(np.uint8)
    return vis_data


def depth_pair_visualizer(pred, gt):
    """
    Args:
        data (HxW): depth data
    Returns:
        vis_data (HxWx3): depth visualization (RGB)
    """

    inv_pred = 1 / (pred + 1e-6)
    inv_gt = 1 / (gt + 1e-6)

    vmax = np.percentile(inv_gt, 95)
    normalizer = mpl.colors.Normalize(vmin=inv_gt.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')

    vis_pred = (mapper.to_rgba(inv_pred)[:, :, :3] * 255).astype(np.uint8)
    vis_gt = (mapper.to_rgba(inv_gt)[:, :, :3] * 255).astype(np.uint8)

    return vis_pred, vis_gt


class DepthEvalEigen():
    def __init__(self):

        self.min_depth = 1e-3

        if args.dataset == 'nyu':
            self.max_depth = None
        elif args.dataset == 'kitti':
            self.max_depth = 80.

    def main(self):
        pred_depths = []

        """ Get result """
        # Read precomputed result
        pred_depths = np.load(os.path.join(args.pred_depth))

        """ Evaluation """
        # --- Carga de GT robusta ---
        if args.dataset == 'nyu':
            gt_depths = load_npz_depths(args.gt_depth)   # [N,H,W]
        elif args.dataset == 'kitti':
            gts = []
            for gt_f in sorted(Path(args.gt_depth).files("*.npy")):
                gts.append(np.load(gt_f))
            gt_depths = np.stack(gts, 0) if len(gts) else np.empty((0,))

        # --- Alineación por si difieren en N ---
        N_pred = pred_depths.shape[0]
        N_gt   = gt_depths.shape[0] if isinstance(gt_depths, np.ndarray) else len(gt_depths)
        N = min(N_pred, N_gt)
        if N_pred != N_gt:
            print(f"[warn] N pred ({N_pred}) != N gt ({N_gt}); evaluando los primeros {N}.")

        pred_depths = pred_depths[:N]
        gt_depths   = gt_depths[:N]

        pred_depths = self.evaluate_depth(gt_depths, pred_depths, eval_mono=True)


        """ Save result """
        # create folder for visualization result
        if args.vis_dir:
            save_folder = Path(args.vis_dir)/'vis_depth'
            mkdir_if_not_exists(save_folder)

            image_paths = sorted(Path(args.img_dir).files('*.png'))

            for i in tqdm(range(len(pred_depths))):
                # reading image
                img = cv2.imread(image_paths[i], 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                h, w, _ = img.shape

                cat_img = 0
                if args.dataset == 'nyu':
                    cat_img = np.zeros((h, 3*w, 3))
                    cat_img[:, :w] = img
                    pred = pred_depths[i]
                    gt = gt_depths[i]
                    vis_pred, vis_gt = depth_pair_visualizer(pred, gt)
                    cat_img[:, w:2*w] = vis_pred
                    cat_img[:, 2*w:3*w] = vis_gt
                elif args.dataset == 'kitti':
                    cat_img = np.zeros((2*h, w, 3))
                    cat_img[:h] = img
                    pred = pred_depths[i]
                    vis_pred = depth_visualizer(pred)
                    cat_img[h:2*h, :] = vis_pred

                # save image
                cat_img = cat_img.astype(np.uint8)
                png_path = os.path.join(save_folder, "{:04}.png".format(i))
                cv2.imwrite(png_path, cv2.cvtColor(cat_img, cv2.COLOR_RGB2BGR))

    def evaluate_depth(self, gt_depths, pred_depths, eval_mono=True):
        """evaluate depth result
        Args:
            gt_depths (NxHxW): gt depths
            pred_depths (NxHxW): predicted depths
            eval_mono (bool): use median scaling if True
        """
        errors = []
        ratios = []
        resized_pred_depths = []

        # Soporta tanto np.ndarray como listas
        N = pred_depths.shape[0] if isinstance(pred_depths, np.ndarray) else len(pred_depths)

        print("==> Evaluating depth result...")
        for i in tqdm(range(N)):
            if np.mean(pred_depths[i]) != -1:
                gt_depth = gt_depths[i]
                gt_height, gt_width = gt_depth.shape[:2]

                # Resizing basado en DISPARIDAD inversa
                pred_inv_depth = 1.0 / (pred_depths[i] + 1e-6)
                pred_inv_depth = cv2.resize(pred_inv_depth, (gt_width, gt_height))
                pred_depth = 1.0 / (pred_inv_depth + 1e-6)

                # Máscara de válidos: usa solo límite inferior; aplica max solo si no es None
                mask = gt_depth > self.min_depth
                if (self.max_depth is not None):
                    mask &= (gt_depth < self.max_depth)

                # Recorte KITTI (solo si aplica)
                if args.dataset == 'kitti':
                    gh, gw = gt_depth.shape
                    crop = np.array([0.40810811 * gh, 0.99189189 * gh,
                                    0.03594771 * gw, 0.96405229 * gw]).astype(np.int32)
                    crop_mask = np.zeros(mask.shape)
                    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                    mask = np.logical_and(mask, crop_mask)

                # Si no hay píxeles válidos, saltamos este sample para evitar NaNs
                if not np.any(mask):
                    continue

                val_pred_depth = pred_depth[mask]
                val_gt_depth = gt_depth[mask]


                # Median scaling (modo monocular)
                ratio = 1.0
                if eval_mono and val_pred_depth.size > 0:
                    ratio = np.median(val_gt_depth) / (np.median(val_pred_depth) + 1e-6)
                    ratios.append(ratio)
                    val_pred_depth = val_pred_depth * ratio

                # Guarda la pred redimensionada (antes de clamping) con el ratio aplicado
                resized_pred_depths.append(pred_depth * ratio)

                # Clamps de seguridad
                val_pred_depth = np.clip(val_pred_depth, self.min_depth, self.max_depth)

                # Métricas
                errors.append(compute_depth_errors(val_gt_depth, val_pred_depth))

        # Stats de los ratios (si hubo)
        if eval_mono and len(ratios) > 0:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(
                med, np.std(ratios / (med + 1e-6))
            ))
            print(" Scaling ratios | mean: {:0.3f} +- std: {:0.3f}".format(
                np.mean(ratios), np.std(ratios)
            ))
            if args.ratio_name:
                np.savetxt(args.ratio_name, ratios, fmt='%.4f')

        mean_errors = np.array(errors).mean(0)

        if args.dataset == 'nyu':
            print("\n  " + ("{:>8} | " * 6).format("abs_rel", "log10", "rmse", "a1", "a2", "a3"))
            print(("&{: 8.3f}  " * 6).format(*mean_errors.tolist()) + "\\\\")
        elif args.dataset == 'kitti':
            print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
            print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

        return np.array(resized_pred_depths, dtype=np.float32)



eval = DepthEvalEigen()
eval.main()
