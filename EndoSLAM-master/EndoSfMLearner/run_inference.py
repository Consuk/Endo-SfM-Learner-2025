import torch

import os
import imageio.v2 as imageio


# --- compat: reemplazo de scipy.misc.imresize ---
try:
    from scipy.misc import imresize  # por si acaso existe
except Exception:
    import cv2
    import numpy as np
    def imresize(arr, size, interp='bilinear'):
        # size suele ser (H, W); cv2.resize espera (W, H)
        h, w = size
        inter = cv2.INTER_LINEAR if interp in ('bilinear', 'linear') else cv2.INTER_NEAREST
        return cv2.resize(arr, (w, h), interpolation=inter)
# --- fin compat ---

import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from models import DispResNet
from utils import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispResNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument('--resnet-layers', required=True, type=int, default=18, choices=[18, 50],
                    help='depth network architecture.')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return

    disp_net = DispResNet(args.resnet_layers, False).to(device)
    # weights = torch.load(args.pretrained)
    # disp_net.load_state_dict(weights['state_dict'])
    weights = torch.load(args.pretrained, map_location=device)
    state = weights.get('state_dict', weights)  # por si viene plano o con otra clave
    disp_net.load_state_dict(state, strict=False)
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    # Extensiones a considerar (minúsculas y mayúsculas)
    exts = list(set(args.img_exts + [e.upper() for e in args.img_exts] + ['jpeg','JPEG']))

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        test_files = []
        for l in lines:
            p = Path(l)
            # si es relativa, únela a dataset_dir; si es absoluta, úsala tal cual
            p = (dataset_dir / p) if not p.isabs() else p
            if p.isdir():
                # recorrer recursivo
                for ext in exts:
                    test_files += list(p.walkfiles(f'*.{ext}'))
            else:
                test_files.append(p)
    else:
        # sin lista: buscar recursivo en dataset_dir
        test_files = []
        for ext in exts:
            test_files += list(dataset_dir.walkfiles(f'*.{ext}'))

    # ordenar y normalizar a Path
    test_files = sorted(map(Path, map(str, test_files)))
    print('{} files to test'.format(len(test_files)))


    for file in tqdm(test_files):

        # img = imread(file).astype(np.float32)
        # img = imageio.imread(file).astype(np.float32)
        img = imageio.imread(str(file)).astype(np.float32)
        # Si viene en HxW, conviértelo a HxWx3
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)


        h, w, _ = img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))

        tensor_img = torch.from_numpy(img).unsqueeze(0)
        tensor_img = ((tensor_img/255 - 0.45)/0.225).to(device)

        output = disp_net(tensor_img)[0]

        file_path_rel, _ = Path(file).relpath(args.dataset_dir).splitext()
        file_name = '-'.join(file_path_rel.splitall())

        # Helper para tener RGB uint8 sin alpha
        def to_rgb8(arr):
            # arr: HxWx{1,3,4} o HxW
            if arr.ndim == 2:
                arr = arr[:, :, None]
            if arr.shape[2] == 4:
                arr = arr[..., :3]
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.dtype != np.uint8:
                a_min, a_max = float(arr.min()), float(arr.max())
                rng = (a_max - a_min) if a_max > a_min else 1e-8
                arr = ((arr - a_min) / rng * 255.0).astype(np.uint8)
            return arr

        if args.output_disp:
            disp = (255 * tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)  # CxHxW
            disp_vis = np.transpose(disp, (1, 2, 0))  # HxWxC
            disp_vis = to_rgb8(disp_vis)
            out_path_disp = output_dir / f'{file_name}_disp.png'  # fuerza PNG
            imageio.imwrite(out_path_disp, disp_vis)

        if args.output_depth:
            depth = 1 / output  # torch tensor [1,1,H,W] o [1,H,W]

            # --- guarda PROFUNDIDAD NUMÉRICA (.npy) ---
            depth_np = depth.squeeze().detach().cpu().numpy()   # [H,W]
            np.save(output_dir / f"{file_name}_depth.npy", depth_np)

            # --- visualización coloreada (PNG) ---
            depth_vis = (255 * tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)  # CxHxW
            depth_vis = np.transpose(depth_vis, (1, 2, 0))  # HxWxC
            depth_vis = to_rgb8(depth_vis)
            out_path_depth = output_dir / f"{file_name}_depth.png"
            imageio.imwrite(out_path_depth, depth_vis)


            # (Opcional) guardar profundidad métrica cruda como .npy:
            # np.save(output_dir / f'{file_name}_depth.npy', depth.squeeze().cpu().numpy())



if __name__ == '__main__':
    main()
