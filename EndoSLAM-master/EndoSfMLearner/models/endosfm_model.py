# models/endosfm_model.py
import torch
import torch.nn as nn
from .DispResNet import DispResNet
from .PoseResNet import PoseResNet

class EndoSfMLearner(nn.Module):
    def __init__(self, num_scales=1, pretrained=True, use_brightness_affine=False):
        super(EndoSfMLearner, self).__init__()

        # Depth network uses ResNet-50
        self.depth_net = ResnetEncoder(50, pretrained)
        self.depth_decoder = DepthDecoder(self.depth_net.num_ch_enc, num_output_channels=1, scales=range(num_scales))

        # Pose network uses ResNet-18
        self.pose_net = PoseResnet(18, pretrained)
        self.pose_decoder = PoseDecoder(self.pose_net.num_ch_enc, num_input_frames=2)

        self.use_brightness_affine = use_brightness_affine


        if self.use_brightness_affine:
            # Pequeño "decoder" ligero para A (contraste) y B (brillo) a partir de features de pose.
            # Mantengo algo simple y barato; si luego quieres, lo movemos a un head más grande.
            self.affine_head = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 2, 1)  # [A,B] (2 canales)
            )

    @torch.no_grad()
    def _disp_to_depth(self, disp_pyramid):
        # Tu DispResNet ya devuelve pirámide de disparidades; profundidad = 1/disp
        depth_pyramid = []
        for d in disp_pyramid:
            depth_pyramid.append(1.0 / (d.clamp(min=1e-6)))
        return depth_pyramid

    def forward(self, tgt_img, ref_imgs):
        """
        Params
        - tgt_img: (B,3,H,W)
        - ref_imgs: list de (B,3,H,W), típicamente [I_{t-1}, I_{t+1}]
        Returns dict:
            {
              "disp": [s0..sS], "depth": [s0..sS],
              "poses":   [(axisangle, trans) por ref],
              "poses_inv":[(axisangle, trans) inversas],
              (opc) "affine": {"A": A, "B": B}  # si use_brightness_affine
            }
        """
        outputs = {}

        # Depth (pirámide)
        disp_pyr = self.depth_net(tgt_img)
        outputs["disp"]  = disp_pyr
        outputs["depth"] = self._disp_to_depth(disp_pyr)

        # Poses (t->ref) y (ref->t)
        poses, poses_inv = [], []
        for ref in ref_imgs:
            poses.append(self.pose_net(tgt_img, ref))
            poses_inv.append(self.pose_net(ref, tgt_img))
        outputs["poses"] = poses
        outputs["poses_inv"] = poses_inv

        # Estimar A,B (opcional): un mapa por escala 0 (resolución de entrada)
        if self.use_brightness_affine:
            # Tomamos features intermedios de PoseResNet (si tu PoseResNet expone features, mejor).
            # Si no los expone, usamos un proxy: concatenamos tgt y primer ref y extraemos features rápidos.
            # Para mantener compatibilidad sin tocar PoseResNet, hago un block ligero aquí:
            x = torch.cat([tgt_img] + ref_imgs, dim=1)  # (B, 3*(1+N), H, W)
            # Proyección rápida a 256 canales
            feat = nn.functional.relu(nn.Conv2d(x.size(1), 256, 3, padding=1, bias=False).to(x.device)(x))
            ab  = self.affine_head(feat)  # (B,2,H,W)
            A, B = torch.split(ab, 1, dim=1)
            outputs["affine"] = {"A": A, "B": B}

        return outputs
