import torch
import torch.nn as nn
from models.resnet_encoder import ResnetEncoder
from models.depth_decoder import DepthDecoder
from models.pose_cnn import PoseCNN
from models.pose_decoder import PoseDecoder
from models.resnet_encoder import ResnetEncoder

# ------------------------------------------------------------
# EndoSfMLearner baseline (paper configuration, compatible with your repo)
# Depth: ResNet50   |   Pose: PoseCNN (equivalente a ResNet18 simplificado)
# Optional A,B affine correction (paper sec. 3.2)
# ------------------------------------------------------------

class EndoSfMLearner(nn.Module):
    def __init__(self, depth_resnet_layers=50, pose_resnet_layers=18,
                 pretrained=True, num_scales=1, use_brightness_affine=False):
        super(EndoSfMLearner, self).__init__()

        self.num_scales = num_scales
        self.use_brightness_affine = use_brightness_affine

        # ----- Depth branch -----
        self.depth_encoder = ResnetEncoder(depth_resnet_layers, pretrained)
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc)
        self.pose_encoder = ResnetEncoder(num_layers=resnet_pose, pretrained=with_pretrain)


        # ----- Pose branch -----
        # PoseCNN toma concatenación de imágenes (target + ref)
        self.pose_cnn = PoseCNN(num_input_frames=2)
        self.pose_decoder = PoseDecoder(self.pose_encoder.num_ch_enc, num_input_features=6)


        # ----- Optional affine correction -----
        if self.use_brightness_affine:
            self.affine_layer = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 6, 1)
            )
        else:
            self.affine_layer = None

    def forward(self, tgt_img, ref_imgs):
        outputs = {}

        # ---- Depth ----
        features = self.depth_encoder(tgt_img)
        disp = self.depth_decoder(features)
        outputs["depth"] = [disp[("disp", i)] for i in range(self.num_scales)]

        # ---- Pose ----
        poses, poses_inv = self.predict_poses(tgt_img, ref_imgs)
        outputs["poses"], outputs["poses_inv"] = poses, poses_inv

        # ---- Optional illumination correction ----
        if self.use_brightness_affine:
            ab = self.affine_layer(tgt_img)
            a, b = torch.chunk(ab, 2, dim=1)
            outputs["affine"] = (a, b)

        return outputs

    def predict_poses(self, tgt_img, ref_imgs):
        poses, poses_inv = [], []
        for ref_img in ref_imgs:
            # Concatenate target and reference frame
            input_pair = torch.cat([tgt_img, ref_img], dim=1)
            pose_feat = self.pose_cnn(input_pair)
            pose = self.pose_decoder(pose_feat)
            poses.append(pose)
            poses_inv.append(-pose)
        return poses, poses_inv
