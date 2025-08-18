# models/resnet_encoder.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
import numpy as np

"""
Compatibilidad con torchvision >= 0.13:
- Usamos enums de pesos (ResNet18_Weights, etc.) en lugar de model_urls/pretrained=.
- Para entrada multi-imagen (p.ej., PoseNet con 2 imágenes), expandimos conv1 de 3->3*num_imgs
  y la inicializamos copiando pesos preentrenados (promedio por canal y réplica).
"""

def _get_weights_enum(num_layers, pretrained: bool):
    if not pretrained:
        return None
    if num_layers == 18:
        return ResNet18_Weights.IMAGENET1K_V1
    elif num_layers == 34:
        return ResNet34_Weights.IMAGENET1K_V1
    elif num_layers == 50:
        return ResNet50_Weights.IMAGENET1K_V1
    else:
        raise ValueError(f"Unsupported resnet layers: {num_layers}")

def _build_resnet_backbone(num_layers, weights):
    if num_layers == 18:
        return resnet18(weights=weights)
    elif num_layers == 34:
        return resnet34(weights=weights)
    elif num_layers == 50:
        return resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported resnet layers: {num_layers}")

def resnet_multiimage_input(num_layers=18, pretrained=True, num_input_images=2):
    """
    Crea un ResNet preentrenado y adapta conv1 para num_input_images>1.
    """
    weights = _get_weights_enum(num_layers, pretrained)
    resnet = _build_resnet_backbone(num_layers, weights)

    if num_input_images > 1:
        old_conv = resnet.conv1                      # (out_c, 3, k, k)
        in_channels = 3 * num_input_images
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        with torch.no_grad():
            # promedio en canales y réplica equilibrada
            mean_w = old_conv.weight.mean(dim=1, keepdim=True)   # (out_c,1,k,k)
            new_w = mean_w.repeat(1, in_channels, 1, 1) / float(num_input_images)
            new_conv.weight.copy_(new_w)
        resnet.conv1 = new_conv

    return resnet

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=1):
        super(SpatialAttention, self).__init__()
        kernel_size = 1
        padding = 3 if kernel_size == 7 else 0

        self.conv1 = nn.Conv2d(64, 4, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(4, 4, kernel_size , padding=padding, bias=False)
        self.conv3 = nn.Conv2d(4, 64, kernel_size, padding=padding, bias=False)
        self.maxPooling = nn.MaxPool2d(4,stride=4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.upsample = nn.Upsample(scale_factor=4)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.maxPooling(x1)
        reshaped1 = torch.reshape(x2,(x2.shape[0],x2.shape[1],-1,x2.shape[2]))
        y = torch.matmul(reshaped1,x2)
        z = self.relu(y)
        z = self.conv2(z)
        t = self.softmax(z)
        out1 = torch.matmul(t,reshaped1)
        conv3_out = self.conv3(out1)
        upsample_out = self.upsample(conv3_out)
        k = torch.reshape(upsample_out,(upsample_out.shape[0],upsample_out.shape[1],-1,upsample_out.shape[2]))
        output = k + x
        return output

class ResnetEncoder(nn.Module):
    """Encoder ResNet para Pose (multi-imagen)."""
    def __init__(self, num_layers, pretrained, num_input_images=2):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        if num_layers not in [18, 34, 50]:
            raise ValueError(f"{num_layers} is not a valid number of resnet layers")

        self.encoder = resnet_multiimage_input(
            num_layers=num_layers,
            pretrained=pretrained,
            num_input_images=num_input_images
        )

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.SAB = SpatialAttention()

    def forward(self, input_image):
        # input_image: (B, 3*num_imgs, H, W)
        features = []
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        features.append(x)

        # atención espacial en la primera escala
        features.append(self.SAB(features[-1]))

        x = self.encoder.maxpool(features[-1])
        x = self.encoder.layer1(x); features.append(x)
        x = self.encoder.layer2(x); features.append(x)
        x = self.encoder.layer3(x); features.append(x)
        x = self.encoder.layer4(x); features.append(x)
        return features
