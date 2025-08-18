# models/resnet_encoder2.py
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

"""
Compatibilidad con torchvision >= 0.13 para el encoder de DispNet (single-image).
Usa enums de pesos en lugar de model_urls/pretrained=.
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

class ResnetEncoder(nn.Module):
    """Encoder ResNet para DispNet (entrada de 1 imagen)."""
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        if num_layers not in [18, 34, 50]:
            raise ValueError(f"{num_layers} is not a valid number of resnet layers")

        weights = _get_weights_enum(num_layers, pretrained)
        self.encoder = _build_resnet_backbone(num_layers, weights)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        features = []
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        features.append(x)

        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x); features.append(x)
        x = self.encoder.layer2(x); features.append(x)
        x = self.encoder.layer3(x); features.append(x)
        x = self.encoder.layer4(x); features.append(x)
        return features
