
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(CNNClassifier, self).__init__()

        # Load Inception v3
        weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
        self.inception = models.inception_v3(weights=weights)

        # Inception v3 expects 299x299. For CIFAR-10 (32x32), we should upscale
        # or handle the dimensionality. We'll add an upsampler for best results.
        self.upsample = nn.Upsample(
            size=(299, 299), mode='bilinear', align_corners=True)

        # Replace the final fully connected layer
        # Inception v3's fc layer input features is 2048
        in_features = self.inception.fc.in_features

        self.inception.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

        # Handle the auxiliary output (required for Inception v3 during training)
        aux_in_features = self.inception.AuxLogits.fc.in_features
        self.inception.AuxLogits.fc = nn.Linear(aux_in_features, num_classes)

    def forward(self, x):
        # 1. Upscale 32x32 to 299x299 for Inception compatibility
        if x.shape[2:] != (299, 299):
            x = self.upsample(x)

        # 2. Inception v3 returns a named tuple (logits, aux_logits) during training
        if self.training:
            out, aux_out = self.inception(x)
            # Usually, you use (out + 0.3 * aux_out) in your loss function
            return out
        else:
            return self.inception(x)

    def get_config(self):
        return {
            "architecture": "InceptionV3_Optimized",
            "input_size": (299, 299),
            "classes": 10
        }


if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier(num_classes=10, pretrained=True).to(device)

    # Test with CIFAR-10 size input (Batch, Canales, H, W)
    dummy_input = torch.randn(1, 3, 32, 32).to(device)

    model.eval()  # Set to eval mode to get single output
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Architecture: InceptionV3")
    print(f"Output shape: {output.shape}")  # Should be [1, 10]
