import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageUncertaintyModule(nn.Module):
    def __init__(self, feature_dim):
        super(ImageUncertaintyModule, self).__init__()

        # Quality estimator — learns what clean features look like
        self.quality_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )

        # Noise detector — detects how noisy the input is
        self.noise_detector = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x_flat):
        # Estimate feature quality [0=noisy, 1=clean]
        quality = self.quality_net(x_flat)        # [batch, 1]

        # Detect noise level in input
        noise_level = self.noise_detector(x_flat) # [batch, 1]

        # Adaptive correction strength:
        # High noise detected → stronger correction
        # Low noise detected  → lighter correction
        correction = 0.3 + 0.7 * noise_level      # [batch, 1]

        # Enhance features:
        # high quality → features pass mostly unchanged
        # low quality  → features dampened
        x_enhanced = x_flat * (
            (1 - correction) + correction * quality
        )

        return x_enhanced, quality.squeeze(1)      # [batch, feat], [batch]


class TextUncertaintyModule(nn.Module):
    def __init__(self, feature_dim):
        super(TextUncertaintyModule, self).__init__()

        # Lighter quality estimator for text
        self.quality_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x_txt):
        quality = self.quality_net(x_txt)  # [batch, 1]

        # Very light correction for text (trust text more)
        # 0.8 base + 0.2 from quality → range [0.8, 1.0]
        x_enhanced = x_txt * (0.8 + 0.2 * quality)

        return x_enhanced, quality.squeeze(1)  # [batch, feat], [batch]
