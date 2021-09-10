import torch.nn as nn
import torchvision
import torch
from im2mesh.common import normalize_imagenet
import torch.nn.functional as F
class ResNet18Encoder(nn.Module):

    def __init__(
        self,
        c_dim_local,
        c_dim_global,
        backbone="resnet18",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        upsample_interp="bilinear",
        use_first_pool=True,
        normalize=True,
    ):

        super().__init__()

        self.use_first_pool = use_first_pool
        self.c_dim_local = c_dim_local
        self.features = getattr(torchvision.models, backbone)(
            pretrained=pretrained
        )
        self.features.fc = nn.Sequential()
        self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.upsample_interp = upsample_interp

        self.fc = nn.Linear(512, 256)
        self.fc_global = nn.Linear(256, c_dim_global)
        self.normalize = normalize
        if c_dim_local!=0:
            self.conv_local = nn.Conv2d(512, c_dim_local, 1)

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)

        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)

        latents = [x]
        if self.num_layers > 1:
            if self.use_first_pool:
                x = self.features.maxpool(x)
            x = self.features.layer1(x)
            latents.append(x)
        x = self.features.layer2(x)
        if self.num_layers > 2:
            latents.append(x)
        x = self.features.layer3(x)
        if self.num_layers > 3:
            latents.append(x)
        x = self.features.layer4(x)
        if self.num_layers > 4:
            latents.append(x)
        x = self.features.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        global_feature = self.fc_global(x)

        if self.c_dim_local!=0:
            align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:]
            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )

            latent = torch.cat(latents, dim=1)
            latent = self.conv_local(latent)
        else:
            latent = None
        return global_feature, latent
