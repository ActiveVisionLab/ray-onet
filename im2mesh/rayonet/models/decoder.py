import torch
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.common import sample_plane_feature
from im2mesh.layers import (
    ResnetBlockFC, CResnetBlockConv1d,
    CBatchNorm1d, CBatchNorm1d_legacy,
    ResnetBlockConv1d
)
from im2mesh.pos_encoding import encode_position
class Decoder_simple(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim_local (int): dimension of local feature
        c_dim_global (int): dimension of global feature
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=2, c_dim_local=128, c_dim_global=128, z_resolution=32,
                 hidden_size=256, leaky=False, legacy=False,
                 normalize=False,positional_encoding=False, input3='cam_ori', use_mixer=True):
        super().__init__()
        self.c_dim_local = c_dim_local
        self.c_dim_global = c_dim_global
        self.z_resolution = z_resolution
        self.normalize = normalize
        self.input3 = input3

        if positional_encoding:
            geo_in_dim = c_dim_local + dim * (2 * 4 + 1)
        else:
            geo_in_dim = c_dim_local + dim

        if input3 == 'scale':
            c_dim = c_dim_global + 1
        elif input3 == 'cam_ori':
            c_dim = c_dim_global + 3
        elif input3 == 'scale_dir':
            c_dim = c_dim_global + 4
        elif input3 == 'none':
            c_dim = c_dim_global

        D = hidden_size
        self.layers = nn.Sequential(
            nn.Linear(geo_in_dim+c_dim, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.fc_out = nn.Linear(hidden_size, z_resolution)


    def forward(self, scale_factor, points_xy, c_global, c_local):
        if self.c_dim_local != 0:
            c_local = sample_plane_feature(points_xy, c_local)
            c_local = c_local.transpose(1, 2)
            if self.normalize:
                c_local = F.normalize(c_local, dim=-1)
                net_ = torch.cat([points_xy, c_local], dim=-1)
            else:
                net_ = torch.cat([points_xy, c_local], dim=-1)
        else:
            net_ = points_xy

        if self.input3 == 'none':
            c = c_global
        else:
            c = torch.cat([c_global, scale_factor], dim=-1)
        # c = self.fc_global(c)
        B, N, D = net_.size()
        c = c.unsqueeze(1).expand(-1, N, -1)
        net = torch.cat([net_, c], dim=-1)
        net = self.layers(net)
        out = self.fc_out(net)

        return out
class Decoder_CBatchNorm_scale(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim_local (int): dimension of local feature
        c_dim_global (int): dimension of global feature
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=2, c_dim_local=128, c_dim_global=128, z_resolution=32,
                 hidden_size=256, leaky=False, legacy=False,
                 normalize=False,positional_encoding=False, input3='cam_ori', use_mixer=True):
        super().__init__()
        self.c_dim_local = c_dim_local
        self.c_dim_global = c_dim_global
        self.z_resolution = z_resolution
        self.normalize = normalize
        self.positional_encoding = positional_encoding
        self.input3 = input3
        self.use_mixer = use_mixer

        if positional_encoding:
            geo_in_dim = c_dim_local + dim * (2 * 4 + 1)
        else:
            geo_in_dim = c_dim_local + dim
        if self.use_mixer:
            self.fc_geo1 = ResnetBlockFC(geo_in_dim, hidden_size)
            self.fc_geo2 = ResnetBlockFC(hidden_size, hidden_size)
            self.fc_geo3 = ResnetBlockFC(hidden_size, hidden_size)
        else:
            self.fc_local = nn.Linear(geo_in_dim, hidden_size)

        if input3 == 'scale':
            c_dim = c_dim_global + 1
        elif input3 == 'cam_ori':
            c_dim = c_dim_global + 3
        elif input3 == 'scale_dir':
            c_dim = c_dim_global + 4
        elif input3 == 'none':
            c_dim = c_dim_global

        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.fc_out = nn.Conv1d(hidden_size, z_resolution, 1)

    def forward(self, scale_factor, points_xy, c_global, c_local):
        if self.positional_encoding:
            points_xy = F.normalize(points_xy, p=2, dim=-1)
            points_xy = encode_position(points_xy, 4)
        if self.c_dim_local != 0:
            c_local = sample_plane_feature(points_xy, c_local)
            c_local = c_local.transpose(1, 2)
            if self.normalize:
                c_local = F.normalize(c_local, dim=-1)
                net = torch.cat([points_xy, c_local], dim=-1)
            else:
                net = torch.cat([points_xy, c_local], dim=-1)
        else:
            net = points_xy
        if self.use_mixer:
            net = self.fc_geo1(net)
            net = self.fc_geo2(net)
            net = self.fc_geo3(net)
        else:
            net = self.fc_local(net)

        if self.input3 == 'none':
            c = c_global
        else:
            c = torch.cat([c_global, scale_factor], dim=-1)
        net = net.transpose(1, 2)

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.transpose(1, 2)

        return out