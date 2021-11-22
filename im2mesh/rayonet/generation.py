import torch
import numpy as np
import trimesh
from im2mesh.utils import libmcubes
from im2mesh.common import make_2d_grid
import time
import im2mesh.common as common
import torch.nn as nn
import torch.nn.functional as F

class Generator3D(object):
    def __init__(self, model,
                 threshold=0.5, device=None,
                 z_resolution=32,
                 resolution0=16,
                 camera=True, depth_range=[0, 1], resolution_regular=128,
                 input3='scale', dataset='Shapes3D'):
        self.model = model.to(device)
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.z_resolution = z_resolution
        self.camera = camera
        self.depth_range = depth_range
        self.resolution_regular = resolution_regular
        self.input3 = input3
        self.dataset = dataset

    def generate_mesh(self, data, return_stats=True):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('points_view.images').to(device)
        kwargs = {}

        # Encode inputs
        t0 = time.time()
        with torch.no_grad():
            c, c_local = self.model.encode_inputs(inputs)
        stats_dict['time (encode inputs)'] = time.time() - t0

        mesh = self.generate_from_latent(c, c_local, data,stats_dict=stats_dict, **kwargs)
        if return_stats:
            return mesh, stats_dict
        else:
            return mesh
    def generate_from_latent(self, c=None, c_local=None, data=None, stats_dict={}, **kwargs):
        t0 = time.time()
        nz = self.z_resolution
        nx = self.resolution0
        scale_type = self.input3
        nx_regular = self.resolution_regular
        y, x = torch.meshgrid(torch.linspace(0, 1, nx, dtype=torch.float32),
                              torch.linspace(0, 1, nx, dtype=torch.float32))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        points_xy = torch.cat([x, y], dim=-1)
        points_xy = points_xy.unsqueeze(0).to(self.device)

        if self.camera:
            world_mat = data.get('points_view.world_mat').to(self.device)
            if scale_type == 'scale':
                trans = world_mat[:, :3, -1]
                scale_factor = torch.norm(trans, p=2, dim=1).view(1, 1)
            elif scale_type == 'cam_ori':
                scale_factor = common.origin_to_world(
                    1, world_mat).squeeze(1)
            elif scale_type == 'scale_dir':
                trans = world_mat[:, :3, -1]
                scale = torch.norm(trans, p=2, dim=1).unsqueeze(1)
                dir = F.normalize(trans, dim=-1)
                scale_factor = torch.cat([scale, dir], dim=1).view(1, 4)
            elif scale_type == 'none':
                scale_factor = 0
            else:
                raise ValueError('Invalid scale type "%s"' % scale_type)
        else:
            if scale_type == 'scale':
                scale_factor = 1.5 * torch.ones(1, 1).to(self.device)
            elif scale_type == 'none':
                scale_factor = 0
            else:
                raise ValueError('Invalid scale type "%s"' % scale_type)

        ray_dir = points_xy
        values = self.eval_points(scale_factor, points_xy, ray_dir, c, c_local, **kwargs) # (1, 4096, 64)

        value_grid = values.reshape(nx, nx, nz)
        stats_dict['time (eval points)'] = time.time() - t0
        # change to regular grid in order to use marching cube
        t0 = time.time()
        value_grid = self.to_regular_grid(value_grid)
        stats_dict['time (to_regular_grid)'] = time.time() - t0
        value_grid = value_grid.reshape(nx_regular, nx_regular, nz)

        value_grid = value_grid.cpu().numpy()
        if self.camera:
            camera_mat = data.get('points_view.camera_mat').to(self.device)
            scale = data.get('points_view.scale').to(self.device)
            loc = data.get('points_view.loc').to(self.device)
            world_mat = common.fix_Rt_camera(world_mat, loc, scale)
            mesh = self.extract_mesh(value_grid, world_mat, camera_mat, stats_dict=stats_dict)
        else:
            mesh = self.extract_mesh(value_grid, stats_dict=stats_dict)
        return mesh

    def to_regular_grid(self, value_grid):
        depth_max = self.depth_range[1]
        depth_min = self.depth_range[0]
        nx = self.resolution_regular
        nz = self.z_resolution
        value_grid = value_grid.permute(2, 0, 1).unsqueeze(1)
        bias = 1e-6
        scale = depth_max / (depth_min + (depth_max - depth_min) * torch.linspace(0, 1, nz) + bias).to(self.device)
        points_xy = make_2d_grid((-1.0,) * 2, (1.0,) * 2, (nx,) * 2).unsqueeze(-1).expand(nx * nx, 2, nz).to(
            self.device)
        points_xy_scaled = torch.mul(scale, points_xy).permute(2, 0, 1).unsqueeze(1)
        scaled_grid = F.grid_sample(value_grid, points_xy_scaled, padding_mode='zeros', align_corners=True,
                                    mode='bilinear').permute(1, 2, 3, 0)
        return scaled_grid

    def eval_points(self,scale_factor, points_xy, ray_dir, c=None, c_local=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            scale_factor (tensor): scale calibration factor
            points_xy (tensor): points
            ray_dir:
            c (tensor): global feature c
        '''
        with torch.no_grad():
            occ_hat = self.model.decode(scale_factor, points_xy, c, c_local).probs

        occ_hat = occ_hat.detach()

        return occ_hat

    def extract_mesh(self, occ_hat, world_mat=None, cam_mat=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            world_mat (tensor): world matrix
            cam_mat (tensor): camera matrix
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        threshold = self.threshold
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices = torch.from_numpy(vertices.astype(np.float32))
        vertices -= 0.5
        vertices -= 1
        # Normalize to bounding box
        vertices /= torch.from_numpy(np.array([n_x - 1, n_y - 1, n_z - 1]).astype(np.float32))
        # scale
        vertices[:, 0] = vertices[:, 0] - 0.5
        vertices[:, 1] = vertices[:, 1] - 0.5  # x, y in range (-0.5, 0.5), z in range (0, 1)

        depth_max = self.depth_range[1]
        depth_min = self.depth_range[0]

        img_size = 137 # image size for images used during training
        focal = 149.8438 # focal length for images used during training
        scale_x = depth_max * img_size / focal
        scale_z = depth_max - depth_min

        vertices[:, 0] = vertices[:, 0] * scale_x
        vertices[:, 1] = vertices[:, 1] * scale_x
        vertices[:, 2] = vertices[:, 2] * scale_z

        if self.camera:
            vertices[:, 2] = vertices[:, 2] + depth_min
            # transform to world coordinate
            t0 = time.time()
            vertices = vertices.unsqueeze(0).to(self.device)
            stats_dict['time (to gpu)'] = time.time() - t0
            t0 = time.time()
            vertices = common.transform_points_back(vertices, world_mat)
            stats_dict['time (rotate)'] = time.time() - t0
            vertices = vertices.squeeze().cpu().numpy()

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=None,
                               process=False)
        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        return mesh

