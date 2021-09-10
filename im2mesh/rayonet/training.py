import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from im2mesh.common import (
    compute_iou, make_2d_grid
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer
import im2mesh.common as common
class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        z_resolution (int): number of samples along the ray
        depth_range (list): sampling range on the ray
        input3(str): the form of scale factor
    '''
    def __init__(self, model, optimizer, device=None, input_type='images',
                 vis_dir=None, threshold=0.5, z_resolution=32,
                 depth_range=[0, 2.15],
                 input3='cam_ori',
                 ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.z_resolution = z_resolution

        self.input3 = input3
        self.depth_range = depth_range
        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        scale_type = self.input3
        eval_dict = {}

        device = self.device
        threshold = self.threshold

        points_xy = data.get('points_view').to(device)
        occ_z = data.get('points_view.occ').to(device)
        img = data.get('points_view.images').to(device)

        world_mat = data.get('points_view.world_mat').to(device)

        batch_size, n_points, D = points_xy.size()

        if scale_type == 'scale':
            trans = world_mat[:, :3, -1]
            scale_factor = torch.norm(trans, p=2, dim=1).view(batch_size, 1)
        elif scale_type == 'cam_ori':
            scale_factor = common.origin_to_world(
                1,  world_mat).squeeze(1)
        elif scale_type == 'scale_dir':
            trans = world_mat[:, :3, -1]
            scale = torch.norm(trans, p=2, dim=1).unsqueeze(1)
            dir = F.normalize(trans, dim=-1)
            scale_factor = torch.cat([scale, dir], dim=1).view(batch_size, 4)
        elif scale_type == 'none':
            scale_factor = 0
        else:
            raise ValueError('Invalid scale type "%s"' % scale_type)
        # Compute iou
        with torch.no_grad():
            p_out = self.model(scale_factor, points_xy, img)

        occupancy_loss = F.binary_cross_entropy_with_logits(
            p_out.logits, occ_z, reduction='none')
        eval_dict['loss'] = occupancy_loss.sum(-1).mean().item()

        occ_iou_np = (occ_z >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        return eval_dict


    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        scale_type = self.input3
        z_resolution = self.z_resolution

        img = data.get('points_view.images').to(device)
        batch_size, _, H, W = img.size()

        world_mat = data.get('points_view.world_mat').to(device)

        if scale_type == 'scale':
            trans = world_mat[:, :3, -1]
            scale_factor = torch.norm(trans, p=2, dim=1).view(batch_size, 1)
        elif scale_type == 'cam_ori':
            scale_factor = common.origin_to_world(
                1, world_mat).squeeze(1)
        elif scale_type == 'scale_dir':
            trans = world_mat[:, :3, -1]
            scale = torch.norm(trans, p=2, dim=1).unsqueeze(1)
            dir = F.normalize(trans, dim=-1)
            scale_factor = torch.cat([scale, dir], dim=1).view(batch_size, 4)
        elif scale_type == 'none':
            scale_factor = 0
        else:
            raise ValueError('Invalid scale type "%s"' % scale_type)

        shape = (32, 32)
        points_xy = make_2d_grid([0] * 2, [1.0] * 2, shape).to(device)
        points_xy = points_xy.expand(batch_size, *points_xy.size())

        with torch.no_grad():
            p_r = self.model(scale_factor, points_xy, img)

        occ_hat = p_r.probs.view(batch_size, *shape, z_resolution)
        voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                img[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))


    def compute_loss(self, data, eval_mode=False):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        scale_type = self.input3

        points_xy = data.get('points_view').to(device)
        occ_z = data.get('points_view.occ').to(device)
        img = data.get('points_view.images').to(device)

        world_mat = data.get('points_view.world_mat').to(device)
        batch_size, n_points, D = points_xy.size()

        c, c_local = self.model.encode_inputs(img)

        loss = {}
        if scale_type == 'scale':
            trans= world_mat[:, :3, -1]
            scale_factor = torch.norm(trans, p=2, dim=1).view(batch_size, 1)
        elif scale_type == 'scale_dir':
            trans = world_mat[:, :3, -1]
            scale = torch.norm(trans, p=2, dim=1).unsqueeze(1)
            dir = F.normalize(trans, dim=-1)
            scale_factor = torch.cat([scale, dir], dim=1).view(batch_size, 4)
        elif scale_type == 'cam_ori':
            camera_ori_world = common.origin_to_world(
            1, world_mat).squeeze(1)
            scale_factor = camera_ori_world
        elif scale_type == 'none':
            scale_factor = 0
        else:
            raise ValueError('Invalid scale type "%s"' % scale_type)

        occ_pred = self.model.decode(scale_factor, points_xy, c, c_local) # (B, n_points, num_samples)

        # occupancy loss
        occupancy_loss = F.binary_cross_entropy_with_logits(
            occ_pred.logits, occ_z, reduction='none').sum(-1).mean()
        loss['loss'] = occupancy_loss
        return loss if eval_mode else loss['loss']