import trimesh
import torch
import numpy as np
import os
from plyfile import PlyData, PlyElement
import argparse
import glob
import sys
from multiprocessing import Pool
from functools import partial
sys.path.append('..')
from im2mesh.utils.libmesh import check_mesh_contains
import im2mesh.common as common

parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('in_folder', type=str,
                    help='Path to input watertight meshes.')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')
parser.add_argument('--camera_folder', type=str,
                    help='path to camera parameters')
parser.add_argument('--points_folder', type=str,
                    help='Output path for points.')
parser.add_argument('--points_size', type=int, default=5000,
                    help='Size of points.')
parser.add_argument('--z_resolution', type=int, default=32,
                    help='number of samples along the ray.')
parser.add_argument('--depth_min', type=int, default=0.63,
                    help='minimum depth of sampling interval')
parser.add_argument('--depth_max', type=int, default=2.16,
                    help='maximum depth of sampling interval')
parser.add_argument('--packbits', action='store_true',
                help='Whether to save truth values as bit array.')

def b_inv(b_mat):
    ''' Performs batch matrix inversion.

    Arguments:
        b_mat: the batch of matrices that should be inverted
    '''

    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    # b_inv, _ = torch.gesv(eye, b_mat)
    b_inv, _ = torch.solve(eye, b_mat)
    return b_inv

def get_rays3(points_xy, focal, world_mat, n_xy, N_sam, depth_min, depth_max):
    R = world_mat[:, :, :3]
    x = 2 * points_xy[:, 0] - 1
    y = 2 * points_xy[:, 1] - 1

    dirs_x = x / focal  # (H, W)
    dirs_y = y / focal  # (H, W)
    dirs_z = torch.ones(n_xy, dtype=torch.float32)  # (H, W)
    ray_dir_cam = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # (H, W, 3)
    ray_dir_world = ray_dir_cam @ b_inv(R.transpose(1, 2))
    ray_ori_world = common.origin_to_world(1, world_mat)
    t_vals = torch.linspace(depth_min, depth_max, N_sam)
    t_vals_noisy = t_vals.view(1, N_sam).expand(n_xy, N_sam)
    pts = ray_ori_world + ray_dir_world.unsqueeze(2) * t_vals_noisy.unsqueeze(2)
    return pts
def main(args):
    input_files = glob.glob(os.path.join(args.in_folder, '*.off'))
    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(process_path, args=args), input_files)
    else:
        for p in input_files:
            process_path(p, args)

def process_path(in_path, args):
    in_file = os.path.basename(in_path)
    modelname = os.path.splitext(in_file)[0]
    mesh = trimesh.load(in_path, process=False)
    export_points(mesh, modelname, args)

def export_points(mesh, modelname, args):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot sample points.' % modelname)
        return
    depth_min = args.depth_min
    depth_max = args.depth_max
    n_xy = args.points_size
    N_samples = args.z_resolution

    camera_file = os.path.join(args.camera_folder, modelname + '/img_choy2016/cameras.npz')
    out_dir_cat = args.points_folder
    camera_dict = np.load(camera_file)
    out_dir_model = os.path.join(out_dir_cat, modelname)
    if not os.path.exists(out_dir_model):
        os.makedirs(out_dir_model)
    for idx_img in range(24):
        world_mat = torch.tensor(camera_dict['world_mat_%d' % idx_img].astype(np.float32)).unsqueeze(0)
        camera_mat = torch.tensor(camera_dict['camera_mat_%d' % idx_img].astype(np.float32)).unsqueeze(0)
        camera_mat = common.fix_K_camera(camera_mat, img_size=137.)
        focal = camera_mat[0][0][0]
        points_xy = torch.rand(n_xy, 2)
        pts = get_rays3(points_xy, focal, world_mat, n_xy, N_samples, depth_min=depth_min, depth_max=depth_max)
        pts = pts.reshape(-1, 3).numpy()
        occupancies = check_mesh_contains(mesh, pts)
        if args.packbits:
            occupancies = np.packbits(occupancies)
        filename = os.path.join(out_dir_model, str(idx_img) + '.npz')
        print('Writing points: %s' % filename)
        np.savez(filename, points_xy=points_xy, occupancies=occupancies)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
