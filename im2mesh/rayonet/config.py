import torch
import torch.distributions as dist
from torch import nn
import os
from im2mesh.encoder import encoder_dict
from im2mesh.rayonet import models, training, generation
from im2mesh import data
from im2mesh import config


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    dim = cfg['data']['dim']
    c_dim_local = cfg['model']['c_dim_local']
    c_dim_global = cfg['model']['c_dim_global']
    z_resolution = cfg['model']['z_resolution']
    normalize_local = cfg['model']['normalize']
    positional_encoding = cfg['model']['positional_encoding']
    input3 = cfg['model']['input3']
    use_mixer = cfg['model']['use_mixer']


    decoder = models.decoder_dict[decoder](z_resolution=z_resolution,
        dim=dim, c_dim_local=c_dim_local,c_dim_global=c_dim_global, normalize=normalize_local,
        positional_encoding=positional_encoding, input3=input3,use_mixer=use_mixer
    )

    if encoder is not None:
        encoder = encoder_dict[encoder](
            c_dim_local=c_dim_local, c_dim_global=c_dim_global,
        )
    else:
        encoder = None

    model = models.OccupancyNetwork(
        decoder, encoder, device=device
    )

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    z_resolution = cfg['model']['z_resolution']
    depth_range = cfg['data']['depth_range']

    trainer = training.Trainer(
        model, optimizer,
        device=device,
        vis_dir=vis_dir, threshold=threshold,
        z_resolution=z_resolution,
        input3=cfg['model']['input3'],
        depth_range=depth_range,
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        z_resolution=cfg['model']['z_resolution'],
        resolution0=cfg['generation']['resolution_0'],
        camera=cfg['data']['img_with_camera'],
        depth_range=cfg['data']['depth_range'],
        resolution_regular=cfg['generation']['resolution_regular'],
        input3=cfg['model']['input3'],
        dataset=cfg['data']['dataset']
    )
    return generator


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        cfg (dict): imported yaml config
        mode (str): the mode which is used
    '''
    resize_img_transform = data.ResizeImage(cfg['data']['img_size'])
    if mode == 'train':
        points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    elif mode == 'val':
        points_transform = data.SubsamplePoints(cfg['data']['points_subsample_val'])
    else:
        points_transform = None
    random_view = True if (
        mode == 'train'
    ) else False

    fields = {}
    if cfg['data']['dataset'] == 'Shapes3D':
        imgs_points_field = data.Images_points_Field(
            cfg['data']['img_folder'],
            transform=points_transform,
            transform_img=resize_img_transform,
            extension=cfg['data']['img_extension'],
            with_camera=cfg['data']['img_with_camera'],
            with_transforms=cfg['data']['with_transforms'],
            random_view=random_view,
            unpackbits=cfg['data']['points_unpackbits'],
            z_resolution=cfg['model']['z_resolution'],
            points_file_name=cfg['data']['points_file'],
            mode=mode,
        )
        fields['points_view'] = imgs_points_field
    return fields
