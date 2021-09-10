import os
import glob
import random
from PIL import Image
import imageio
import numpy as np
import trimesh
from im2mesh.data.core import Field
from im2mesh.utils import binvox_rw
import torch
import cv2


class Images_points_Field(Field):
    ''' Image and gt occupancy Field.

    It is the field used for loading images.

    Args:
        folder_name (str): image folder name
        transform (list): list of transformations which will be applied to the points tensor
        transform_img (transform): transformations applied to images
        extension (str): image extension
        with_camera (bool): whether camera data should be provided
        random_view (bool): whether to use random view
        unpackbits (bool): whether data need to be unpacked
        z_resolution (int): number of samples along the ray
        points_file_name (str): file name for sample points geenrated by occupancy networks
        with_transforms (bool): whether scale and loc need to be loaded for normalisation
        mode (str): the mode which is used
    '''

    def __init__(self, folder_name,
                 transform=None, transform_img=None, extension='jpg',
                 with_camera=False, random_view=True,
                 unpackbits=False, z_resolution=64, points_file_name=None,
                 with_transforms=True, mode='train',**kwargs):
        self.folder_name = folder_name
        self.transform = transform
        self.transform_img = transform_img
        self.extension = extension
        self.occ_extension = 'npz'

        self.random_view = random_view

        self.with_camera = with_camera

        self.unpackbits = unpackbits
        self.z_resolution = z_resolution

        self.points_file_name = points_file_name
        self.with_transforms = with_transforms # load scale and loc for normalisation
        self.mode = mode

    def load(self, model_path_2, idx, category, input_idx_img=None):
        ''' Loads the field.

        Args:
            model_path_2 (str): path to model and occupancy ground truth
            idx (int): model id
            category (int): category id
            input_idx_img (int): image id which should be used (this
                overwrites any other id). This is used when the fields are
                cached.
        '''
        model_path = model_path_2[0]
        occ_path = model_path_2[1]

        return self.load_field(model_path, occ_path, idx, category, input_idx_img)

    def get_number_files(self, model_path):
        ''' Returns how many views are present for the model.

        Args:
            model_path (str): path to model
        '''
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
        files.sort()
        return len(files)

    def load_input(self, model_path, idx_img, data={}):
        ''' Load the input image
        Args:
            model_path (str): path to model
            idx (int): ID of data point
            data (dict): data dictionary
        '''
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
        files.sort()

        filename = files[idx_img]

        image = Image.open(filename).convert('RGB')

        if self.transform_img is not None:
            image = self.transform_img(image)
        data['images'] = image
        if self.with_camera:
            camera_file = os.path.join(folder, 'cameras.npz')
            camera_dict = np.load(camera_file)
            Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
            K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
            data['world_mat'] = Rt
            data['camera_mat'] = K


    def load_occupancy(self, model_path, idx, data={}):
        ''' Loads the occupancy ground truth for rays

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            data (dict): data dictionary
        '''

        filename = model_path + '/' + str(idx) + '.' + self.occ_extension
        points_dict = np.load(filename)
        points_xy = points_dict['points_xy']
        if points_xy.dtype == np.float16:
            points_xy = points_xy.astype(np.float32)
            points_xy += 1e-4 * np.random.randn(*points_xy.shape)
        else:
            points_xy = points_xy.astype(np.float32)
        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:5000*self.z_resolution]
        occupancies = occupancies.reshape(5000, self.z_resolution)
        occupancies = occupancies.astype(np.float32)

        data['occ'] = occupancies
        data[None] = points_xy


    def load_field(self, model_path, occ_path, idx, category, input_idx_img=None):
        ''' Loads the data.

        Args:
            model_path (str): path to model
            occ_path (str): path to occupancy ground truth
            idx (int): ID of data point
            category (int): index of category
            input_idx_img (int): image id which should be used (this
                overwrites any other id). This is used when the fields are
                cached.
        '''

        n_files = self.get_number_files(model_path)

        if input_idx_img is not None:
            idx_img = input_idx_img
        elif self.random_view:
            idx_img = random.randint(0, n_files - 1)
        else:
            idx_img = 0
        # Load the data
        data = {}
        self.load_input(model_path, idx_img, data)
        if self.mode in ('train', 'val'):
            self.load_occupancy(occ_path, idx_img, data)
        file_path = os.path.join(model_path, self.points_file_name)
        points_dict = np.load(file_path)
        if self.with_transforms:
            data['loc'] = points_dict['loc'].astype(np.float32)
            data['scale'] = points_dict['scale'].astype(np.float32)
        if self.transform is not None:  # subsample points
            data = self.transform(data)
        return data

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        complete = (self.folder_name in files)
        # TODO: check camera
        return complete