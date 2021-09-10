import torch
import torch.nn as nn
from torch import distributions as dist
from im2mesh.rayonet.models import decoder

# Decoder dictionary
decoder_dict = {
    'cbatchnorm_local_scale': decoder.Decoder_CBatchNorm_scale,
    'simple':decoder.Decoder_simple
}


class OccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoder, encoder=None,
                 device=None):
        super().__init__()

        self.decoder = decoder.to(device)


        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

    def forward(self, scale_factor, points_xy, img):
        ''' Performs a forward pass through the network.

        Args:
            scale_factor (tensor): scale factor
            p_xy (tensor): sampled 2d points representing a ray
            img (tensor): input image
        '''
        c, c_local = self.encode_inputs(img)
        probs = self.decode(scale_factor, points_xy, c, c_local)  # (B, n_points, num_samples)
        return probs


    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c, c_local = self.encoder(inputs)
        else:
            # Return inputs?
            c, c_local = torch.empty(inputs.size(0), 0).to(torch.device('cuda:0'))

        return c, c_local

    def decode(self, scale_factor, points_xy, c_global, c_local):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            scale_factor (tensor): scale factor
            points_xy (tensor): sampled 2d points representing a ray
            c_global (tensor): global feature
            c_local (tensor): local feature
        '''

        logits = self.decoder(scale_factor, points_xy, c_global, c_local)
        p_r = dist.Bernoulli(logits=logits)
        return p_r


    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
