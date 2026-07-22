import torch
import torch.nn as nn
import torch.nn.functional as F


class SparsePositionalEncoding(nn.Module):
    """
    Sparse positional encoding for point tokens, similar to the standard version
    """
    def __init__(self, num_pos_feats=512, temperature=10000, max_pos=1400):   ##TODO: 1400 for 0.002, 800 for 0.005
        super().__init__()
        ''' max_pos: position range will be [-max_pos/2, max_pos/2) along X/Y/Z-axis.
                     remeber to keep this value fixed in your training and evaluation.
                     800 for voxel_size=0.005 in our experiments. TODO: may need point centralization?
        '''
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.max_pos = max_pos
        self.origin_pos = max_pos // 2
        self._init_position_vector()

    def _init_position_vector(self):
        x_steps = y_steps = self.num_pos_feats // 3
        z_steps = self.num_pos_feats - x_steps - y_steps
        xyz_embed = torch.arange(self.max_pos, dtype=torch.float32)[:,None]

        x_dim_t = torch.arange(x_steps, dtype=torch.float32)
        y_dim_t = torch.arange(y_steps, dtype=torch.float32)
        z_dim_t = torch.arange(z_steps, dtype=torch.float32)
        x_dim_t = self.temperature ** (2 * (x_dim_t // 2) / x_steps)
        y_dim_t = self.temperature ** (2 * (y_dim_t // 2) / y_steps)
        z_dim_t = self.temperature ** (2 * (z_dim_t // 2) / z_steps)

        pos_x_vector = xyz_embed / x_dim_t
        pos_y_vector = xyz_embed / y_dim_t
        pos_z_vector = xyz_embed / z_dim_t
        self.pos_x_vector = torch.stack([pos_x_vector[:,0::2].sin(), pos_x_vector[:,1::2].cos()], dim=2).flatten(1)
        self.pos_y_vector = torch.stack([pos_y_vector[:,0::2].sin(), pos_y_vector[:,1::2].cos()], dim=2).flatten(1)
        self.pos_z_vector = torch.stack([pos_z_vector[:,0::2].sin(), pos_z_vector[:,1::2].cos()], dim=2).flatten(1)

    def forward(self, coords_list):
        pos_list = []
        for coords in coords_list:
            coords = (coords[:,1:4] + self.origin_pos).long()
            coords[:,0] = torch.clamp(coords[:,0], 0, self.max_pos-1)
            coords[:,1] = torch.clamp(coords[:,1], 0, self.max_pos-1)
            coords[:,2] = torch.clamp(coords[:,2], 0, self.max_pos-1)
            pos_x = self.pos_x_vector.to(coords.device)[coords[:,0]]
            pos_y = self.pos_y_vector.to(coords.device)[coords[:,1]]
            pos_z = self.pos_z_vector.to(coords.device)[coords[:,2]]
            pos = torch.cat([pos_x, pos_y, pos_z], dim=1)
            pos_list.append(pos)
        return pos_list