import torch
import torch.nn as nn

from policy.policy_modules.RISE.sparse_modules.aligner import SpatialAligner
from policy.policy_modules.RISE.sparse_modules.pos_emb import SparsePositionalEncoding
from policy.policy_modules.RISE.sparse_modules.minkowski.resnet import ResNet14, ResNet14Mini, ResNet14Max


class SparseEncoder(torch.nn.Module):
    def __init__(
        self, 
        dim_input = 6, 
        dim_output = 512,
        dense_encoder = None,
        dim_sparse_feat = 128,
        interp_fn_mode = "custom",
    ):
        super(SparseEncoder, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        
        self.with_spatial_fusion = dense_encoder is not None

        if self.with_spatial_fusion:
            self.sparse_encoder = ResNet14Mini(
                in_channels = dim_input, 
                out_channels = dim_sparse_feat, 
                conv1_kernel_size = 3, 
                strides = (1, 1, 1, 2), 
                dilations = (1, 2, 4, 8), 
                bn_momentum = 0.02, 
                init_pool = "avg"
            )
            self.dense_encoder = dense_encoder
            dim_dense_feat = dense_encoder.dim_output
            self.spatial_aligner = SpatialAligner(
                mlps = [dim_sparse_feat + dim_dense_feat] * 3,
                interp_fn_mode = interp_fn_mode
            )
            self.fusion_encoder = ResNet14Max(
                in_channels = dim_sparse_feat + dim_dense_feat, 
                out_channels = dim_output, 
                conv1_kernel_size = 3, 
                strides = (4, 2, 2, 2),
                dilations = (4, 1, 1, 1),
                bn_momentum = 0.02, 
                init_pool = None
            )
        else:
            self.sparse_encoder = ResNet14(
                in_channels = dim_input, 
                out_channels = dim_output, 
                conv1_kernel_size = 3,
                strides = (2, 2, 2, 2),
                dilations = (1, 1, 1, 1), 
                bn_momentum = 0.02
            )
        self.position_embedding = SparsePositionalEncoding(dim_output)

    def forward(self, cloud, image = None, image_coord = None, lang = None, max_num_token = 100, batch_size = None):
        ''' max_num_token: maximum token number for each point cloud, which can be adjusted depending on the scene density.
                           100 for voxel_size=0.005 in our experiments
        '''
        cloud_feat = self.sparse_encoder(cloud)

        if self.with_spatial_fusion:
            assert image is not None and image_coord is not None

            image_feat = self.dense_encoder(image, lang = lang)
            image_feat = image_feat.flatten(2).permute(0, 2, 1)
            image_coord = image_coord.flatten(1, 2) # because image_coord here is (B, H, W, 3)

            merged_cloud = self.spatial_aligner(cloud_feat, image_feat, image_coord)
            cloud_feat = self.fusion_encoder(merged_cloud)

        feats_batch, coords_batch = cloud_feat.F, cloud_feat.C
        
        # Infer batch size from coordinates if not provided
        if batch_size is None:
            batch_size = int(coords_batch[:, 0].max().item()) + 1
        
        feats_list = []
        coords_list = []
        for i in range(batch_size):
            mask = (coords_batch[:, 0] == i)
            feats_list.append(feats_batch[mask])
            coords_list.append(coords_batch[mask])
        pos_list = self.position_embedding(coords_list)

        tokens = torch.zeros([batch_size, max_num_token, self.dim_output], dtype = feats_batch.dtype, device = feats_batch.device)
        pos_emb = torch.zeros([batch_size, max_num_token, self.dim_output], dtype = feats_batch.dtype, device = feats_batch.device)
        token_padding_mask = torch.ones([batch_size, max_num_token], dtype = torch.bool, device = feats_batch.device)
        for i, (feats, pos) in enumerate(zip(feats_list, pos_list)):
            num_token = min(max_num_token, len(feats))
            tokens[i, :num_token] = feats[:num_token]
            pos_emb[i, :num_token] = pos[:num_token]
            token_padding_mask[i, :num_token] = False
        
        return tokens, pos_emb, token_padding_mask
