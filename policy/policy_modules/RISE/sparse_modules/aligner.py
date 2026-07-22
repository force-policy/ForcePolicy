import torch
from torch import nn
import MinkowskiEngine as ME

from policy.policy_modules.RISE.sparse_modules.minkowski.resnet import ResNet14Max, ResNet14Mini
from policy.policy_modules.RISE.sparse_modules.pos_emb import SparsePositionalEncoding


class MLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        '''x: (B, C, N)'''
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class SharedMLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.blocks = nn.Sequential()
        for i in range(len(layers) - 1):
            self.blocks.append(MLPBlock(layers[i], layers[i+1]))

    def forward(self, x):
        '''x: (B, C, N)'''
        x = self.blocks(x)
        return x


class CustomWeightedInterpFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src_feats, selected_idxs):
        """
        src_feats: (C, m)
        selected_idxs: (n, k)
        """
        selected_idxs_expand = selected_idxs.unsqueeze(1).expand(-1, src_feats.size(0), -1) # (n, C, k)
        curr_src_feats_expand = src_feats.unsqueeze(0).expand(selected_idxs.size(0), -1, -1) # (n, C, m)
        selected_feats = torch.gather(curr_src_feats_expand, 2, selected_idxs_expand) # (n, C, k)

        ctx.save_for_backward(src_feats, selected_idxs, selected_idxs_expand)

        return selected_feats
    
    @staticmethod
    def backward(ctx, grad_out):
        src_feats, selected_idxs, selected_idxs_expand = ctx.saved_tensors

        grad_src_feats = torch.zeros_like(src_feats) # (C, m)
        grad_selected_idxs = torch.zeros_like(selected_idxs)
        selected_idxs_expand = selected_idxs_expand.permute(1, 0, 2).contiguous() # (C, n, k)
        grad_out = grad_out.permute(1, 0, 2).contiguous() # (C, n, k)

        for i in range(grad_out.size(2)):
            grad_src_feats.scatter_add_(1, selected_idxs_expand[:,:,i], grad_out[:,:,i])

        return grad_src_feats, grad_selected_idxs


class WeightedSpatialInterpolation(nn.Module):
    def __init__(self, interp_fn_mode = 'custom'):
        super().__init__()
        assert interp_fn_mode in ['custom', 'naive']
        self.interp_fn_mode = interp_fn_mode


    def forward(
        self, tgt, src, tgt_feats, src_feats, k=3
    ) -> torch.Tensor:
        """
        Args:
            tgt: (B, n, 3) tensor of the xyz positions of the target features
            src: (B, m, 3) tensor of the xyz positions of the source features
            tgt_feats: (B, C1, n) tensor of the target features
            src_feats: (B, C2, m) tensor of the source features

        Returns:
            interp_features : (B, mlp[-1], n) tensor of the features of the interpolated features
        """
        interpolated_feats = []
        for i in range(tgt.size(0)):
            all_dists = torch.linalg.norm(tgt[i].unsqueeze(1)-src[i].unsqueeze(0), dim=2) # (n, m)
            all_dists, all_idxs = torch.sort(all_dists, dim=1)

            if self.interp_fn_mode == 'naive':
                selected_idxs = all_idxs[:, :k].unsqueeze(1).expand(-1, src_feats.size(1), -1) # (n, C2, k)
                curr_src_feats = src_feats[i:i+1].expand(selected_idxs.size(0), -1, -1) # (n, C2, m)
                selected_feats = torch.gather(curr_src_feats, 2, selected_idxs) # (n, C2, k)
            else: # 'custom'
                selected_idxs = all_idxs[:, :k] # (n, k)
                selected_feats = CustomWeightedInterpFn.apply(src_feats[i], selected_idxs) # (n, C2, k)

            weight = 1.0 / (all_dists[:, :k] + 1e-6)
            norm = torch.sum(weight, dim=1, keepdim=True)
            weight = weight / norm
            selected_feats = (selected_feats * weight.unsqueeze(1)).sum(dim=2) # (n, C2)
            interpolated_feats.append(selected_feats)

        interpolated_feats = torch.stack(interpolated_feats, dim=0).permute(0, 2, 1) # (B, C2, n)
        interpolated_feats = torch.cat([interpolated_feats, tgt_feats], dim=1)  #(B, C2 + C1, n)

        return interpolated_feats


class SpatialAligner(nn.Module):
    def __init__(self, mlps, interp_fn_mode = 'custom'):
        """
        Args:
            interp_fn_mode: str, "naive"/"custom"
        """
        super().__init__()
        self.interp = WeightedSpatialInterpolation(interp_fn_mode = interp_fn_mode)
        self.interp_proj = SharedMLP(mlps)

    def forward(self, sinput, image_feat, image_coord):
        batch_size = image_feat.size(0)
        cloud_feat, cloud_coord = sinput.F, sinput.C

        cloud_feat_list = []
        for i in range(batch_size):
            cloud_mask_i = cloud_coord[:, 0] == i
            cloud_coord_i = cloud_coord[cloud_mask_i][:, 1:].unsqueeze(0)
            cloud_feat_i = cloud_feat[cloud_mask_i].permute(1, 0).unsqueeze(0)
            image_coord_i = image_coord[i:i+1]
            image_feat_i = image_feat[i].permute(1, 0).unsqueeze(0)
            cloud_feat_i = self.interp(cloud_coord_i.float(), image_coord_i.float(), cloud_feat_i, image_feat_i)
            cloud_feat_list.append(cloud_feat_i)
        cloud_feat = torch.cat(cloud_feat_list, dim=2)
        cloud_feat = self.interp_proj(cloud_feat)
        cloud_feat = cloud_feat.squeeze(0).permute(1, 0)
        return ME.SparseTensor(cloud_feat, sinput.C)