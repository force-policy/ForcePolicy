"""
Data Processor
"""
from typing import Dict, Any, List, Tuple, Literal, Optional

import sys
import torch

import torch.nn.functional as F

from logger import logger

from data_infra.configs import (
    DataType,
    RotationType,
    SpatialDataTypes,
    DataProcessorConfig
)

from data_infra.ops.point_cloud import voxel_downsample
from data_infra.ops.point_cloud import fixed_number_downsample
from data_infra.ops.data_source import get_image
from data_infra.ops.data_source import get_lowdim
from data_infra.ops.data_source import get_depth_points
from data_infra.ops.augmentation import AugmentationContext
from data_infra.ops.augmentation import augment_random_transform
from data_infra.ops.normalization import normalize, unnormalize

from utils.transforms.pose import xyz_rot_to_mat
from utils.transforms.pose import transform_matmul
from utils.transforms.pose import xyz_rot_transform
from utils.transforms import projection as projection_utils


class CombinedDataset:
    def __init__(self, obs: Dict[str, Any], action: Optional[Dict[str, Any]] = None):
        self.obs = obs
        self.action = action or {}

    def __getitem__(self, key: str) -> Any:
        if key.startswith("obs."):
            return self.obs[key[4:]]
        if key.startswith("action."):
            return self.action[key[7:]]
        
        if key in self.action:
            return self.action[key]
        if key in self.obs:
            return self.obs[key]
        raise KeyError(key)
    
    def __contains__(self, key: str) -> bool:
        if key.startswith("obs."):
            return key[4:] in self.obs
        if key.startswith("action."):
            return key[7:] in self.action
        return key in self.action or key in self.obs

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class DataProcessor:
    """
    Data Processor Class.
    """
    def __init__(
        self,
        config: DataProcessorConfig
    ) -> None:
        self.obs_data_configs = config.obs_data_configs
        self.action_data_configs = config.action_data_configs
        self.action_data_reverse_configs = config.action_data_reverse_configs
        self.prefix_config = config.prefix_config
        self.augmentation_configs = config.augmentation_configs
        self.spatial_aug_frame = config.spatial_aug_frame

        # TODO: check: each spatial augmentation should be only in one point cloud config
        # TODO: check: observation should not use action source


    def get_world_transform(
        self,
        data: Dict[str, Any],
        frame: str,
        device: torch.device,
        **kwargs
    ) -> torch.Tensor:
        """ Get the transformation from world to the specified frame. """
        if frame == "world":
            return torch.eye(4, dtype = torch.float32, device = device)

        elif frame.startswith("camera/"):
            return data[f"{self.prefix_config.extrinsic}/{frame[7:]}"]

        elif frame.startswith("robot_base/"):
            return data[f"{self.prefix_config.robot}/{frame[11:]}"]

        elif frame.startswith("robot_tcp/"):
            info_list = frame[10:].split('/')
            robot_name = info_list[0]
            aux_tcp_key = info_list[1]
            rotation_rep = RotationType.from_str(info_list[2])
            convention = info_list[3] if len(info_list) == 4 else None
            T_world_base = data[f"{self.prefix_config.robot}/{robot_name}"]
            T_base_tcp = xyz_rot_to_mat(
                data[aux_tcp_key],
                rotation_rep = rotation_rep,
                convention = convention
            )
            return transform_matmul(T_world_base, T_base_tcp)

        else:
            raise ValueError(f"Unknown frame: {frame}")
    
    def get_transform(
        self,
        data: Dict[str, Any],
        from_frame: str,
        to_frame: str,
        device: torch.device,
        **kwargs
    ):
        """ Get the transformation from the from_frame to the to_frame. """
        if from_frame == to_frame:
            return torch.eye(4, dtype = torch.float32, device = device)
        
        T_world_from = self.get_world_transform(data, from_frame, device)
        T_world_to = self.get_world_transform(data, to_frame, device)
        return transform_matmul(torch.linalg.inv(T_world_from), T_world_to)
    

    def _process_configs(
        self,
        configs: Dict[str, Any],
        raw_dataset: CombinedDataset,
        enable_aug: bool,
        aug_ctx: Optional[AugmentationContext]
    ) -> Dict[str, Any]:
        """ Process configs. """
        results = {}
        
        # Separate configs
        pcd_configs = {k: v for k, v in configs.items() if v.type == DataType.POINT_CLOUD}
        other_configs = {k: v for k, v in configs.items() if v.type != DataType.POINT_CLOUD}
        
        ####################
        # Point Cloud Data #
        ####################
        for out_name, data_cfg in pcd_configs.items():
            # Load point clouds
            pcd_list = []
            for src_cfg in data_cfg.src:
                # TODO: consider inhand camera point cloud merge.
                assert src_cfg.frame == "world" # Currently, point cloud should be merged in world frame.
                pcd_list.append(get_depth_points(
                    raw_dataset.obs, 
                    source_config = src_cfg,
                    prefix_config = self.prefix_config,
                    enable_aug = enable_aug,
                    aug_configs = self.augmentation_configs,
                    aug_ctx = aug_ctx
                ))
            
            batch_size = len(pcd_list[0])
            device = pcd_list[0][0].device    

            # Merge and Downsample
            merged_pcds = []
            for b in range(batch_size):
                merged_pcd = torch.cat([pcd_list[s][b] for s in range(len(pcd_list))], dim = 0)
                if data_cfg.voxelization:
                    merged_pcd = voxel_downsample(merged_pcd, data_cfg.voxel_size)
                if data_cfg.fixed_number:
                    merged_pcd = fixed_number_downsample(
                        merged_pcd,     
                        data_cfg.num_points,
                        data_cfg.sampling_method
                    )
                merged_pcds.append(merged_pcd)

            # Augmentation and transformation
            if enable_aug and len(data_cfg.aug_groups) > 0:
                # Compute centroid as reference
                centroids = torch.zeros((batch_size, 3), device = device, dtype = merged_pcds[0].dtype)
                for b in range(batch_size):
                    centroids[b] = merged_pcds[b][:, :3].mean(dim = 0)
                
                # Compute transformation from world to aug frame
                T_aug_frame_world = self.get_transform(raw_dataset, self.spatial_aug_frame, "world", device)

                # Project to spatial augmentation frame
                centroids = projection_utils.apply_mat_to_point(centroids, mat = T_aug_frame_world)

                # Calculate transformation matrix for augmentation
                T_aug = augment_random_transform(
                    batch_size = batch_size,
                    dtype = merged_pcds[0].dtype,
                    device = device,
                    aug_groups = data_cfg.aug_groups,
                    aug_configs = self.augmentation_configs,
                    aug_ctx = aug_ctx,
                    centroids = centroids
                )

                # Calculate final transformation
                T_target_aug_frame = self.get_transform(raw_dataset, data_cfg.frame, self.spatial_aug_frame, device)
                T_final = transform_matmul(T_target_aug_frame, transform_matmul(T_aug, T_aug_frame_world))

            else:
                # Project from world frame
                T_final = self.get_transform(raw_dataset, data_cfg.frame, "world", device)

            for b in range(batch_size):
                # Apply all transformation
                merged_pcds[b][:, :3] = projection_utils.apply_mat_to_point(merged_pcds[b][:, :3], mat = T_final[b])
                # Normalization
                merged_pcds[b] = normalize(merged_pcds[b], data_cfg.norm_config)
                    
            # Final: to target representation
            if data_cfg.fixed_number:
                results[out_name] = torch.cat(merged_pcds, dim = 0)
            else:
                if data_cfg.backend == 'MinkowskiEngine':
                    import MinkowskiEngine as ME
                    coords_batch = [torch.floor(pcd[:, :3] / data_cfg.voxel_size).long() for pcd in merged_pcds]
                    feats_batch = [pcd for pcd in merged_pcds]
                    coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
                    results[out_name] = ME.SparseTensor(feats_batch, coords_batch)
                else:
                    raise ValueError(f"Unknown backend: {data_cfg.backend}")
        
        ##############
        # Other Data #
        ##############
        for out_name, data_cfg in other_configs.items():
            # Get Data
            if data_cfg.type in [DataType.IMAGE, DataType.DEPTH]:
                data_getter_fn = get_image
            elif data_cfg.type == DataType.POINT and data_cfg.src.type == DataType.DEPTH:
                data_getter_fn = get_depth_points
            else:
                data_getter_fn = get_lowdim
            
            data = data_getter_fn(
                data_dict = raw_dataset,
                source_config = data_cfg.src,
                prefix_config = self.prefix_config,
                enable_aug = enable_aug,
                aug_configs = self.augmentation_configs,
                aug_ctx = aug_ctx,
            )

            # Rotation representation transform for pose data
            if data_cfg.type == DataType.POSE:
                data = xyz_rot_transform(
                    xyz_rot = data,
                    from_rep = data_cfg.src.rotation_rep,
                    to_rep = data_cfg.rotation_rep,
                    from_convention = data_cfg.src.convention,
                    to_convention = data_cfg.convention
                )
            
            # Resize for image data
            if data_cfg.type in [DataType.IMAGE, DataType.DEPTH]:
                if data_cfg.size:
                    assert data_cfg.interp_mode is not None
                    data = F.interpolate(
                        data, 
                        size = data_cfg.size,
                        mode = data_cfg.interp_mode
                    )
            
            # Transformations for spatial data
            if data_cfg.type in SpatialDataTypes:
                batch_size = data.shape[0]
                device = data.device

                # augmentation
                if enable_aug and len(data_cfg.aug_groups) > 0:
                    # Calculate spatial augmentation transformations
                    T_aug = augment_random_transform(
                        batch_size = batch_size,
                        dtype = data.dtype,
                        device = device,
                        aug_groups = data_cfg.aug_groups,
                        aug_configs = self.augmentation_configs,
                        aug_ctx = aug_ctx,
                        centroids = None
                    )

                    # Calculate final transformation
                    T_aug_frame_src = self.get_transform(raw_dataset, self.spatial_aug_frame, data_cfg.src.frame, device)
                    T_target_aug_frame = self.get_transform(raw_dataset, data_cfg.frame, self.spatial_aug_frame, device)
                    T_final = transform_matmul(T_target_aug_frame, transform_matmul(T_aug, T_aug_frame_src))
                else:
                    # Only calculate transformation
                    T_final = self.get_transform(raw_dataset, data_cfg.frame, data_cfg.src.frame, device)

                # Apply transformations
                transform_kwargs = {}
                if data_cfg.type == DataType.POSE:
                    transform_kwargs["rotation_rep"] = data_cfg.rotation_rep
                    transform_kwargs["convention"] = data_cfg.convention
                elif data_cfg.type in [DataType.WRENCH, DataType.TWIST, DataType.TORQUE, DataType.LINEAR_VELOCITY]:
                    transform_kwargs["rotation_only"] = data_cfg.rotation_only

                data = getattr(projection_utils, f"apply_mat_to_{data_cfg.type.value}")(data, mat = T_final, **transform_kwargs)

            # Calculate relative data
            if data_cfg.relative_key is not None:
                if data_cfg.relative_key not in raw_dataset:
                     raise RuntimeError(f"Reference key '{data_cfg.relative_key}' not found in raw_dataset.")
                data = data - raw_dataset[data_cfg.relative_key]

            # Normalization
            results[out_name] = normalize(data, data_cfg.norm_config)
            
        return results


    def _reverse_process_configs(
        self,
        configs: Dict[str, Any],
        raw_dataset: CombinedDataset
    ) -> Dict[str, Any]:
        """ Reverse process configs. """
        results = {}

        for out_name, data_cfg in configs.items():
            # Retrieve data
            data = raw_dataset[data_cfg.src_key]

            # Unnormalize
            data = unnormalize(data, data_cfg.norm_config)

            # Unrelative
            if data_cfg.relative_key is not None:
                 if data_cfg.relative_key not in raw_dataset:
                     raise RuntimeError(f"Reference key '{data_cfg.relative_key}' not found in raw_dataset.")
                 data = data + raw_dataset[data_cfg.relative_key]

            # Unproject
            if data_cfg.type in SpatialDataTypes:
                device = data.device
                T = self.get_transform(raw_dataset, data_cfg.frame, data_cfg.src_frame, device = device)
                
                transform_kwargs = {}
                if data_cfg.type == DataType.POSE:
                    transform_kwargs["rotation_rep"] = data_cfg.src_rotation_rep
                    transform_kwargs["convention"] = data_cfg.src_convention
                elif data_cfg.type in [DataType.WRENCH, DataType.TWIST, DataType.TORQUE, DataType.LINEAR_VELOCITY]:
                    transform_kwargs["rotation_only"] = data_cfg.rotation_only
                
                data = getattr(projection_utils, f"apply_mat_to_{data_cfg.type.value}")(data, mat = T, **transform_kwargs)

            # Reverse Rotation Representation
            if data_cfg.type == DataType.POSE:
                data = xyz_rot_transform(
                    xyz_rot = data,
                    from_rep = data_cfg.src_rotation_rep,
                    to_rep = data_cfg.rotation_rep,
                    from_convention = data_cfg.src_convention,
                    to_convention = data_cfg.convention
                )

            results[out_name] = data
        
        return results


    def __call__(
        self,
        obs: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None,
        enable_aug: bool = False,
        process_type: Literal["forward", "backward"] = "forward"
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Process inputs/outputs for training/inference.
        """
        if process_type == "forward":
            is_inference = (action is None)
            raw_dataset = CombinedDataset(obs, action)
            aug_ctx = AugmentationContext() if enable_aug and not is_inference else None
            obs_results = self._process_configs(self.obs_data_configs, raw_dataset, enable_aug, aug_ctx)
            if not is_inference:
                action_results = self._process_configs(self.action_data_configs, raw_dataset, enable_aug, aug_ctx)
                return obs_results, action_results
            else:
                return obs_results
        else:
            raw_dataset = CombinedDataset(obs, action)
            action_results = self._reverse_process_configs(self.action_data_reverse_configs, raw_dataset)
            return action_results
    