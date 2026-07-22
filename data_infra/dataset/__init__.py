from data_infra.dataset.lowdim_dataset import LowdimPolicyDataset
from data_infra.dataset.vision_dataset import VisionPolicyDataset


def get_dataset(config):
    DatasetClass = VisionPolicyDataset if config.type == "vision" else LowdimPolicyDataset
    return DatasetClass(config)
