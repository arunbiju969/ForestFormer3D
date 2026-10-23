import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_mean

class RegionType:
    HYPER_CUBE = 0
    HYPER_CROSS = 1
    def __init__(self, val=0):
        self.val = val

class MinkowskiReLU(nn.ReLU):
    pass

class MinkowskiBatchNorm(nn.BatchNorm1d):
    pass

class MinkowskiInstanceNorm(nn.InstanceNorm1d):
    pass

class KernelGenerator:
    def __init__(self, *args, **kwargs):
        pass

class MinkowskiConvolution(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

class MinkowskiConvolutionTranspose(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

class SparseTensor:
    def __init__(self, features, coordinates, coordinate_map_key="map_key"):
        self.features = features
        self.coordinates = coordinates
        self.coordinate_map_key = coordinate_map_key

class TensorField:
    def __init__(self, features=None, coordinates=None, device=None):
        self.features = features
        self.coordinates = coordinates
        self._sparse_tensor = None
        self._inverse_mapping = None
        self.coordinate_map_key = "map_key"

    def sparse(self, quantization_mode=None):
        coords_int = torch.floor(self.coordinates).int()
        unique_coords, inverse_mapping = torch.unique(coords_int, dim=0, return_inverse=True)
        if self.features is not None:
            pooled_features = scatter_mean(self.features, inverse_mapping, dim=0)
        else:
            pooled_features = None
        self._inverse_mapping = inverse_mapping
        self._sparse_tensor = SparseTensor(
            features=pooled_features,
            coordinates=unique_coords,
            coordinate_map_key=self.coordinate_map_key
        )
        return self._sparse_tensor

    def inverse_mapping(self, coordinate_map_key=None):
        return self._inverse_mapping

class Utils:
    @staticmethod
    def batch_sparse_collate(data, device=None):
        batch_coords = []
        batch_feats = []
        for b, (coords, feats) in enumerate(data):
            if isinstance(coords, np.ndarray):
                coords = torch.from_numpy(coords)
            if isinstance(feats, np.ndarray):
                feats = torch.from_numpy(feats)
            
            b_col = torch.full((coords.shape[0], 1), b, dtype=coords.dtype, device=coords.device)
            b_coords = torch.cat([b_col, coords], dim=1)
            batch_coords.append(b_coords)
            batch_feats.append(feats)
        
        coordinates = torch.cat(batch_coords, dim=0)
        features = torch.cat(batch_feats, dim=0)
        return coordinates, features

utils = Utils()

class MinkowskiOpsModule:
    @staticmethod
    def cat(*tensors):
        return torch.cat(tensors)

MinkowskiOps = MinkowskiOpsModule()
