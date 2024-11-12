"""
This module defines ResNet-based feature extractors for generating feature vectors from images. It includes classes for 
ImageNet, MTDP, and RetCCL pretrained models.
"""


# Local dependencies
from NET_CUP.utils import general_utils
from NET_CUP.data_loading.feature_type import FeatureType
import NET_CUP.datasources_config as datasources_config

# Other dependencies
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import numpy as np
from typing import Union
from abc import ABC, abstractmethod
from collections import OrderedDict


class FeatureExtractor(ABC):
    @property
    @abstractmethod
    def FEATURE_TYPE(self):
        """Defines the feature type for the extractor."""
        raise NotImplementedError("Subclasses must define FEATURE_TYPE.")

    @abstractmethod
    def get_vector(self, img: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        pass

    def __repr__(self) -> str:
        return repr(self.FEATURE_TYPE)


class ResNetFeatureExtractor(FeatureExtractor):
    """Base class for ResNet-based feature extractors."""

    def __init__(self, feature_type, model_weights=None):
        self._feature_type = feature_type

        # Load model and replace fully connected layer with Identity
        self._model = models.resnet50(weights=model_weights)
        self._model.fc = nn.Identity()

        # Set device and move model
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

        # Define image transforms
        self._scaler = transforms.Resize((224, 224))
        self._normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
        self._to_tensor = transforms.ToTensor()

    @property
    def FEATURE_TYPE(self):
        return self._feature_type

    def _prepare_image(self, img) -> torch.Tensor:
        """Convert an image to a suitable tensor for model input."""
        img = general_utils.convert_to_pil(img).convert('RGB')
        return Variable(self._normalize(self._to_tensor(self._scaler(img)))).unsqueeze(0)

    def get_vector(self, img) -> np.ndarray:
        """Extract feature vector from an image."""
        t_img = self._prepare_image(img).to(self._device)
        with torch.no_grad():
            tensor = self._model(t_img).cpu()
        return tensor.squeeze().numpy()


class ImageNetResnetFeatureExtractor(ResNetFeatureExtractor):
    def __init__(self):
        super().__init__(FeatureType.IMAGENET, model_weights='IMAGENET1K_V1')


class MtdpResnetFeatureExtractor(ResNetFeatureExtractor):
    def __init__(self):
        super().__init__(FeatureType.MTDP)

        # Load MTDP-specific weights
        pretext_model = torch.load(datasources_config.MTDP_WEIGHTS_PATH,
                                   map_location="cuda" if torch.cuda.is_available() else "cpu")

        # Filter and adjust keys to match structure
        new_model = OrderedDict((k[len('features.'):] if k.startswith('features.') else k, v)
                                for k, v in pretext_model.items() if not
                                k.startswith(('head', 'features.fc')))
        self._model.load_state_dict(new_model, strict=True)


class RetCCLResnetFeatureExtractor(ResNetFeatureExtractor):
    def __init__(self):
        super().__init__(FeatureType.RETCCL)

        # Load RETCCL-specific weights
        pretext_model = torch.load(datasources_config.RETCCL_WEIGHTS_PATH,
                                   map_location="cuda" if torch.cuda.is_available() else "cpu")
        self._model.load_state_dict(pretext_model, strict=True)