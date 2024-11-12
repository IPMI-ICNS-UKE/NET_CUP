from enum import Enum

class FeatureType(Enum):
    """
    Enum class representing different pretrained ResNet backbones used for feature extraction.
    Each feature type corresponds to a specific pretrained ResNet model (Imagenet, MTDP, or RETCCL) 
    that is used extracting features vectors from an image.
    """
    IMAGENET = 0
    MTDP = 1
    RETCCL = 2

    def __repr__(self) -> str:
        if self.value == 0:
            return('IMAGENET_features')
        elif self.value == 1:
            return('MTDP_features')
        elif self.value == 2:
            return('RETCCL_features')
