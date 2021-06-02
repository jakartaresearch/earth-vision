"""Import specific class that will use."""
from .drone_deploy import DroneDeploy
from .aerialcactus import AerialCactus
from .resisc45 import RESISC45
from .ucmercedland import UCMercedLand
from .deepsat import DeepSat

__all__ = ['DroneDeploy', 'AerialCactus',
           'RESISC45', 'UCMercedLand', 'DeepSat']
