"""Import specific class that will use."""
from .drone_deploy import DroneDeploy
from .aerialcactus import AerialCactus
from .eurosat import EuroSat
from .resisc45 import RESISC45
from .ucmercedland import UCMercedLand

__all__ = ['DroneDeploy', 'AerialCactus', 'RESISC45', 'UCMercedLand', 'EuroSat']