# earth-vision

Earth Vision is a python library for solving computer vision tasks specifically for satellite imagery.

## Objective

To ease researcher to run ML pipelines for AI or Deep Learning Applications in solving **Earth Observation** (EO) tasks.

## Examples

### 1. Dataset Download

```
from torch.utils.data import DataLoader
from earthvision.datasets import DeepSat

train_dataset = DeepSat(root='./', dataset_type='SAT-4', download=True, data_mode=0)
test_dataset = DeepSat(root='./', dataset_type='SAT-4', download=False, data_mode=1)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

train_data, train_label = next(iter(train_dataloader))
test_data, test_label = next(iter(test_dataloader))
```

## Features

1. Wrapper to download open sourced EO dataset for ML tasks:

- AerialCactus
- COWC
- DeepSat
- DroneDeploy
- EuroSat
- L8SPARCS
- LandCover
- RESISC45
- UCMercedLand


