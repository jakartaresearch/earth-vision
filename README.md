# earthvision

Earth Vision is a python library for solving computer vision tasks specifically for satellite imagery.

## Objective

To ease researcher to run ML pipelines for AI or Deep Learning Applications in solving **Earth Observation** (EO) tasks.

## Examples

### 1. Dataset Download

```
from earthvision import dataset

l8sparcs = dataset.L8Sparcs()

# Downloading (1 tifs downloaded at root directory dataset)
l8sparcs.download(n=1, out_dir='./dataset')
```

## Features

1. Wrapper to download open sourced EO dataset for ML tasks:

- Landsat 8 Biome
- Landsat 8 SPARCS
- Landsat 7
