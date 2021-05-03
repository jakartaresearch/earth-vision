# Jakarta Vision
Jakarta AI Research (JAIR) Earth Observatory Library

## Objective
To ease dataset preparation for AI or Deep Learning analytics in solving __Earth Observation__ (EO) tasks.

## Examples
### 1. Dataset Download
```
from jreo import dataset

l8sparcs = dataset.L8Sparcs()

# Downloading (1 tifs downloaded at root directory dataset)
l8sparcs.download(n=1, out_dir='./dataset')
```


## Features done
1. Wrapper to download open sourced EO dataset for ML tasks:
- Landsat 8 Biome
- Landsat 8 SPARCS
- Landsat 7

## Features In Progress

1. Data pre-processors:
- Resizers
- Reflectance normalizers
- Dataset loader integration with popular ML frameworks

2. More benchmark open-sourced datasets
3. State-of-the-art EO models

## Features Plans
Feel free to suggest features you would like to see by __opening an issue__.
1. GPU memory optimization
2. High-level pipeline to integrate varied data sources

