# JREO (Jakarta Research Earth Observation)
Jakarta Research Earth Observatory - Library

## Objective
To ease dataset preparation for AI or Deep Learning analytics in solving __Earth Observation__ (EO) tasks.

## Examples
### 1. Dataset Download
```
from jreo import dataset

l8biome = dataset.L8Biome()

# Downloading (10 tifs downloaded at root directory dataset)
l8biome.download(n=10, out_dir='./dataset')
```


## Features In Progress
1. Wrapper to download open sourced EO dataset for ML tasks:
- Landsat 8 Biome
- Landsat 8 SPARCS
- Landsat 7
2. Data pre-processors:
- Resizers
- Reflectance normalizers
- Dataset loader integration with popular ML frameworks

## Features Plans
Feel free to suggest features you would like to see by __opening an issue__.
1. GPU memory optimization
2. High-level pipeline to integrate varied data sources

