# earth-vision
`Earth Vision` is a python library for solving computer vision tasks specifically for satellite imagery.

## Objective
To ease researcher to run ML pipelines for AI or Deep Learning Applications in solving Earth Observation (EO) tasks.

## Installation
We recommend Anaconda as Python package management system and using Python 3.9.

pip:
```
pip install earth-vision
conda install gdal
```

From source:
```
python setup.py install
conda install gdal
```
GDAL is actually a C++ library with python bindings. That means it relies on underlying C++ code and the package must be built/compiled in a certain manner to be usable with Python. So, we prefer to install it from Anaconda.

## Example
```python
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize
from earthvision.datasets import RESISC45

# Transformation
preprocess = Compose([ToTensor(), 
                      Normalize(mean=[0.3680, 0.3810, 0.3436], 
                                std=[0.1454, 0.1356, 0.1320])])

# Dataset and Dataloader
dataset = RESISC45(root='../dataset', transform=preprocess, download=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Features In Progress
- Pretrained model for `earthvision.datasets`

## Features Plans
Feel free to suggest features you would like to see by __opening an issue__.

- GPU memory optimization [TBD]
- High-level pipeline to integrate varied data sources [TBD]
