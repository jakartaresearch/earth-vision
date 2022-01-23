# How to use Vision Transformers Pre-trained Model RESISC45 Datasets

## Step 1: Install Huggingface Transformers
This pre-trained model using huggingface transformers, so you can install it first with following command:

**Install with pip**
```
pip install transformers
```     
**Install from source**
```    
pip install git+https://github.com/huggingface/transformers
```
**Install with Conda**
```
conda install -c huggingface transformers
```
<br>

## Step 2: Change Parameter in Vision Transformers Model
We will change the repository where our pre-trained models are stored.
By default if you using pre-trained model from ImageNet, the parameter of _ViTForImageClassification_ and _VitFeatureExtractor_ will be like this:
```
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
```
But since we already trained with RESISC45 datasets, the code should change like this:
```
feature_extractor = ViTFeatureExtractor.from_pretrained('adhisetiawan/vit-resisc45')
model = ViTForImageClassification.from_pretrained('adhisetiawan/vit-resisc45')
```
<br>

## Step 3: Let's do Inference with Full Code
```
import torch
import requests
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification

#Choose GPU if available 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Load image from url
url = 'https://drive.google.com/u/0/uc?id=1Rn8r9Mf9AadA_ChtUQOc5JVOD3dVPnQ4&export=download'
im = Image.open(requests.get(url, stream=True).raw)
display(im)

#Load pre-trained model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('adhisetiawan/vit-resisc45')
model = ViTForImageClassification.from_pretrained('adhisetiawan/vit-resisc45').to(device)

#Preprocess image
encoding = feature_extractor(images=im, return_tensors="pt")
pixel_values = encoding['pixel_values'].to(device)

#Forward Pass
outputs = model(pixel_values)
logits = outputs.logits

#Predict
prediction = logits.argmax(-1)
print("Predicted class:", model.config.id2label[prediction.item()])
```
