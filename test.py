from earthvision import dataset

l8biome = dataset.L8Biome()
l8biome.download(n=1, land_category= ['barren', 'forest'], out_dir='./dataset')
