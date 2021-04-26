from jreo import dataset

try:
    l8biome = dataset.L8Biome()
    print("Your development environment has been setup properly")
except:
    print("Installation failed, please open an issue to let us know!")