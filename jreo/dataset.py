import requests
import yaml
import random
from abc import ABC, abstractmethod

class Dataset(ABC):
    """Base class for dataset object
    """

    def __init__(self, dataset_id, **kwargs):
        self.dataset_id = dataset_id

    def download(self, n, out_dir, land_category=[], cloud_status=[],  shadows=False):
        """Function to download specific dataset.

        Args:
            n (int): The number of scenes to download.
            out_dir (str): The output directory to store the downloaded dataset.
            land_category (list[str]): The land types, choose within [barren, forest, grass, shrubland, snow, urban, water, wetlands] (default: []).
            cloud_status (list[str]): The cloud status of the scene, choose within [random, clear, cloudy] (default: []).
            shadows (boolean): The presence of shadows in the scene (default: False)
        
        Returns:
            None

        TODO:
            - Download request to the designated output directory
        """

        assert n > 0
        
        with open(f"./constants/{self.dataset_id}/source.yaml") as stream:
            try:
                source_metadata = yaml.safe_load(stream)
                url = source_metadata['url']
                del source_metadata['url']

                download_list = list(source_metadata.keys())

                for scene, meta in source_metadata.items():
                    # Find based on land_category
                    if (len(land_category) > 0 and meta['category'] not in land_category) \
                        or (len(cloud_status) > 0 and meta['cloud_status'] not in cloud_status) \
                        or (meta['shadows'] != shadows):
                        print(f"Remove: {scene}")
                        try:
                            download_list.remove(scene)
                        except:
                            pass

            except yaml.YAMLError as exc:
                return exc

        while len(download_list) < n:
            download_list.append(random.choice(list(source_metadata.keys())))
            download_list = list(set(download_list))

        if len(download_list) > n:
            random.shuffle(download_list)
            download_list = download_list[0:n]

        return download_list



class L8Biome(Dataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_id='L8Biome')
