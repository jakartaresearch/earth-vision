import os
import requests
import yaml
import shutil
import random
from abc import ABC, abstractmethod

class Dataset(ABC):
    """Base class for dataset object
    """

    def __init__(self, dataset_id, **kwargs):
        self.dataset_id = dataset_id

    def download(self, out_dir, land_category=None, cloud_status=None,  shadows=False, n=1):
        """Function to download specific dataset.

        Args:
            n (int): The number of scenes to download (default: 1).
            out_dir (str): The output directory to store the downloaded dataset.
            land_category (list[str]): The land types, choose within [barren, forest, grass, shrubland, snow, urban, water, wetlands] (default: None).
            cloud_status (list[str]): The cloud status of the scene, choose within [random, clear, cloudy] (default: None).
            shadows (boolean): The presence of shadows in the scene (default: False)
        
        Returns:
            None
        """

        # Assertions and Type-checkings
        if self.dataset_id == "L8Biome": assert n > 0, "The number of files to be downloaded should be greater than 0"
        if self.dataset_id == "L8Sparcs": assert n == 1, "The number of file to be downloaded should be exactly 1"
        if self.dataset_id == "L7Irish": assert n > 0, "The number of file to be downloaded should be greater than 0"

        if land_category is None: land_category = []
        else: assert isinstance(land_category, list), "Make sure the input is in list"
        if cloud_status is None: cloud_status = []
        else: assert isinstance(cloud_status, list), "Make sure the input is in list"

        if not os.path.exists(f"{out_dir}"):
            os.mkdir(f"{out_dir}")
        
        current_path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{current_path}/constants/{self.dataset_id}/source.yaml") as stream:
            try:
                source_metadata = yaml.safe_load(stream)
                url = source_metadata["url"]
                del source_metadata["url"]

                if self.dataset_id == "L8Biome" or self.dataset_id == "L7Irish":
                    download_list = list(source_metadata.keys())

                    for scene, meta in source_metadata.items():
                        # Find based on land_category
                        if (len(land_category) > 0 and meta["category"] not in land_category) \
                            or (len(cloud_status) > 0 and meta["cloud_status"] not in cloud_status) \
                            or (meta["shadows"] != shadows):
                            try:
                                download_list.remove(scene)
                            except:
                                pass

            except yaml.YAMLError as exc:
                return exc

        if self.dataset_id == "L8Biome" or self.dataset_id == "L7Irish":
            while len(download_list) < n:
                download_list.append(random.choice(list(source_metadata.keys())))
                download_list = list(set(download_list))

            if len(download_list) > n:
                random.shuffle(download_list)
                download_list = download_list[0:n]

        if self.dataset_id == "L8Biome" or self.dataset_id == "L7Irish":
            download_urls = [url + file_name + ".tar.gz" for idx, file_name in enumerate(download_list)]
        elif self.dataset_id == "L8Sparcs":
            download_urls = [url]

        # Check if directory exists, if not create one
        if not os.path.exists(f"{out_dir}/{self.dataset_id}"):
            os.mkdir(f"{out_dir}/{self.dataset_id}")

        # Downloading in streams and chunks
        for idx, file_url in enumerate(download_urls):
            print(f"Downloading file {idx + 1} out of {len(download_urls)}...")
            root_output_dir = f"{out_dir}/{self.dataset_id}/"
            local_filename = f"{root_output_dir}/{file_url.split('/')[-1]}"
            
            with requests.get(file_url, stream=True) as r:
                with open(local_filename, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)

            # Unzip tar files
            print("Unzipping...")
            if self.dataset_id == "L8Biome" or self.dataset_id == "L7Irish":
                unzipped_dir = f"{local_filename}".replace(".tar.gz", "")
            elif self.dataset_id == "L8Sparcs":
                unzipped_dir = f"{local_filename}".replace(".zip", "")
            
            os.mkdir(unzipped_dir)
            shutil.unpack_archive(local_filename, f"{unzipped_dir}")

            # Remove the tar file
            print(f"Deleting {file_url.split('/')[-1]}...")
            os.remove(local_filename)


class L8Biome(Dataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_id="L8Biome")


class L8Sparcs(Dataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_id="L8Sparcs")

class L7Irish(Dataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_id="L7Irish")
