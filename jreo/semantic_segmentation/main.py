import datasets

if __name__ == '__main__':
    dataset = 'dataset-sample'  #  0.5 GB download
    #dataset = 'dataset-medium' # 9.0 GB download
    
    datasets.download_dataset(dataset)