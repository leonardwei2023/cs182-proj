import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt

class ShapeNetDataset(Dataset):
    def __init__(self, root, train=True, n=10000):
        classes = [c for c in os.listdir(root) 
                   if os.path.isdir(os.path.join(root, c))]
        self.classes = {k: c for k, c in enumerate(classes)}
        self.path = root
        self.train = train
        self.n = n
        self.df = pd.DataFrame()
        for label, c in self.classes.items():
            path = None
            if self.train:
                path = os.path.join(self.path, c, "train")
            else:
                path = os.path.join(self.path, c, "test")
            dir_list = [dir for dir in os.listdir(path) if dir.endswith('.off')]
            label_list = [label]*len(dir_list)
            df = pd.DataFrame(list(zip(dir_list, label_list)), columns=['path', 'label'])
            self.df = pd.concat((self.df, df))
        self.df.reset_index(inplace=True, drop=True)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        '''
        Returns (3 X N, label)
        '''
        path, label = self.df.loc[idx, 'path'], self.df.loc[idx, 'label']
        get_file = lambda p, t: PyntCloud.from_file(os.path.join(self.path, self.classes[label], t, p))
        test_train = 'train' if self.train else 'test'
        pointcloud = get_file(path, test_train).get_sample('mesh_random', n=self.n)
        return torch.Tensor(pointcloud.values).transpose(0,1), label  

def BatchPyntCloudToTensor(pyntcloud):
    # B x PyntCloud(N X 3) -> B x 3 X N 
    pointcloud = pyntcloud.points.values
    return torch.Tensor(pointcloud).transpose(1, 2)

# class S3DISDataset(Dataset): /datautils
#     # you can download dataset through this google form http://buildingparser.stanford.edu/dataset.html
# and here we might use the dataloader with reference from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/S3DISDataLoader.py
# /datautils need to be modified.

if __name__ == "__main__":

    #create tran/test split
    print("Creating Test/Train Split")
    train_dataset = ShapeNetDataset('datasets/ModelNet10', train=True)
    test_dataset = ShapeNetDataset('datasets/ModelNet10', train=False)
    print("Finished Test/Train Split")
    print("Train Dataset Size:", len(train_dataset))
    print("Train Dataset Size:", len(test_dataset))
    print()

    #plot histogram of dataset
    train_class_counts = train_dataset.df['label'].value_counts()
    test_class_counts = test_dataset.df['label'].value_counts()

    x_labels = [train_dataset.classes[label] for label in train_class_counts.index]
    x = range(len(x_labels))

    plt.bar(x, train_class_counts, label='train', width=0.4)
    plt.bar([xi + 0.4 for xi in range(len(x_labels))], test_class_counts, label='test', width=0.4)
    plt.xticks([xi for xi in x], x_labels, rotation=45)
    plt.ylabel('Number of Samples')
    plt.title('Train/Test Dataset Contents')
    plt.legend()
    
    #example showing representation of monitor sample
    for key, value in train_dataset.classes.items():
        if value == "monitor":
            label = key
            break

    idx = train_dataset.df[train_dataset.df['label'] == label].index[0]
    point_cloud = train_dataset[idx]
    print(f"Class: monitor")
    print("example: first point xyz coords")
    print(point_cloud[0].points.iloc[0])
    print()
    
    #show plot
    # plt.show()
