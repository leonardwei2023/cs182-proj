import os
import pandas as pd
from torch.utils.data import Dataset
from pyntcloud import PyntCloud
import subprocess

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
            dir_list = os.listdir(path)
            label_list = [label]*len(dir_list)
            df = pd.DataFrame(list(zip(dir_list, label_list)), columns=['path', 'label'])
            self.df = pd.concat((self.df, df))
        self.df.reset_index(inplace=True, drop=True)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        '''
        Returns (PyntCloud, label)
        '''
        path, label = self.df.loc[idx, 'path'], self.df.loc[idx, 'label']
        get_file = lambda p, t: PyntCloud.from_file(os.path.join(self.path, self.classes[label], t, path))
        test_train = 'train' if self.train else 'test'
        return get_file(path, test_train).get_sample('mesh_random', n=self.n, as_PyntCloud=True), label        

if __name__ == "__main__":
    train_dataset = ShapeNetDataset('datasets/ModelNet10', train=True)
    test_dataset = ShapeNetDataset('datasets/ModelNet10', train=False)

    print(len(train_dataset))
    print(len(test_dataset))

    print(train_dataset[10][0].points.boxplot())
    print(test_dataset[10][0].points.boxplot())
