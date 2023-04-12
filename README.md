# PointNet

## Installation
- [Pytorch](https://pytorch.org/get-started/locally/)
- [pyntcloud](https://pyntcloud.readthedocs.io/en/latest/installation.html)
- pandas
- ipywidgets
- threejs
- TODO: Add missing packages

## Datasets
- [ModelNet10](http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip)
- [ModelNet40](http://modelnet.cs.princeton.edu/ModelNet40.zip)

## Files
#### **dataset.py**
```python
# Load train/test dataset (3 X N)
n = 10000
train_data = ShapeNetDataset("datasets/ModelNet10", train=True, n=n)
test_data = ShapeNetDataset("datasets/ModelNet10", train=False, n=n)

# Create DataLoaders
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

# Getting points from point cloud
train_data[10][0].points # Is in Pandas DataFrame

# Visualizing PyntCloud (Only in notebook)
train_data[10][0].plot(backend='threejs')
```