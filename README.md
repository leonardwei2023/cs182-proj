# PointNet
## About
This resource was created as a final project for UC Berkeley's Deep Learning course CS 182. We reimplemented PointNet and create a homework for other students to study it. You can start with the following materials: 
- Notebook 👉.
- Summary Document 👉.

## Installation
- [Pytorch](https://pytorch.org/get-started/locally/)
- [pyntcloud](https://pyntcloud.readthedocs.io/en/latest/installation.html)
- pandas
- ipywidgets
- threejs
- open3d
- TODO: Add missing packages

## Datasets
#### For classfication
- Student will implement this part and train this model.
- [ModelNet10](http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip)
- [ModelNet40](http://modelnet.cs.princeton.edu/ModelNet40.zip)
You can download the dataset with the following codes
```python
def load_data(opt):
    if not os.path.exists('./ModelNet10.zip'):
        subprocess.run(["wget", "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"])
    subprocess.run(["unzip", "-o", "ModelNet10.zip", "-d", "./datasets/"])
    if not os.path.exists('./ModelNet40.zip'):
        subprocess.run(["wget", "http://modelnet.cs.princeton.edu/ModelNet40.zip"])
    subprocess.run(["unzip", "-o", "ModelNet40.zip", "-d", "./datasets/"])
```

#### For Semantic Segmentation
- Student only need to construct the model structure, and then use pretrained model to do evaluation on test samples.
- You might need to fill the google form for access, and it might take 1 hour to download the dataset [S3DIS](http://buildingparser.stanford.edu/dataset.html). It is not necessary for you to download it.

#### For Part Segmentation
- You might need to register an account for access, and it takes 1 hour to download [ShapeNet](https://shapenet.org/download/shapenetcore). It is not necessary for you to download it.
- Students will mainly use PointNet for classfication, if you want to use it for part segmentation, feel free to use the above dataset to train your model.

## How to load 3D Data
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
train_data[10][0] # Torch Tensor [3,N]
```

#### visualize it
```python
# Visualizing PyntCloud
import open3d as o3d
points = (train_data[1000][0]).T # Torch Tensor [N,3]
o3d_cloud = o3d.geometry.PointCloud()
o3d_cloud.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([o3d_cloud])
```
![image](https://user-images.githubusercontent.com/106426767/235564933-aa714f97-18fc-4372-b94e-b3c885b37e85.png)


## Reference
```
@inproceedings{qi2017pointnet,
  title={Pointnet: Deep learning on point sets for 3d classification and segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={652--660},
  year={2017}
}
```
