
import torch
from dataset import ShapeNetDataset
from model import ClassificationNN, SegmentationNN, FeatureTransform, train
from torch.utils.data import random_split

import subprocess
import os
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default = 'cls', help = 'cls or seg')
    parser.add_argument('--dataset_root', type=str, default = './datasets/ModelNet10', help='dataset root dir')
    parser.add_argument('--epochs', type=int, default = 20, help='epochs')
    parser.add_argument('--batch_size', type=int, default = 32, help='batch size')
    parser.add_argument('--lr', type=float, default = 0.0001,  help='learning rate')
    parser.add_argument('--train_split', type=float, default = 0.7,  help='train/test split')
    parser.add_argument('--point_num', type=int, default = 1000, help = 'point num of point cloud')
    parser.add_argument('--cls_num', type=int, default = 10, help = 'class num')


    return parser.parse_args()

def load_data(path):
    if os.path.exists(path):
        pass
    else:
        if not os.path.exists("./datasets/"): os.makedirs("./datasets/")
        if path=='./datasets/ModelNet10':
            if not os.path.exists('./ModelNet10.zip'):
                print('downloading')
                process = subprocess.Popen(["wget", "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"])
                process.wait()
            subprocess.Popen(["unzip", "ModelNet10.zip", "-d", "./datasets/"])
        if path=='./datasets/ModelNet40':
            if not os.path.exists('./ModelNet40.zip'):
                process = subprocess.Popen(["wget", "http://modelnet.cs.princeton.edu/ModelNet40.zip"])
            process.wait()
            subprocess.Popen(["unzip", "-o", "ModelNet40.zip", "-d", "./datasets/"])

def prepare_dataset(opt):
    trainset = ShapeNetDataset(opt.dataset_root, train=True, n=opt.point_num)
    dataset_length = len(trainset)
    train_len = int(dataset_length * opt.train_split)
    test_len = dataset_length - train_len
    trainset, testset = random_split(trainset, [train_len, test_len])
    testset.train = False
    return trainset, testset

def train_pointNet_cls():
    opt = parse()
    load_data(opt.dataset_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset, valset = prepare_dataset(opt)
    classification_model = ClassificationNN(opt.cls_num).to(device)

    optimizer = torch.optim.Adam(classification_model.parameters(), lr=opt.lr)
    train(classification_model, trainset, valset, optimizer, epochs=opt.epochs, batch_size=opt.batch_size, device=device)

# def train_pointNet_seg():
#     opt = parse()
#     load_data(opt)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     trainset, valset = prepare_dataset(opt)
#     segmentation_model = SegmentationNN(opt.cls_num).to(device)

#     optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=opt.lr)
#     train(segmentation_model, trainset, valset, optimizer, epochs=opt.epochs, batch_size=opt.batch_size, device=device)

# def train_pointNet():
#     if 
if __name__=='__main__':
    train_pointNet_cls()