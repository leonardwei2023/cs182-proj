
import torch
from dataset import ShapeNetDataset
from model import ClassificationNN, SegmentationNN, FeatureTransform, train
from torch.utils.data import random_split

import subprocess
import os
import argparse

if os.path.exists('./datasets'):
    pass
else:
    if not os.path.exists('./ModelNet10.zip'):
        subprocess.run(["wget", "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"])
    subprocess.run(["unzip", "-o", "ModelNet10.zip", "-d", "./datasets/"])

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default = './datasets/ModelNet10', help='dataset root dir')
    parser.add_argument('--epochs', type=int, default = 20, help='epochs')
    parser.add_argument('--batch_size', type=int, default = 32, help='batch size')
    parser.add_argument('--lr', type=float, default = 0.0001,  help='learning rate')
    parser.add_argument('--train_split', type=float, default = 0.7,  help='train/test split')

    return parser.parse_args()

def train_pointNet_cls():
    opt = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    trainset = ShapeNetDataset(opt.dataset_root, train=True, n=10000)
    dataset_length = len(trainset)
    train_len = int(dataset_length * opt.train_split)
    test_len = dataset_length - train_len
    trainset, testset = random_split(trainset, [train_len, test_len])
    testset.train = False

    num_classes = 10
    classification_model = ClassificationNN(num_classes).to(device)

    optimizer = torch.optim.Adam(classification_model.parameters(), lr=opt.lr)
    train(classification_model, trainset, testset, optimizer, epochs=opt.epochs, batch_size=opt.batch_size, device=device)

if __name__=='__main__':
    train_pointNet_cls()