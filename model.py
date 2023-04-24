import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pytorch_model_summary import summary

from dataset import BatchPyntCloudToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Sequential):
    def __init__(self, layer_sizes, dropout=0.7):
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
        super(MLP, self).__init__(*layers)
            

class SharedMLP(nn.Sequential):
    def __init__(self, layer_sizes):
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Conv1d(layer_sizes[i], layer_sizes[i+1], 1))
            layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            layers.append(nn.ReLU())
        super(SharedMLP, self).__init__(*layers)


class T_net(nn.Module):
    def __init__(self, size, dropout=0.7, bn_momentum=None):
        super(T_net, self).__init__()
        self.size = size

        self.shared_mlp = SharedMLP([size, 64, 128, 1024])

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, size*size, bias=False)
        self.fc3.requires_grad_(False)

        self.bn1 = nn.BatchNorm1d(512, momentum=bn_momentum)
        self.bn2 = nn.BatchNorm1d(256, momentum=bn_momentum)

    def forward(self, x):
        '''
            Input: B x size x N
        '''
        out = self.shared_mlp(x)
        out = F.max_pool1d(out, kernel_size=x.size(-1))
        out = out.view(-1, 1024)
        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.bn2(self.fc2(out)))

        out = self.fc3(out)
        out = out.view(-1, self.size, self.size)
        bias = torch.eye(self.size, requires_grad=True).expand(x.size(0), -1, -1).to(device)
        return out + bias


class InputTransform(nn.Module):
    def __init__(self):
        super(InputTransform, self).__init__()
        self.T_net = T_net(3)

    def forward(self, x):
        out = self.T_net(x)
        return torch.bmm(x.transpose(1, 2), out).transpose(1, 2)


class FeatureTransform(nn.Module):
    def __init__(self):
        super(FeatureTransform, self).__init__()
        self.T_net = T_net(64)

    def forward(self, x):
        out = self.T_net(x)
        self.A = out
        return torch.bmm(x.transpose(1, 2), out).transpose(1, 2)


class ClassificationNN(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationNN, self).__init__()
        self.input_transform = InputTransform()
        self.feature_transform = FeatureTransform()
        
        self.shared_mlp_1 = SharedMLP([3, 64, 64])
        self.shared_mlp_2 = SharedMLP([64, 64, 128, 1024])
        self.mlp = MLP([1024, 512, 256, num_classes])

    def forward(self, x):
        out = self.input_transform(x)
        out = self.shared_mlp_1(out)
        feat_out = self.feature_transform(out)
        out = self.shared_mlp_2(feat_out)
        out = F.max_pool1d(out, x.size(-1)).view(x.size(0), -1)
        out = self.mlp(out)
        return out, feat_out

    def predict(self, x):
        return self.forward(x)[0]


class SegmentationNN(nn.Module):
    def __init__(self, num_features: int):
        super(SegmentationNN, self).__init__()
        self.input_transform = InputTransform()
        self.feature_transform = FeatureTransform()

        self.mlp_1 = SharedMLP([3, 64, 64])
        self.mlp_2 = SharedMLP([64, 64, 128, 1024])
        self.mlp_3 = SharedMLP([1088, 512, 256, 128])
        self.mlp_4 = SharedMLP([128, 128, num_features])
        
    def forward(self, x):
        out = self.input_transform(x)
        out = self.mlp_1(out)
        feat_out = self.feature_transform(feat_out)
        global_feature = self.mlp_2(out)
        global_feature = F.max_pool1d(global_feature, x.size(2))
        global_feature = global_feature.expand(-1, -1, x.size(-1))
        out = torch.cat([out, global_feature], 1)
        out = self.mlp_3(out)
        out = self.mlp_4(out)
        return out, feat_out

    def predict(self, x):
        return self.forward(x)[0]


def loss_fn(preds, labels, feature_transform, reg=0.0001):
    loss = torch.nn.NLLLoss()
    def feat_loss(A):
        I = torch.eye(64, requires_grad=True).expand(A.size(0), -1, -1)
        AA_T = torch.bmm(A, A.transpose(1, 2))
        return torch.linalg.norm(I - AA_T, ord='fro', dim=(1,2))
    return loss(preds, labels) + reg * torch.mean(feat_loss(feature_transform))

def train(model, trainset, validset, optimizer, epochs=10, batch_size=32, device=torch.device('cpu')):
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=True)

    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []

    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        # Train and get training loss and accuracy
        train_loss, train_num_correct = [], []
        for x, y in tqdm(train_dataloader, unit='batch'):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred, feat = model(x)
            loss = loss_fn(pred, y, feat)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_num_correct.append(torch.sum(pred.argmax(1) == y).item())
        train_losses.append(np.mean(train_loss))
        train_accs = train_num_correct / len(trainset)

        model.eval()
        # Get validation loss and accuracy
        with torch.no_grad():
            valid_loss, valid_num_correct = [], []
            for x, y in tqdm(valid_dataloader, unit='batch'):
                x, y = x.to(device), y.to(device)
                pred, feat = model(x)
                loss = loss_fn(pred, y, feat)
                valid_loss.append(loss.item())
                valid_num_correct.append(torch.sum(pred.argmax(1) == y).item())
        
        print('Finished Epoch {}\n training loss: {}, validation loss: {} \n training accuracy: {}, validation accuracy: {}'
                .format(epoch+1, train_losses[-1], valid_losses[-1], train_accs[-1], valid_accs[-1]))

    return train_losses, valid_losses, train_accs, valid_accs


    

def plot_stats(epochs, train_losses, valid_losses, train_accs, valid_accs):
    epochs = range(0, self.params['epochs'] + 1)
    _, axs = plt.subplots(1, 2, layout='constrained', sharex=True, sharey=True)
    axs[0].plot(epochs, [0] + train_losses, label='training')
    axs[0].plot(epochs, [0] + valid_losses, label='validation')
    axs[0].title('Epochs vs Loss')
    axs[0].xlabel('Epochs')
    axs[0].ylabel('Loss')
    axs[0].legend()

    axs[1].plot(epochs, [0] + train_accs, label='training')
    axs[1].plot(epochs, [0] + valid_accs, label='validation')
    axs[1].title('Epochs vs Accuracy')
    axs[1].xlabel('Epochs')
    axs[1].ylabel('Accuracy')
    axs[1].legend()


if __name__ == '__main__':
    rand_data = Variable(torch.rand(32, 3, 2000)) # B X 3 X N
    shared_mlp = SharedMLP([3, 64, 64])
    out = shared_mlp(rand_data)
    print(summary(shared_mlp, rand_data), '\n', out.size()) # B X 64 X N

    rand_data = Variable(torch.rand(32, 3, 2000)) # B X 3 X N
    input_transform = InputTransform()
    out = input_transform(rand_data)
    print(summary(input_transform, rand_data), '\n', out.size()) # B X 3 X 3

    rand_data = Variable(torch.rand(32, 64, 2000)) # B X 64 X N
    input_transform = FeatureTransform()
    out = input_transform(rand_data)
    print(summary(input_transform, rand_data), '\n', out.size()) # B X 64 X 64

    rand_data = Variable(torch.rand(32, 3, 2000)) # B X 3 X N
    cls_net = ClassificationNN(num_classes=10)
    out = cls_net(rand_data)
    print(summary(cls_net, rand_data),' \n', out.size())

    rand_data = Variable(torch.rand(32, 3, 2000)) # B X 3 X N
    seg_net = SegmentationNN(num_features=10)
    out = seg_net(rand_data)
    print(summary(seg_net, rand_data), '\n', out.size())

    

    
